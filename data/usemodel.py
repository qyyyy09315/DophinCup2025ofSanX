import os
import pickle
import warnings

import numpy as np
import pandas as pd
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.metrics import confusion_matrix

# 导入 XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    print("警告: 未找到 xgboost 库。请运行 'pip install xgboost' 进行安装。")
    XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """
    修改版cascade_predict，适用于单个基模型。
    修复了当meta_model为None时的错误处理
    """
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        probas_base = base_model.predict(X).astype(float)

    preds = (probas_base >= threshold).astype(int)

    # 简化的不确定性判断：这里我们假设当概率接近0.5时不确定
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)  # 可调整阈值

    # 只有当meta_model不为None且存在不确定样本时才使用meta_model
    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


# 定义一个带残差连接的块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # 主路径
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 残差连接路径 (如果输入输出维度不同)
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.relu1(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.linear2(out)
        # 残差连接
        out += identity
        out = self.relu2(out)
        if self.dropout2:
            out = self.dropout2(out)

        return out


class DeepFeatureSelector(nn.Module):
    """更深的全连接网络，包含残差连接"""

    def __init__(self, input_dim):
        super().__init__()
        # 使用残差块构建网络
        self.block1 = ResidualBlock(input_dim, 1024, 0.4)
        self.block2 = ResidualBlock(1024, 512, 0.3)
        self.block3 = ResidualBlock(512, 256, 0.3)
        self.block4 = ResidualBlock(256, 128, 0.2)

        # 输出层
        self.output_layer = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=200, batch_size=128, lr=1e-3, device="cpu"):
    # 确保模型在指定设备上
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()
    # 可以考虑使用 weight_decay 进行 L2 正则化
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将数据也移动到指定设备
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for xb, yb in loader:
            # 数据已经在设备上了，无需再次移动
            # xb, yb = xb.to(device), yb.to(device)
            yb = yb.unsqueeze(1)  # 调整标签形状
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        # 可选：打印平均损失
        # if (epoch + 1) % 50 == 0:
        #     avg_loss = total_loss / num_batches
        #     print(f"Epoch {epoch+1}/{epochs}, Avg Loss={avg_loss:.4f}")

    # 提取第一层权重以计算特征重要性
    # 注意：现在第一层是 ResidualBlock，我们需要访问其内部的 linear1 层
    first_linear_layer = None
    if hasattr(model, 'block1') and hasattr(model.block1, 'linear1'):
        first_linear_layer = model.block1.linear1
    else:
        # 如果结构改变，尝试查找第一个 nn.Linear 层
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear_layer = module
                break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(1024, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0)
    else:
        # 如果找不到线性层，则返回零重要性（理论上不应发生）
        print("警告：无法找到第一层线性层以计算特征重要性。")
        feature_importance = np.zeros(input_dim)

    return feature_importance


def youden_threshold(y_true, probas):
    thresholds = np.linspace(0, 1, 101)
    best_t, best_j = 0.5, -1
    for t in thresholds:
        preds = (probas >= t).astype(int)
        # 处理只有一个类别的情况
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # 所有预测都是同一类
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # All predicted negative
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # All predicted positive
        else:
            # 不规则矩阵，填充缺失项
            padded_cm = np.pad(cm, ((0, 2 - cm.shape[0]), (0, 2 - cm.shape[1])), mode='constant')
            tn, fp, fn, tp = padded_cm.ravel()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        J = sensitivity + specificity - 1
        if J > best_j:
            best_j, best_t = J, t
    return best_t
class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            # 返回二维数组 [[prob_0, prob_1], ...]
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)

def load_object(filepath):
    """加载通过pickle保存的对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已加载: {filepath}")
    return obj


class LoadedPipelineWrapper:
    """
    包装加载的模型字典，使其行为类似于sklearn Pipeline，
    并处理内部的XGBoost模型和级联逻辑。
    """

    def __init__(self, model_dict):
        self.model_dict = model_dict
        # 从字典中提取各个组件
        self.imputer = self.model_dict["imputer"]
        self.scaler = self.model_dict["scaler"]
        self.dnn_feature_ranking = self.model_dict["dnn_feature_ranking"]

        # 基模型是 WrappedXGBModel 实例
        self.base_model = self.model_dict["base_models"]
        # 元模型是 GaussianNB 实例
        self.meta_model = self.model_dict["meta_nb"]
        # 阈值
        self.threshold = self.model_dict.get("threshold", 0.5)

    def _preprocess(self, X_raw):
        """执行与训练时相同的预处理步骤"""
        # 1. 填充缺失值
        X_imp = self.imputer.transform(X_raw)
        # 2. 标准化
        X_sc = self.scaler.transform(X_imp)
        # 3. 特征选择（根据DNN排名重排特征）
        X_selected = X_sc[:, self.dnn_feature_ranking]
        return X_selected

    def predict_proba(self, X_raw):
        """
        对原始特征数据进行预处理并预测概率。
        实现与训练时一致的cascade_predict_single_model逻辑。
        """
        X_processed = self._preprocess(X_raw)

        # --- 级联预测逻辑 ---
        try:
            probas_base = self.base_model.predict_proba(X_processed)[:, 1]
        except Exception:
            probas_base = self.base_model.predict(X_processed).astype(float)

        # 简化的不确定性判断
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        # 初始化最终预测概率为基模型概率
        final_probas = probas_base.copy()

        # 如果有不确定样本且元模型存在，则使用元模型修正
        if uncertain_mask.sum() > 0:
            # 注意：这里我们只返回主类的概率，因为这是后续阈值判断所需要的
            # cascade_predict_single_model 返回的是 (preds, probas_base)
            # 我们需要模拟其对不确定样本使用meta_model预测的部分
            # 但 meta_model (GaussianNB) 的 predict_proba 会返回两个类别的概率
            meta_probas = self.meta_model.predict_proba(X_processed[uncertain_mask])
            # 假设 meta_probas 是 [[p0_class0, p0_class1], [p1_class0, p1_class1], ...]
            # 我们取 class1 的概率来替换原来的基模型概率
            final_probas[uncertain_mask] = meta_probas[:, 1]

        # 为了兼容接口，返回二维数组 [[prob_0, prob_1], ...]
        # 这里简化处理，因为我们主要关心 prob_1 (final_probas)
        prob_0 = 1 - final_probas
        return np.vstack([prob_0, final_probas]).T

    def predict(self, X_raw):
        """
        对原始特征数据进行预处理并预测类别。
        使用存储在模型中的阈值。
        """
        probas = self.predict_proba(X_raw)
        preds = (probas[:, 1] >= self.threshold).astype(int)
        return preds


def load_model(model_path):
    """加载整个模型管道字典，并用包装器包装"""
    model_dict = load_object(model_path)
    return LoadedPipelineWrapper(model_dict)


if __name__ == "__main__":
    # -----------------------------
    # 1. 加载测试数据
    # -----------------------------
    test_data_path = "testClean.csv"
    print(f"正在加载测试数据: {test_data_path}")
    if not os.path.exists(test_data_path):
        # 创建示例测试数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到测试数据文件 '{test_data_path}'。正在创建示例测试数据...")
        sample_test_data = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.rand(200) * 100,
            'category_A': ['Y'] * 60 + ['N'] * 140,
            # target列通常在测试集中不存在，这里仅作占位符展示，不会参与预测
        })
        sample_test_data = pd.get_dummies(sample_test_data, drop_first=True)
        # 确保测试集特征与训练集对齐（假设训练集有feature1, feature2, category_A_Y）
        required_cols = ['feature1', 'feature2', 'category_A_Y']
        for col in required_cols:
            if col not in sample_test_data.columns:
                # 添加缺失的列，默认填0
                sample_test_data[col] = 0
        sample_test_data = sample_test_data[required_cols]  # 重新排序以匹配预期顺序

        # 添加 company_id 列
        sample_test_data.insert(0, 'company_id', [f"COMP_{i}" for i in range(len(sample_test_data))])

        sample_test_data.to_csv(test_data_path, index=False)
        print(f"已生成示例测试数据并保存至 {test_data_path}")

    test_data = pd.read_csv(test_data_path)
    print(
        f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1] - 1 if 'company_id' in test_data.columns else test_data.shape[1]}")

    # -----------------------------
    # 2. 加载模型管道 (包括预处理器)
    # -----------------------------
    model_path = "./model_pipeline_cascade_dnn_torch.pkl"  # 确保此路径正确
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。请确保模型已训练并保存。")
        exit(1)

    print(f"正在加载模型管道: {model_path}")
    loaded_pipeline = load_model(model_path)

    # -----------------------------
    # 3. 读取阈值
    # -----------------------------
    threshold = loaded_pipeline.threshold
    print(f"从模型对象读取阈值: {threshold}")

    # -----------------------------
    # 4. 测试数据处理
    # -----------------------------
    # 去掉 ID 列
    feature_columns = [col for col in test_data.columns if col != "company_id"]
    X_test_raw = test_data[feature_columns]
    print(f"用于预测的原始特征数: {X_test_raw.shape[1]}")

    # 预处理已在 pipeline 内部的 predict_proba/predict 方法中完成
    # X_test_proc = X_test_raw.copy()
    # if imputer is not None:
    #     X_test_proc = imputer.transform(X_test_proc)
    #     print("已应用缺失值填充。")
    #
    # if scaler is not None:
    #     X_test_proc = scaler.transform(X_test_proc)
    #     print("已应用标准化。")
    #
    # if selector is not None:
    #     X_test_proc = selector.transform(X_test_proc)
    #     print(f"已应用特征选择，最终特征数: {X_test_proc.shape[1]}")

    # -----------------------------
    # 5. 预测
    # -----------------------------
    print("正在进行预测...")
    # 使用包装器的 predict_proba 方法，它会自动处理预处理和级联逻辑
    y_proba_full = loaded_pipeline.predict_proba(X_test_raw.values)
    y_proba = y_proba_full[:, 1]  # 获取正类概率
    print(f"预测完成，共 {len(y_proba)} 个概率值。")

    print(f"应用分类阈值: {threshold}")
    # 使用包装器的 predict 方法，它会自动处理预处理、级联逻辑和阈值判断
    y_pred = loaded_pipeline.predict(X_test_raw.values)
    print("分类完成。")

    # -----------------------------
    # 6. 结果输出
    # -----------------------------
    if "company_id" in test_data.columns:
        uuid_column = test_data["company_id"]
    else:
        print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
        uuid_column = test_data.index.astype(str)  # 确保索引是字符串类型

    results_df = pd.DataFrame({
        "uuid": uuid_column,
        "proba": y_proba,
        "prediction": y_pred
    })
    print("结果数据框创建成功。")

    output_path = "submit_predictions.csv"
    print(f"正在保存结果到: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"预测完成，阈值 {threshold}，结果已保存到 {output_path}")




