import os
import pickle

import numpy as np
import pandas as pd


def load_object(filepath):
    """加载通过pickle保存的对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已加载: {filepath}")
    return obj


class ModelPipelineWrapper:
    """
    将训练脚本中保存的各个组件封装成一个类似sklearn Pipeline的预测接口。
    """

    def __init__(self, model_dict):
        self.imputer = model_dict["imputer"]
        self.scaler = model_dict["scaler"]
        self.variance_selector = model_dict["variance_selector"]
        self.brf_feature_selector = model_dict["brf_feature_selector"]
        self.dnn_feature_ranking = model_dict["dnn_feature_ranking"]
        self.selected_feature_names = model_dict["selected_feature_names"]

        # 假设 base_models 是一个包含单一模型的列表
        # 根据训练脚本，这里应该是单个模型
        self.base_model = model_dict["base_models"]
        self.meta_model = model_dict["meta_nb"]  # 在训练脚本中已经是 LogisticRegression
        self.threshold = model_dict["threshold"]

    def _preprocess(self, X_raw):
        """
        对原始特征矩阵执行完整的预处理流水线。
        """
        # 1. 方差过滤 (需要与训练时相同的特征顺序)
        X_var_filtered = self.variance_selector.transform(X_raw)

        # 2. 缺失值填充和标准化
        X_imputed = self.imputer.transform(X_var_filtered)
        X_scaled = self.scaler.transform(X_imputed)

        # 3. BRF特征选择
        X_brf_selected = self.brf_feature_selector.transform(X_scaled)

        # 4. DNN特征排名筛选 (根据训练时保存的排名索引进行选择)
        # 注意：ranked_idx_by_importance 是对 BRF 选出特征的重要性排序
        # 我们需要选择前 N 个最重要的特征，N 由 selected_feature_names 的长度决定
        # 或者更准确地，使用 ranking 索引来选择对应列
        # 训练时是 selected_top_feature_indices = ranked_idx_by_importance[:num_top_features]
        # 所以我们也需要取 ranking 的前 len(selected_feature_names) 项作为列索引
        num_final_features = len(self.selected_feature_names)
        selected_top_indices_from_brf = self.dnn_feature_ranking[:num_final_features]

        # 确保索引不越界（理论上不应该）
        if np.max(selected_top_indices_from_brf) >= X_brf_selected.shape[1]:
            raise IndexError("DNN feature ranking indices are out of bounds for BRF selected features.")

        X_final_selected = X_brf_selected[:, selected_top_indices_from_brf]

        return X_final_selected

    def predict_proba(self, X_raw):
        """
        对原始特征数据进行预处理并返回基模型的概率预测。
        注意：此函数直接返回基模型的概率，级联逻辑在predict中处理阈值，
        但为了与旧代码兼容，我们在这里也应用级联逻辑的一部分（获取基础概率）。
        实际上，应将 cascade_predict_single_model 逻辑整合进来。
        为保持一致性，我们模仿其行为：只返回基模型概率。
        """
        X_processed = self._preprocess(X_raw)
        # 调用基模型的 predict_proba
        try:
            probas_base = self.base_model.predict_proba(X_processed)[:, 1]
        except Exception:
            # 如果没有 predict_proba，尝试用 predict 并转换
            preds_base = self.base_model.predict(X_processed)
            # 这里简化处理，假设输出是 0/1，转为概率近似 (0.01, 0.99)
            probas_base = np.where(preds_base == 1, 0.99, 0.01)
        return np.column_stack([1 - probas_base, probas_base])

    def predict(self, X_raw):
        """
        对原始特征数据进行预处理、预测，并根据阈值和级联模型返回最终预测。
        """
        X_processed = self._preprocess(X_raw)

        # 导入 cascade_predict_single_model 函数中的核心逻辑
        # 因为它不是全局可导入的，我们需要在这里重现实现关键部分
        # 或者更好的方式是在此类中定义一个相似的方法

        # --- 重现 cascade_predict_single_model 逻辑 ---
        try:
            probas_base = self.base_model.predict_proba(X_processed)[:, 1]
        except Exception:
            probas_base = self.base_model.predict(X_processed).astype(float)

        # 使用类中存储的阈值
        preds = (probas_base >= self.threshold).astype(int)

        # 简化的不确定性判断
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        # 使用 meta_model 进行修正
        if self.meta_model is not None and uncertain_mask.sum() > 0:
            meta_preds = self.meta_model.predict(X_processed[uncertain_mask])
            preds[uncertain_mask] = meta_preds

        return preds
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 引入进度条库

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


class DeepFeatureSelector(nn.Module):
    """更深的全连接网络，不含自注意力机制"""

    def __init__(self, input_dim):
        super().__init__()
        # 使用线性层构建网络，修改为1024-512-256-128-64结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=1500, batch_size=256, lr=1e-3, device="cpu", patience=50):
    """
    训练带有早停和学习率调度的深度特征选择器

    参数:
        X : array-like, shape (n_samples, n_features)
            输入特征.
        y : array-like, shape (n_samples,)
            目标变量.
        input_dim : int
            输入特征维度.
        epochs : int, optional (default=1000)
            最大训练轮数.
        batch_size : int, optional (default=128)
            批次大小.
        lr : float, optional (default=1e-3)
            初始学习率.
        device : str or torch.device, optional (default="cpu")
            计算设备 ("cpu" 或 "cuda").
        patience : int, optional (default=20)
            早停耐心值.

    返回:
        feature_importance : ndarray, shape (input_dim,)
            特征重要性分数.
    """
    # 确保模型在指定设备上
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()

    # 使用 AdamW 优化器，通常比 Adam 更好
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 添加学习率调度器，在验证损失停滞时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2
                                                     )

    # 将数据也移动到指定设备
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化早停相关变量
    best_loss = float('inf')
    trigger_times = 0
    early_stop = False

    model.train()

    # 使用tqdm显示训练进度条
    pbar = tqdm(range(epochs), desc="Training DNN Feature Selector")

    for epoch in pbar:
        total_loss = 0.0
        num_batches = 0

        for xb, yb in loader:
            yb = yb.unsqueeze(1)  # 调整标签形状
            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            # 添加梯度裁剪防止爆炸梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # 更新学习率调度器
        scheduler.step(avg_loss)

        # 更新进度条信息
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'LR': f'{current_lr:.2e}'})

        # --- Early Stopping Logic ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            # 在这里可以保存最佳模型状态字典，但因为我们只需要最终权重来计算特征重要性，
            # 并且不会恢复到这个检查点，所以省略了保存步骤。
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    # 训练结束后关闭进度条
    pbar.close()

    # 提取第一层权重以计算特征重要性
    first_linear_layer = None
    if hasattr(model, 'network') and len(list(model.network)) > 0:
        # 获取第一个 Linear 层
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                first_linear_layer = layer
                break
    else:
        # 如果直接是 Sequential 或者其他情况，尝试查找第一个 nn.Linear 层
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear_layer = module
                break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(output_dim_of_first_layer, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0)
    else:
        # 如果找不到线性层，则返回零重要性（理论上不应发生）
        print("警告：无法找到第一层线性层以计算特征重要性。")
        feature_importance = np.zeros(input_dim)

    return feature_importance


def f1_2_threshold(y_true, probas):
    """
    基于 F1.2 分数寻找最优阈值。
    F1.2 更重视召回率 (Recall)。
    """
    thresholds = np.linspace(0, 1, 101)
    best_t, best_f1_2 = 0.5, -1
    beta = 1.2  # F1.2 的 beta 值

    for t in thresholds:
        preds = (probas >= t).astype(int)
        # 计算 F1.2 分数
        # 注意：fbeta_score 在某些极端情况下（如所有预测为一类）可能返回 0.0
        # 我们接受这种行为，因为它反映了模型在该阈值下的性能不佳
        try:
            f1_2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        except ValueError:
            # 如果 y_true 或 preds 只有一个类别，fbeta_score 会报错
            # 在这种情况下，我们将其视为 F1.2 = 0
            f1_2 = 0.0

        if f1_2 > best_f1_2:
            best_f1_2, best_t = f1_2, t
    return best_t


def custom_resample(X, y, pos_ratio=1.4, neg_ratio=0.6, random_state=None):
    """
    根据指定的比例对数据进行重采样。
    pos_ratio: 正类相对于原始正类数量的倍数 (例如 1.4 表示变为140%)
    neg_ratio: 负类相对于原始负类数量的倍数 (例如 0.6 表示变为60%)
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError("仅支持二分类问题")
    neg_class, pos_class = unique_classes[0], unique_classes[1]
    neg_count, pos_count = class_counts[0], class_counts[1]

    print(f"原始数据分布: 负类({neg_class})={neg_count}, 正类({pos_class})={pos_count}")

    neg_indices = np.where(y == neg_class)[0]
    pos_indices = np.where(y == pos_class)[0]

    target_neg_count = int(neg_count * neg_ratio)
    target_pos_count = int(pos_count * pos_ratio)

    print(f"目标采样后分布: 负类={target_neg_count}, 正类={target_pos_count}")

    # 下采样负类
    if target_neg_count < neg_count:
        sampled_neg_indices = np.random.choice(neg_indices, size=target_neg_count, replace=False)
    else:  # 上采样负类
        sampled_neg_indices = np.random.choice(neg_indices, size=target_neg_count, replace=True)

    # 上采样正类
    if target_pos_count < pos_count:
        sampled_pos_indices = np.random.choice(pos_indices, size=target_pos_count, replace=False)
    else:  # 上采样正类
        sampled_pos_indices = np.random.choice(pos_indices, size=target_pos_count, replace=True)

    # 合并并打乱
    combined_indices = np.concatenate([sampled_neg_indices, sampled_pos_indices])
    np.random.shuffle(combined_indices)

    X_resampled = X[combined_indices]
    y_resampled = y[combined_indices]

    resampled_neg_count = np.sum(y_resampled == neg_class)
    resampled_pos_count = np.sum(y_resampled == pos_class)
    print(f"重采样后实际分布: 负类={resampled_neg_count}, 正类={resampled_pos_count}")

    return X_resampled, y_resampled
class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)

            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)

if __name__ == "__main__":
    # -----------------------------
    # 1. 加载测试数据
    # -----------------------------
    test_data_path = "testClean.csv"
    print(f"正在加载测试数据: {test_data_path}")
    if not os.path.exists(test_data_path):
        # 创建示例测试数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到测试数据文件 '{test_data_path}'。正在创建示例测试数据...")
        np.random.seed(100)  # 固定种子以便示例一致
        sample_n = 200
        sample_test_data = pd.DataFrame({
            'f0': np.random.randn(sample_n),
            'f1': np.random.rand(sample_n) * 100,
            'f2': np.random.randn(sample_n) * 5,
            # 添加一些零方差特征列名，即使数据不同，只要名字匹配即可被过滤
            'zero_var_0': [5.0] * sample_n,
            'zero_var_1': [5.0] * sample_n,
            # 模拟 one-hot 编码后的类别特征
            'category_A_Y': np.random.choice([0, 1], size=sample_n, p=[0.7, 0.3]),

        })
        # 确保列名与训练集（包括经过get_dummies后）可能存在的列名一致
        # 这里的列名是基于训练脚本中生成的示例数据的
        # 实际使用时，应确保测试集列名与训练集处理后一致

        # 添加 company_id 列
        sample_test_data.insert(0, 'company_id', [f"COMP_{i}" for i in range(len(sample_test_data))])

        sample_test_data.to_csv(test_data_path, index=False)
        print(f"已生成示例测试数据并保存至 {test_data_path}")

    test_data = pd.read_csv(test_data_path)
    print(
        f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1] - 1 if 'company_id' in test_data.columns else test_data.shape[1]}")

    # -----------------------------
    # 2. 加载模型字典并包装
    # -----------------------------
    model_path = "./model.pkl"  # 使用训练脚本最后保存的模型文件
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。请确保模型已训练并保存。")
        exit(1)

    print(f"正在加载模型字典: {model_path}")
    loaded_model_dict = load_object(model_path)

    # 包装模型字典以便使用
    loaded_pipeline = ModelPipelineWrapper(loaded_model_dict)

    # -----------------------------
    # 3. 读取阈值
    # -----------------------------
    threshold = loaded_pipeline.threshold
    print(f"从模型对象读取阈值: {threshold:.4f}")

    # -----------------------------
    # 4. 测试数据处理与预测
    # -----------------------------
    # 去掉 ID 列
    feature_columns = [col for col in test_data.columns if col != "company_id"]
    X_test_raw = test_data[feature_columns].values  # 转换为 NumPy array
    print(f"用于预测的原始特征数: {X_test_raw.shape[1]}")

    print("正在进行预测...")
    # 获取概率 (注意 predict_proba 返回的是二维数组 [[neg_prob, pos_prob], ...])
    y_proba_full = loaded_pipeline.predict_proba(X_test_raw)
    y_proba = y_proba_full[:, 1]  # 获取正类概率
    print(f"预测完成，共 {len(y_proba)} 个概率值。")

    print(f"应用分类阈值: {threshold:.4f}")
    # 获取最终预测结果 (0 或 1)
    y_pred = loaded_pipeline.predict(X_test_raw)
    print("分类完成。")

    # -----------------------------
    # 5. 结果输出到桌面
    # -----------------------------
    if "company_id" in test_data.columns:
        uuid_column = test_data["company_id"]
    else:
        print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
        uuid_column = test_data.index.map(str)  # 确保索引是字符串类型

    results_df = pd.DataFrame({
        "uuid": uuid_column,
        "proba": y_proba,
        "prediction": y_pred
    })
    print("结果数据框创建成功。")

    # 构造桌面路径 (跨平台基本兼容的方式)
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop_path):
        desktop_path = "."  # 如果桌面路径不存在，则保存到当前目录

    output_filename = "submit_template.csv"
    output_path = os.path.join(desktop_path, output_filename)

    print(f"正在保存结果到: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"预测完成，阈值 {threshold:.4f}，结果已保存到 {output_path}")

