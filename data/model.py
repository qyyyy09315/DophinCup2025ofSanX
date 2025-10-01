import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as ImbPipeline  # 避免命名冲突
# 注意：如果未安装 imblearn，请运行 'pip install imbalanced-learn'
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 导入 XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    print("警告: 未找到 xgboost 库。请运行 'pip install xgboost'。")
    XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    """将对象保存到文件"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """
    为单个基础模型修改的 cascade_predict。
    修复了 meta_model 为 None 时的错误处理。
    """
    try:
        # 尝试获取预测概率
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        # 如果 predict_proba 不可用，则回退到 predict 并转换为浮点数
        probas_base = base_model.predict(X).astype(float)

    # 根据阈值生成初步预测
    preds = (probas_base >= threshold).astype(int)

    # 简化的不确定性判断：假设概率接近 0.5 时为不确定
    # 可以根据具体需求调整此逻辑。
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)  # 阈值可调

    # 仅当 meta_model 不为 None 且存在不确定样本时才使用 meta_model
    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


# 定义带有残差连接的块
class ResidualBlock(nn.Module):
    """残差块，包含跳跃连接"""
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # 主路径
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu1 = nn.ReLU()
        # 如果 dropout_rate > 0，则添加 Dropout 层
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 捷径连接路径（如果输入/输出维度不同）
        self.shortcut = nn.Identity() # 默认为恒等映射
        if in_features != out_features:
            # 如果维度不同，使用线性层进行维度匹配
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 保存输入作为捷径连接
        identity = self.shortcut(x)

        # 主路径计算
        out = self.linear1(x)
        out = self.relu1(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.linear2(out)
        # 添加残差连接
        out += identity
        out = self.relu2(out)
        if self.dropout2:
            out = self.dropout2(out)

        return out


class DeepFeatureSelector(nn.Module):
    """更深的全连接网络，带有残差连接，用于特征选择。"""

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
        # 依次通过各个残差块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # 最终输出
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=200, batch_size=128, lr=1e-3, device="cpu"):
    """
    使用 PyTorch 训练深度特征选择器。
    """
    # 确保模型在指定设备上
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss() # 二元交叉熵损失
    # 考虑使用权重衰减 (weight_decay) 进行 L2 正则化
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将数据移动到指定设备
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
            # 数据已在此前移动到设备，无需再次移动
            # xb, yb = xb.to(device), yb.to(device)
            yb = yb.unsqueeze(1)  # 调整标签形状以匹配输出
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
    # 注意：第一层现在是一个 ResidualBlock；我们需要访问其内部的 linear1 层
    first_linear_layer = None
    if hasattr(model, 'block1') and hasattr(model.block1, 'linear1'):
        first_linear_layer = model.block1.linear1
    else:
        # 如果结构改变，尝试找到第一个 nn.Linear 层
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear_layer = module
                break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(1024, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0) # 计算平均绝对权重作为重要性
    else:
        # 如果找不到线性层，返回零重要性（理论上不应发生）
        print("警告: 无法找到第一层线性层来计算特征重要性。")
        feature_importance = np.zeros(input_dim)

    return feature_importance


def fbeta_threshold(y_true, probas, beta=1.0):
    """
    找到能最大化 F-beta 分数的最佳阈值。

    Args:
        y_true (array-like): 真实二元标签。
        probas (array-like): 正类的预测概率。
        beta (float): 组合分数中召回率的权重。默认为 1.0 (F1)。

    Returns:
        tuple: 最佳阈值和对应的最大 F-beta 分数。
    """
    # 在 0 和 1 之间生成 1001 个阈值，排除 0 和 1 以保证数值稳定性
    thresholds = np.linspace(0, 1, 1001)[1:-1]
    best_t, max_fbeta = 0.5, -1 # 初始化最佳阈值和分数

    for t in thresholds:
        # 根据当前阈值生成预测
        preds = (probas >= t).astype(int)

        # 计算混淆矩阵，避免除以零
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.size == 4:
            # 正常的 2x2 混淆矩阵
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # 所有预测都属于一个类
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # 全部预测为负
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # 全部预测为正
        else:
            # 不规则矩阵，用零填充缺失项
            padded_cm = np.pad(cm, ((0, 2 - cm.shape[0]), (0, 2 - cm.shape[1])), mode='constant')
            tn, fp, fn, tp = padded_cm.ravel()

        # 计算精度和召回率
        precision_val = tp / (tp + fp + 1e-15)
        recall_val = tp / (tp + fn + 1e-15)

        # 计算 F-beta 分数
        numerator = (1 + beta ** 2) * (precision_val * recall_val)
        denominator = ((beta ** 2 * precision_val) + recall_val + 1e-15)

        if denominator > 0:
            fbeta = numerator / denominator
        else:
            fbeta = 0.0

        # 更新最佳阈值和分数
        if fbeta > max_fbeta:
            max_fbeta, best_t = fbeta, t

    return best_t, max_fbeta


if __name__ == "__main__":
    # 检查是否安装了 XGBoost
    if not XGB_AVAILABLE:
        print("错误: 缺少必需的依赖库 'xgboost'。程序退出。")
        exit(1)

    print("=" * 70)
    print("开始训练流程:")
    print("DNN 特征选择 (使用 ResNet) -> ")
    print("组合采样 (ADASYN + ClusterCentroids) -> ")
    print("成本敏感 XGBoost -> 级联校正 (GaussianNB)")
    print("(如果可用，通过 PyTorch/XGBoost 利用 GPU 加速)")
    print("=" * 70)

    # --- 配置 ---
    data_path = "./clean.csv"
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 为演示目的创建示例数据。在实践中请替换为真实数据路径。
        print(f"警告: 未找到数据文件 '{data_path}'。正在生成示例数据...")
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.rand(1000) * 100,
            'category_A': ['Y'] * 300 + ['N'] * 700,
            'target': ([1] * 150 + [0] * 150) + ([1] * 50 + [0] * 650)  # 创建不平衡
        })
        sample_data = pd.get_dummies(sample_data, drop_first=True)
        sample_data.to_csv(data_path, index=False)
        print(f"示例数据已生成并保存到 {data_path}")

    test_size = 0.10 # 验证集比例
    random_state = 42 # 随机种子
    beta_for_threshold = 2.0  # 在阈值选择期间优化 F2-score
    # 关键修改：确定设备，优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # --- 成本敏感 XGBoost 超参数 ---
    # 计算 scale_pos_weight 用于成本敏感学习
    # 我们将在拟合缩放器/重采样器后首先获取准确的计数。
    # 对于初始设置，我们定义占位符或使用粗略估计。
    # 它将在重采样步骤后更新。
    # 初始占位符值
    pos_count_initial = (pd.read_csv(data_path)['target'] == 1).sum()
    neg_count_initial = (pd.read_csv(data_path)['target'] == 0).sum()
    scale_pos_weight_initial = neg_count_initial / pos_count_initial if pos_count_initial > 0 else 1

    xgb_params = {
        'max_depth': 6, # 最大深度
        'learning_rate': 0.1, # 学习率
        'n_estimators': 200, # 树的数量
        'objective': 'binary:logistic', # 二分类目标函数
        'eval_metric': 'logloss',  # 用于早期停止
        'random_state': random_state, # 随机种子
        'n_jobs': -1, # 使用所有 CPU 核心
        'scale_pos_weight': scale_pos_weight_initial,  # 成本敏感性的占位符
        # 启用 GPU (如果可用且 XGBoost 是用 GPU 支持构建的)
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist', # 树构建方法
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor' # 预测器
    }
    print(f"初始 XGBoost 参数 (包括 scale_pos_weight 估计值): {xgb_params}")

    # --- 1. 加载和预处理数据 ---
    data = pd.read_csv(data_path)
    # 删除 company_id 列 (如果存在)
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # 执行 One-Hot 编码
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到目标列 'target'。")

    X_all = data.drop(columns=["target"]).values # 特征
    y_all = data["target"].values # 标签
    print(f"数据加载完成: X={X_all.shape}, y={y_all.shape}, 正例={y_all.sum()}, 负例={(y_all == 0).sum()}")

    # 填充缺失值
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 保存预处理器
    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # --- 2. DNN 特征加权 ---
    print("正在训练 Torch DNN (带有残差连接) 以获取特征权重 ...")
    # 关键修改：将设备传递给训练函数
    feature_importance = train_dnn_feature_selector_torch(
        X_scaled, y_all,
        input_dim=X_scaled.shape[1],
        device=device  # 使用指定设备
    )
    # 根据重要性对特征进行排序
    ranked_idx = np.argsort(-feature_importance)
    # 根据排名选择特征 (这里使用所有特征，但可以选择前 N 个)
    X_selected = X_scaled[:, ranked_idx]
    print(f"Torch DNN 特征加权完成。使用的特征总数={X_selected.shape[1]}")

    # 保存特征排名
    save_object(ranked_idx, "./dnn_feature_ranking.pkl")

    # --- 3. 划分训练/验证集 ---
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # --- 4. 组合采样策略 (ADASYN + ClusterCentroids) ---
    print("正在进行组合采样: ADASYN 过采样后接 ClusterCentroids 欠采样 ...")
    # 定义组合采样流水线
    sampler_pipeline = ImbPipeline([
        ('over_sampler', ADASYN(random_state=random_state)), # 过采样器
        ('under_sampler', ClusterCentroids(random_state=random_state)) # 欠采样器
    ])

    # 应用采样
    X_resampled, y_resampled = sampler_pipeline.fit_resample(X_train_full, y_train_full)
    print(
        f"组合采样后: X={X_resampled.shape}, 正例={y_resampled.sum()}, 负例={(y_resampled == 0).sum()}")

    # 根据实际重采样后的比例更新 XGBoost 参数，以实现精确的成本敏感性
    unique, counts = np.unique(y_resampled, return_counts=True)
    count_dict = dict(zip(unique, counts))
    neg_count_actual = count_dict.get(0, 0)
    pos_count_actual = count_dict.get(1, 0)
    scale_pos_weight_actual = neg_count_actual / pos_count_actual if pos_count_actual > 0 else 1
    xgb_params['scale_pos_weight'] = scale_pos_weight_actual
    print(f"根据重采样数据更新的 XGBoost scale_pos_weight: {scale_pos_weight_actual}")

    # --- 5. 训练基础模型 (成本敏感 XGBoost) ---
    print("正在训练成本敏感 XGBoost 基础模型 ...")
    # 准备 XGBoost 数据集以用于潜在的早期停止
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    # 使用更新后的参数训练 XGBoost 模型
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')], # 验证集用于评估
        early_stopping_rounds=20,  # 可选：启用早期停止
        verbose_eval=False,  # 设置为 True 可查看训练进度
        evals_result=evals_result
    )


    # 包装模型以提供类似 scikit-learn 的接口
    class WrappedXGBModel:
        """包装 XGBoost 模型以兼容 scikit-learn 接口"""
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            """预测概率"""
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            # 返回二维数组 [[prob_0, prob_1], ...]
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            """预测类别"""
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    # 保存基础模型
    save_object(wrapped_xgb_model, "./base_model_xgboost_cost_sensitive.pkl")
    print("基础模型训练完成: 成本敏感 XGBoost")

    # --- 6. 训练级联校正器 ---
    print("识别基础模型的误分类样本，训练级联校正器...")
    # 使用新的预测方法，初始时传入 None 作为 meta_model
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    # 找出被基础模型误分类的样本
    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    # 如果存在足够的误分类样本（且包含两个类别），则在这些样本上训练校正器
    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:
        meta_clf = GaussianNB()
        meta_clf.fit(X_hard, y_hard)
        print(f"级联校正器训练成功。困难样本数量={len(X_hard)}")
    else:
        # 如果没有足够的误分类样本或困难集中只有一个类别，则在全部重采样数据上训练
        meta_clf = GaussianNB()
        meta_clf.fit(X_resampled, y_resampled)
        print("未找到足够的误分类样本。校正器在全部重采样数据上训练。")

    # --- 7. 阈值选择 (针对 F-beta 分数优化) ---
    # 使用新的预测函数
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    # 寻找最优阈值
    best_thresh, max_fbeta = fbeta_threshold(y_holdout, holdout_probas, beta=beta_for_threshold)
    print(
        f"F{beta_for_threshold} 优化的阈值搜索完成。最佳阈值={best_thresh:.4f}, 最大 F{beta_for_threshold}={max_fbeta:.4f}")

    # 使用最佳阈值生成最终预测
    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    # 计算评估指标
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # 自定义评分公式 (注意：原始注释中的系数未加和到 100%，此处保留原样)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"评估指标 (阈值={best_thresh:.4f}):")
    print(f"召回率={recall:.5f}, AUC={auc:.5f}, 精确率={precision:.5f}, F1={f1:.5f}, 最终得分={final_score:.5f}")

    # --- 8. 保存完整的模型流水线 ---
    model_dict = {
        "imputer": imputer, # 缺失值填充器
        "scaler": scaler, # 特征缩放器
        "dnn_feature_ranking": ranked_idx, # DNN 特征排名
        "base_models": wrapped_xgb_model,  # 单个基础模型
        "base_types": ["XGBoost"],  # 基础模型类型列表
        "meta_nb": meta_clf, # 级联校正器 (朴素贝叶斯)
        "threshold": float(best_thresh), # 最佳阈值
    }
    # 保存整个模型流水线
    save_object(model_dict, "./model_pipeline_cascade_dnn_torch_improved.pkl")
    print("完整的模型流水线已保存到 ./model_pipeline_cascade_dnn_torch_improved.pkl")
    print("=" * 70)




