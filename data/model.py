import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.feature_selection import VarianceThreshold
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
    print("警告: 未找到 xgboost 库。请运行 'pip install xgboost' 进行安装。")
    XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F # For Focal Loss
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


# --- Focal Loss 定义 ---
class FocalLoss(nn.Module):
    """
    Focal Loss, 用于解决类别不平衡和易分样本主导问题。
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.8, gamma=1.5, reduction='mean'):
        """
        Args:
            alpha (float): 平衡因子，用于正负样本权重。这里设为正样本权重。
            gamma (float): 聚焦参数，用于减少易分样本的损失贡献。
            reduction (str): 指定应用于输出的规约方式：'none' | 'mean' | 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): 模型输出的 logits (未经过 sigmoid)。
                             Shape: (N, *) 其中 N 是样本数。
            targets (Tensor): 真实标签。必须是 float 类型，与 inputs 同形状。
                             Shape: (N, *)
        Returns:
            Tensor: 计算得到的 Focal Loss。
        """
        # 计算 BCE Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # 计算 p_t
        pt = torch.exp(-BCE_loss)
        # 计算 Focal Loss
        # 对于正样本，权重为 alpha；对于负样本，权重为 (1 - alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# --- 带 KL 散度正则化的深度特征选择器 ---
class DeepFeatureSelectorWithKL(nn.Module):
    """更深的全连接网络，不含自注意力机制，并包含 KL 散度正则化"""

    def __init__(self, input_dim):
        super().__init__()
        # 定义网络结构
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.05)

        self.fc_out = nn.Linear(64, 1)

        # 用于 KL 散度计算的先验分布参数 (标准正态分布)
        self.prior_mu = 0.0
        self.prior_sigma = 1.0

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        output = torch.sigmoid(self.fc_out(x))
        return output

    def kl_divergence_loss(self):
        """
        计算模型所有可学习参数相对于标准正态先验的 KL 散度之和。
        这里简化处理，将所有参数视为独立的高斯分布，均值为其参数值，方差设为一个固定小值或根据参数学习。
        为了简化，我们直接将参数视为从 N(0, 1) 中采样，计算其与 N(0, 1) 的 KL 散度。
        注意：这在实际贝叶斯神经网络中是不准确的，通常我们会为权重和偏置分别建模并学习其分布参数。
        此处为简化实现，仅对所有参数计算与标准正态的差异。
        """
        kl_loss = 0
        # 遍历所有参数
        for param in self.parameters():
            # 假设参数的后验分布是 N(param.data, sigma^2)，其中 sigma 是一个很小的固定值或可学习的
            # 这里我们简化地计算 (param - 0)^2 / (2 * 1^2) = param^2 / 2
            # 实际 KL(p||q) = 0.5 * (log(sigma2^2/sigma1^2) + (sigma1^2 + (mu1-mu2)^2)/sigma2^2 - 1)
            # 其中 p=N(mu1, sigma1^2), q=N(mu2, sigma2^2)
            # 设 p=N(param, small_sigma^2), q=N(0, 1)
            # KL = 0.5 * (log(1/small_sigma^2) + (small_sigma^2 + param^2)/1 - 1)
            # 为了进一步简化，我们忽略 log 项和 small_sigma^2 项，近似为 0.5 * param^2
            # 或者更简单地，直接使用参数的 L2 范数平方作为正则化项，这与权重衰减类似。
            # PyTorch 的 weight_decay 参数在 AdamW 中已经实现了 L2 正则化。
            # 因此，如果使用了 weight_decay，这里可以不加额外的 KL 惩罚，或者 KL 惩罚非常小。
            # 这里我们实现一个更标准的 KL 计算，假设参数分布是 N(param, sigma^2_fixed)
            # 并且我们希望它接近 N(0, 1)。
            # 我们可以为每个参数层添加可学习的 log_sigma 参数，但这会使模型复杂化。
            # 因此，我们将使用一种启发式方法，将参数偏离 0 的程度作为 KL 的代理。
            # 一个简单的近似是 KL ≈ 0.5 * (param^2 - 1 - log(param^2 + eps))，但这对单个参数不稳定。
            # 最简单的近似是 KL ≈ 0.5 * param^2，这等价于 L2 正则化。
            # 为了体现 KL 的思想，我们计算参数与 N(0,1) 的平方差之和。
            # 但这与 weight_decay 冲突。因此，我们只在没有使用 weight_decay 或需要额外约束时使用。
            # 本实现中，我们将 KL 损失定义为参数平方和的一半，模拟标准正态先验下的 KL 散度。
            kl_loss += torch.sum(0.5 * param ** 2) # 简化版 KL 散度
        return kl_loss


def train_dnn_feature_selector_torch_with_focal_kl(
    X, y, input_dim, epochs=1000, batch_size=256, lr=1e-3, device="cpu", patience=50,
    focal_alpha=0.8, focal_gamma=1.5, kl_beta=0.01
):
    """
    使用 Focal Loss 和 KL 散度正则化训练深度特征选择器

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
        focal_alpha : float, optional (default=0.8)
            Focal Loss 的 alpha 参数 (正样本权重).
        focal_gamma : float, optional (default=1.5)
            Focal Loss 的 gamma 参数 (聚焦参数).
        kl_beta : float, optional (default=0.01)
            KL 散度正则化的强度.

    返回:
        feature_importance : ndarray, shape (input_dim,)
            特征重要性分数.
    """
    # 初始化模型、损失函数和优化器
    model = DeepFeatureSelectorWithKL(input_dim).to(device)
    focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # weight_decay 作为基础 L2 正则
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2)

    # 准备数据加载器
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device) # BCE/Focal Loss 需要 FloatTensor
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 早停初始化
    best_loss = float('inf')
    trigger_times = 0
    early_stop = False

    model.train()
    pbar = tqdm(range(epochs), desc="Training DNN Feature Selector (Focal+KL)")

    for epoch in pbar:
        total_loss = 0.0
        total_focal_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for xb, yb in loader:
            yb = yb.unsqueeze(1) # 调整标签形状为 (batch_size, 1)
            optimizer.zero_grad()

            outputs = model(xb) # outputs shape: (batch_size, 1)

            # 计算 Focal Loss
            focal_loss = focal_criterion(outputs, yb)

            # 计算 KL 散度损失
            kl_loss = model.kl_divergence_loss()

            # 总损失 = Focal Loss + β * KL Loss
            total_batch_loss = focal_loss + kl_beta * kl_loss

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_focal_loss += focal_loss.item()
            total_kl_loss += (kl_beta * kl_loss).item() # 记录实际加到 loss 上的 KL 部分
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_focal_loss = total_focal_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Avg Loss': f'{avg_loss:.4f}',
            'Focal': f'{avg_focal_loss:.4f}',
            'KL': f'{avg_kl_loss:.4f}',
            'LR': f'{current_lr:.2e}'
        })

        # --- Early Stopping Logic ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    pbar.close()

    # 提取第一层权重以计算特征重要性
    first_linear_layer = model.fc1 # 直接访问第一层 Linear 层
    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy() # shape=(output_dim_of_first_layer, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0)
    else:
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


if __name__ == "__main__":
    if not XGB_AVAILABLE:
        print("错误: 缺少必要依赖库 xgboost。程序退出。")
        exit(1)

    print("=" * 60)
    print("开始训练：Variance Filter -> SMOTEENN -> XGBoost -> 级联 GaussianNB (Torch DNN 特征权重)")
    print("-> DNN 不含自注意力机制，并利用 GPU (如果可用)，新增早停、学习率调度和进度条")
    print("-> 新增按特征重要性排序后选取 Top 90% 的特征")
    print("-> 改造损失函数：引入 Focal Loss 和 KL 散度正则化")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 创建示例数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到数据文件 '{data_path}'。正在创建示例数据...")
        # 示例数据生成代码（简化版）
        sample_n = 1000
        sample_features = 20
        # 添加一些零方差特征
        zero_var_features = 2
        np.random.seed(42)
        sample_X = np.random.randn(sample_n, sample_features - zero_var_features)
        zero_var_data = np.full((sample_n, zero_var_features), 5.0)  # 零方差列
        sample_X = np.hstack([sample_X, zero_var_data])

        # 创建一些有意义的相关性
        sample_y = ((sample_X[:, 0] + sample_X[:, 1] - sample_X[:, 2]) > 0).astype(int)
        feature_names = [f"f{i}" for i in range(sample_features - zero_var_features)] + [f"zero_var_{i}" for i in
                                                                                         range(zero_var_features)]
        sample_df = pd.DataFrame(sample_X, columns=feature_names)
        sample_df['target'] = sample_y
        sample_df.to_csv(data_path, index=False)
        print(f"...示例数据已保存至 '{data_path}'")

    test_size = 0.10
    random_state = 42
    variance_threshold_value = 0.0  # 设置方差阈值
    top_percentile_to_select = 0.9  # 保留前90%重要的特征

    # Focal Loss 和 KL 散度参数
    focal_alpha = 0.8
    focal_gamma = 1.5
    kl_beta = 0.01

    # 关键修改: 确定设备，优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # XGBoost 超参
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # 用于早停
        'random_state': random_state,
        'n_jobs': -1,
        # 启用 GPU (如果可用且配置了 GPU 支持的 XGBoost)
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
    }

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path)
    # 删除 company_id 列（如果存在）
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # 进行独热编码
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    initial_feature_names = data.drop(columns=["target"]).columns.tolist()
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all == 0).sum()}")

    # ---- 2. 特征过滤：移除低方差特征 ----
    print(f"应用方差过滤器 (阈值={variance_threshold_value}) ...")
    selector_variance = VarianceThreshold(threshold=variance_threshold_value)
    X_var_filtered = selector_variance.fit_transform(X_all)
    selected_feature_indices_variance = selector_variance.get_support(indices=True)
    selected_feature_names_variance = [initial_feature_names[i] for i in selected_feature_indices_variance]
    print(
        f"方差过滤后特征数: {X_var_filtered.shape[1]} (移除了 {len(initial_feature_names) - len(selected_feature_names_variance)} 个特征)")

    # ---- 3. 数据清洗与标准化 ----
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_var_filtered)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")
    save_object(selector_variance, "./variance_selector.pkl")  # 保存方差过滤器

    # ----- 4. DNN 特征权重 (使用改造后的损失函数) -----
    print("使用 Torch DNN (Focal Loss + KL Reg) 训练获取特征权重 ...")
    # 关键修改: 将设备和新参数传递给训练函数，并增加迭代次数
    feature_importance = train_dnn_feature_selector_torch_with_focal_kl(
        X_scaled, y_all,
        input_dim=X_scaled.shape[1],
        epochs=1000,  # 设置为1000次迭代
        device=device,  # 使用指定的设备
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        kl_beta=kl_beta
    )
    ranked_idx_by_importance = np.argsort(-feature_importance)

    # 关键修改: 根据特征重要性排序选择Top百分比的特征
    num_top_features = int(top_percentile_to_select * len(feature_importance))
    if num_top_features <= 0:
        num_top_features = 1
    selected_top_feature_indices = ranked_idx_by_importance[:num_top_features]

    X_selected = X_scaled[:, selected_top_feature_indices]
    selected_feature_names_final = [selected_feature_names_variance[i] for i in selected_top_feature_indices]
    print(
        f"Torch DNN 特征加权完成，并选择了 Top {top_percentile_to_select * 100}% ({num_top_features}/{len(feature_importance)}) 的特征.")

    save_object(ranked_idx_by_importance, "./dnn_feature_ranking.pkl")

    # ----- 5. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # ----- 6. SMOTEENN 重采样 -----
    print("正在进行 SMOTEENN 重采样 ...")
    smoteenn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train_full, y_train_full)
    print(f"重采样后: X={X_resampled.shape}, 正类={y_resampled.sum()}, 负类={(y_resampled == 0).sum()}")

    # ----- 7. 训练基模型 (改为 XGBoost) -----
    print("训练 XGBoost 基模型 ...")
    # 准备 XGBoost 数据集以便使用早停功能（可选）
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    # 训练 XGBoost 模型
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,  # 可选：启用早停
        verbose_eval=False,  # 设置为True可以看到训练过程
        evals_result=evals_result
    )


    # 为了兼容后续的 predict_proba 接口，我们可以包装一下
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


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost.pkl")
    print("已训练基模型：XGBoost")

    # ----- 8. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器...")
    # 使用新的预测方法，传入None作为meta_model
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:  # 确保困难样本集中至少有两个类别
        meta_clf = GaussianNB()
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器训练完成，困难样本数={len(X_hard)}")
    else:
        # 如果没有误分类样本或困难样本只属于一个类别，则使用全部重采样后的数据训练
        meta_clf = GaussianNB()
        meta_clf.fit(X_resampled, y_resampled)
        print("未发现足够的误分类样本，使用全部重采样数据训练修正器")

    # ----- 9. 阈值选择（Youden's J） -----
    # 使用新的预测函数
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    best_thresh = youden_threshold(y_holdout, holdout_probas)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # 自定义评分公式 (注意：原注释中的系数总和不是100%，这里保持原样)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"最佳阈值={best_thresh:.4f}, F1={f1:.4f}")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, FinalScore={final_score:.5f}")

    # ----- 10. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "variance_selector": selector_variance,  # 保存方差过滤器
        "dnn_feature_ranking": ranked_idx_by_importance,  # 保存原始排名索引
        "selected_feature_names": selected_feature_names_final,  # 保存最终选择的特征名称
        "base_models": wrapped_xgb_model,  # 单一模型
        "base_types": ["XGBoost"],  # 类型列表
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
        # 保存 Focal Loss 和 KL 参数
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "kl_beta": kl_beta,
    }
    save_object(model_dict, "./model.pkl")
    print("已保存完整模型 ./model.pkl")
    print("=" * 60)




