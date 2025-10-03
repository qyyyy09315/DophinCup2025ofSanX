import os
import pickle
import warnings

import numpy as np
import pandas as pd

# 导入 KNNImputer 和 特征选择工具
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, RFECV
# 导入交叉验证相关的工具
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier # 导入 GBM
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
np.random.seed(42)
# 设置 PyTorch 随机种子以获得可重复的结果
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        # 如果基模型不支持 predict_proba，则尝试直接预测概率形式（不太常见）
        # 或者假设输出就是概率（对于某些模型包装器）
        # 这里简化处理，假定返回的是决策函数或类似分数，并手动 sigmoid 化
        # 但 safest way 是要求模型有 predict_proba 方法
        # fallback to predict if necessary and treat as probability-like
        pred_or_score = base_model.predict(X)
        if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
            probas_base = np.clip(pred_or_score.flatten(), 0, 1)
        else:
            probas_base = pred_or_score[:, 1]

    preds = (probas_base >= threshold).astype(int)
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

    if meta_model is not None and uncertain_mask.sum() > 0:
        # Check if meta_model has predict_proba method for consistency
        if hasattr(meta_model, 'predict_proba'):
            try:
                meta_probas = meta_model.predict_proba(X[uncertain_mask])[:, 1]
                meta_preds = (meta_probas >= 0.5).astype(int)
            except:
                meta_preds = meta_model.predict(X[uncertain_mask])
        else:
            meta_preds = meta_model.predict(X[uncertain_mask])

        preds[uncertain_mask] = meta_preds.astype(int)  # Ensure type match

    return preds, probas_base


# --- 修改后的阈值搜索函数 (限制阈值范围) ---
def cost_sensitive_threshold(y_true, probas, recall_weight=30, auc_weight=50, precision_weight=20):
    """
    使用自定义加权分数 (FinalScore = w1*Recall + w2*AUC + w3*Precision) 进行代价敏感阈值搜索。
    阈值搜索范围被限制在 [0.2, 0.8]。

    注意: AUC 是一个全局指标，不依赖于单一阈值。为了使其适应阈值搜索，
    我们在这里计算的是使用当前阈值 `t` 得到的预测结果 `preds` 时的 AUC。
    这意味着 AUC 的计算是基于该特定阈值下的二分类结果与真实标签之间的关系。
    如果 y_true 和 probas 是固定的，AUC 在不同阈值下计算的结果会是相同的。
    因此，这里的 AUC 实际上是使用 probas 本身计算的，与阈值 t 无关。
    更合理的做法可能是忽略 AUC 作为阈值依赖的指标，或者将其权重设为0，
    或者寻找一个与阈值相关的替代指标（例如，将 AUC 拆解为 TPR 和 FPR 的函数）。
    但为了遵循原始意图，我们仍然在计算中包含它，但需注意其含义。

    更常见的做法是优化 F1, F-beta, 或 Recall-Precision 平衡。
    这里我们按要求实现，但需明确 AUC 的处理方式。

    Args:
        y_true: 真实标签 (numpy array)。
        probas: 预测为正类的概率 (numpy array)。
        recall_weight: Recall 的权重。
        auc_weight: AUC 的权重 (注意：AUC 全局固定)。
        precision_weight: Precision 的权重。

    Returns:
        best_t: 最佳阈值。
    """
    # --- 修改点：限制阈值搜索范围 ---
    min_threshold = 0.2
    max_threshold = 0.8
    thresholds = np.linspace(min_threshold, max_threshold, 101)
    print(
        f"开始代价敏感阈值搜索 (范围: [{min_threshold}, {max_threshold}], 权重: Recall={recall_weight}, AUC={auc_weight}, Precision={precision_weight})...")
    # --- 修改结束 ---

    # 预先计算全局 AUC，因为它不依赖于阈值
    try:
        global_auc = roc_auc_score(y_true, probas)
    except Exception:
        print("警告: 无法计算全局 AUC，设置为 0.5")
        global_auc = 0.5

    best_t, best_score = 0.5, -np.inf  # 初始值设为0.5，但最终会被搜索范围内的值覆盖

    for t in thresholds:
        preds = (probas >= t).astype(int)

        # 计算各项指标
        try:
            recall = recall_score(y_true, preds, zero_division=0)
        except ValueError:
            recall = 0.0

        # AUC 使用全局值，因为它不是阈值的函数
        auc = global_auc

        try:
            precision = precision_score(y_true, preds, zero_division=0)
        except ValueError:
            precision = 0.0

        # 计算加权分数
        final_score = recall_weight * recall + auc_weight * auc + precision_weight * precision

        if final_score > best_score:
            best_score, best_t = final_score, t

    print(f"代价敏感阈值搜索完成。最佳阈值: {best_t:.4f}, 最佳加权分数: {best_score:.4f}")
    return best_t


# --- RFA (Recursive Feature Addition) 实现 ---
def recursive_feature_addition(estimator, X, y, cv=None, scoring='roc_auc', min_features=5):
    """
    递归特征添加 (RFA) 算法，自动确定最优特征数量。

    :param estimator: 用于评估特征子集的 scikit-learn 估计器。
    :param X: 特征矩阵 (numpy array or pandas DataFrame)。
    :param y: 目标向量 (numpy array)。
    :param cv: 交叉验证策略 (例如 StratifiedKFold 对象)。
    :param scoring: 用于评估的指标 (例如 'roc_auc', 'f1')。
    :param min_features: 最小特征数量。
    :return: selected_indices (被选中的特征索引), scores_history (每步的得分历史).
    """
    print("开始执行递归特征添加 (RFA)...")
    n_features = X.shape[1]

    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    selected_indices = []
    remaining_indices = list(range(n_features))
    scores_history = []

    while len(selected_indices) < n_features and len(selected_indices) < 50:  # 限制最大特征数
        scores_with_candidates = []

        # 单步添加：评估每个候选特征
        for i in remaining_indices:
            candidate_indices = selected_indices + [i]
            try:
                score = cross_val_score(estimator, X[:, candidate_indices], y, cv=cv, scoring=scoring).mean()
                scores_with_candidates.append((score, i))
            except Exception as e:
                print(f"警告: 评估特征 {i} 时出错: {e}")
                scores_with_candidates.append((-np.inf, i))  # 给出极差得分

        if not scores_with_candidates:
            break

        scores_with_candidates.sort(reverse=True)  # 降序排列
        best_score, best_idx = scores_with_candidates[0]

        # 检查是否应该停止添加特征（基于性能改善）
        if len(scores_history) > 0 and best_score <= max(scores_history) and len(selected_indices) >= min_features:
            print(f"  性能不再提升，停止特征添加。最优特征数: {len(selected_indices)}")
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        scores_history.append(best_score)
        print(f"  添加特征索引 {best_idx} (得分: {best_score:.4f})")

    print(f"RFA 完成，最终选择了 {len(selected_indices)} 个特征。")
    return np.array(selected_indices), scores_history


# --- WGAN-GP 相关定义 (含自注意力机制) ---

class SelfAttention(nn.Module):
    """自注意力模块"""

    def __init__(self, data_dim):
        super(SelfAttention, self).__init__()
        self.data_dim = data_dim
        # 将每个特征维度视为长度为1的向量，方便计算注意力
        self.feature_dim = 1
        # 查询、键、值的线性变换 (这里可以扩展 feature_dim)
        self.query = nn.Linear(self.feature_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.value = nn.Linear(self.feature_dim, self.feature_dim)
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([self.feature_dim])).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # 输出投影层 (可选，用于增加模型表达能力)
        self.out_proj = nn.Linear(data_dim, data_dim)
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(data_dim)

    def forward(self, x):
        # x shape: (batch_size, data_dim)
        batch_size = x.size(0)

        # Reshape for attention: (batch_size, seq_len=data_dim, feat_dim=1)
        x_reshaped = x.unsqueeze(-1)  # (batch_size, data_dim, 1)

        # Linear transformations
        Q = self.query(x_reshaped)  # (batch_size, data_dim, 1)
        K = self.key(x_reshaped)  # (batch_size, data_dim, 1)
        V = self.value(x_reshaped)  # (batch_size, data_dim, 1)

        # 计算注意力分数 (Q @ K^T) / sqrt(d_k)
        # (batch_size, data_dim, 1) @ (batch_size, 1, data_dim) -> (batch_size, data_dim, data_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, data_dim, data_dim)

        # Apply attention weights to values
        # (batch_size, data_dim, data_dim) @ (batch_size, data_dim, 1) -> (batch_size, data_dim, 1)
        attended_values = torch.matmul(attention_weights, V)

        # Squeeze back to (batch_size, data_dim)
        attended_values = attended_values.squeeze(-1)  # (batch_size, data_dim)

        # Optional: Output projection
        attended_values = self.out_proj(attended_values)

        # 残差连接和层归一化
        out = self.layer_norm(x + attended_values)
        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.data_dim = data_dim
        # 注意力模块
        self.attention = SelfAttention(data_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, data_dim),
            # 线性输出，因为数据经过StandardScaler后范围不定
        )

    def forward(self, z):
        raw_output = self.model(z)
        # 应用自注意力
        attended_output = self.attention(raw_output)
        return attended_output


class Critic(nn.Module):  # 判别器更名为Critic
    def __init__(self, data_dim):
        super(Critic, self).__init__()
        self.data_dim = data_dim
        # 注意力模块
        self.attention = SelfAttention(data_dim)

        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # 注意：WGAN 中 Critic 最后一层没有激活函数 Sigmoid
        )

    def forward(self, data):
        # 应用自注意力
        attended_data = self.attention(data)
        output = self.model(attended_data)
        return output


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), device=device, requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgangp_resample(X_train, y_train, minority_class=1, latent_dim=100, epochs=300, batch_size=64, lr=0.0001,
                    device='cpu', n_critic=5, clip_value=0.01, lambda_gp=10, lr_decay_rate=0.99):
    """
    使用 WGAN-GP 对少数类样本进行过采样，并加入动态学习率调度。
    :param X_train: 训练特征 (numpy array)
    :param y_train: 训练标签 (numpy array)
    :param minority_class: 少数类的标签
    :param latent_dim: 生成器输入的潜在空间维度
    :param epochs: GAN 训练轮数
    :param batch_size: 批次大小
    :param lr: 初始学习率
    :param device: 'cpu' 或 'cuda'
    :param n_critic: 训练几次判别器才训练一次生成器
    :param lambda_gp: 梯度惩罚系数
    :param lr_decay_rate: 学习率衰减率 (例如 0.99 表示每轮衰减1%)
    :return: resampled_X, resampled_y (numpy arrays)
    """
    print("开始 WGAN-GP 过采样...")
    # 1. 准备少数类数据
    X_minority = X_train[y_train == minority_class]
    if len(X_minority) == 0:
        print("警告: 少数类样本数为0，无法进行WGAN-GP采样。")
        return X_train, y_train

    data_dim = X_train.shape[1]
    num_minority = len(X_minority)
    num_majority = len(y_train) - num_minority
    num_to_generate = num_majority - num_minority  # 目标是平衡

    if num_to_generate <= 0:
        print("警告: 少数类样本数已大于等于多数类，无需过采样。")
        return X_train, y_train

    print(f"当前少数类样本数: {num_minority}, 多数类样本数: {num_majority}")
    print(f"计划生成 {num_to_generate} 个少数类样本...")

    # 转换为 Tensor
    X_minority_tensor = torch.tensor(X_minority, dtype=torch.float32).to(device)

    # 2. 初始化网络
    generator = Generator(latent_dim, data_dim).to(device)
    critic = Critic(data_dim).to(device)  # 使用 Critic

    # 3. 优化器 (通常使用 Adam)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    # 4. 添加学习率调度器
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=lr_decay_rate)
    scheduler_C = optim.lr_scheduler.ExponentialLR(optimizer_C, gamma=lr_decay_rate)
    print(f"启用 ExponentialLR 调度器, gamma={lr_decay_rate}")

    # 5. 训练循环
    generator.train()
    critic.train()

    batches_done = 0
    for epoch in range(epochs):

        # ---------------------
        #  训练 Critic (n_critic 次)
        # ---------------------
        for _ in range(n_critic):
            # Configure input
            idx = np.random.choice(len(X_minority_tensor), size=batch_size,
                                   replace=False if len(X_minority_tensor) >= batch_size else True)
            real_data = X_minority_tensor[idx]

            # -----------------
            # Train Critic
            # -----------------

            optimizer_C.zero_grad()

            # Sample noise as generator input
            z = torch.randn(batch_size, latent_dim, device=device)

            # Generate a batch of new data
            fake_data = generator(z).detach()
            # Adversarial loss (Wasserstein loss)
            loss_critic = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))

            # Calculate gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_data.data, fake_data.data, device)

            # Total loss
            c_loss = loss_critic + lambda_gp * gradient_penalty

            c_loss.backward()
            optimizer_C.step()

        # -----------------
        #  训练 Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of data
        gen_data = generator(torch.randn(batch_size, latent_dim, device=device))
        # Adversarial loss
        g_loss = -torch.mean(critic(gen_data))

        g_loss.backward()
        optimizer_G.step()

        # 更新学习率
        scheduler_G.step()
        scheduler_C.step()

        # Print losses occasionally
        if (epoch + 1) % 100 == 0 or epoch == 0:
            current_lr_g = optimizer_G.param_groups[0]['lr']
            current_lr_c = optimizer_C.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"C Loss: {loss_critic.item():.4f}, "
                f"GP: {gradient_penalty.item():.4f}, "
                f"G Loss: {g_loss.item():.4f}, "
                f"LR_G: {current_lr_g:.6f}, LR_C: {current_lr_c:.6f}"
            )

    # 6. 生成新样本
    generator.eval()
    with torch.no_grad():
        z_new = torch.randn(num_to_generate, latent_dim, device=device)
        generated_data = generator(z_new).cpu().numpy()

    # 7. 合并新旧数据
    resampled_X = np.vstack([X_train, generated_data])
    new_labels = np.full(num_to_generate, minority_class)
    resampled_y = np.hstack([y_train, new_labels])

    print(f"WGAN-GP过采样完成。新样本数: {num_to_generate}")
    print(f"采样后训练集分布: {dict(zip(*np.unique(resampled_y, return_counts=True)))}")
    return resampled_X, resampled_y


if __name__ == "__main__":

    print("=" * 60)
    print(
        "开始训练：Variance Filter -> RFE (XGBoost) -> WGAN-GP (含自注意力+动态学习率) -> XGBoost -> 级联 GradientBoostingClassifier (原为 Logistic Regression)")
    print("-> 已移除平衡随机森林 (Balanced Random Forest) 特征选择阶段")
    print("-> 修改: RFE阶段基分类器由 LogisticRegression 改为 XGBoost")
    print("-> 关键改进: 缺失值填充方法由均值填充改为KNN填充 (n_neighbors=5)")
    print("-> 关键改进: 在 RFE 阶段加入了交叉验证来评估特征子集性能")
    print("-> 新增功能: WGAN-GP中加入动态学习率调度 (ExponentialLR)")
    print("-> 修改: AdaptiveAttention 改为 SelfAttention")
    print("-> 新增: 集成递归特征添加 (RFA) 作为特征选择的可选补充")
    print("-> 修改: 不再限定最终选择的特征数量，而是使用 RFE 的 top_percentile_to_select 参数")
    print("-> 修改: 阈值选择改为基于加权分数 (Recall, AUC, Precision) 的代价敏感搜索")
    print("-> 修改: 代价敏感阈值搜索范围限制在 [0.2, 0.8]")
    print("-> 修改: 级联修正器 (Meta Model) 从 LogisticRegression 改为 XGBoost")
    print("-> 修改: 再次修改: 级联修正器 (Meta Model) 从 XGBoost 改为 GradientBoostingClassifier (Scikit-learn GBM)")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    if not os.path.exists(data_path):
        print(f"警告: 未找到数据文件 '{data_path}'。正在创建示例数据...")
        sample_n = 1000
        sample_features = 20
        zero_var_features = 2
        np.random.seed(42)
        sample_X = np.random.randn(sample_n, sample_features - zero_var_features)
        zero_var_data = np.full((sample_n, zero_var_features), 5.0)
        sample_X = np.hstack([sample_X, zero_var_data])
        # 创建不平衡数据集
        sample_y = ((sample_X[:, 0] + sample_X[:, 1] - sample_X[:, 2]) > 0).astype(int)
        # 人为减少类别1的数量
        indices_class_1 = np.where(sample_y == 1)[0]
        to_remove = int(0.8 * len(indices_class_1))  # 移除80%的类别1样本
        if to_remove > 0:
            remove_indices = np.random.choice(indices_class_1, size=to_remove, replace=False)
            sample_X = np.delete(sample_X, remove_indices, axis=0)
            sample_y = np.delete(sample_y, remove_indices, axis=0)

        feature_names = [f"f{i}" for i in range(sample_features - zero_var_features)] + [f"zero_var_{i}" for i in
                                                                                         range(zero_var_features)]
        sample_df = pd.DataFrame(sample_X, columns=feature_names)
        sample_df['target'] = sample_y
        sample_df.to_csv(data_path, index=False)
        print(f"...示例不平衡数据已保存至 '{data_path}'")

    test_size = 0.10
    random_state = 42
    variance_threshold_value = 0.0
    cv_folds_for_rfe = 3
    use_rfa = True  # <--- 新增配置项：是否使用 RFA

    # WGAN-GP 配置
    wgangp_config = {
        'latent_dim': 100,
        'epochs': 300,  # 可根据需要调整
        'batch_size': 64,
        'lr': 0.0001,  # WGAN-GP 推荐较小的学习率
        'lambda_gp': 10,  # 梯度惩罚系数
        'n_critic': 5,  # 训练几次判别器才训练一次生成器
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr_decay_rate': 0.99  # 新增配置项
    }

    # XGBoost 超参 (用于最终模型和RFE)
    xgb_params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': random_state,
        'n_jobs': -1,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor'
    }

    # Meta Model (Cascade Corrector) GradientBoostingClassifier 超参
    # 使用 Sklearn 默认参数为基础，稍作调整
    meta_gbm_params = {
        'n_estimators': 100,       # 较少迭代次数
        'learning_rate': 0.1,
        'max_depth': 3,           # 较浅防止过拟合
        'subsample': 1.0,         # 不进行子采样
        'criterion': 'friedman_mse',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.0,
        'init': None,
        'random_state': random_state + 1, # Different seed
        'max_features': None,     # 使用所有特征
        'verbose': 0,
        'max_leaf_nodes': None,
        'warm_start': False,
        'validation_fraction': 0.1,
        'n_iter_no_change': None,
        'tol': 1e-4,
        'ccp_alpha': 0.0
    }


    # 阈值搜索权重配置
    threshold_weights = {
        'recall_weight': 30,
        'auc_weight': 50,
        'precision_weight': 20
    }

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path)
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
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
    # 使用 KNNImputer 替代 SimpleImputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_var_filtered)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")
    save_object(selector_variance, "./variance_selector.pkl")

    # ---- 4. Tree-based 特征权重 (替代DNN) -> 改为 RFE with Cross-Validation using XGBoost -----
    print("使用 RFECV (XGBoost) 结合交叉验证自动选择最优特征数 ...")

    # 定义用于RFE的XGBoost估计器
    estimator_rfe = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'n_estimators'})
    # 注意：为了兼容RFE接口，我们不传入'n_estimators'到构造函数

    skf_cv = StratifiedKFold(n_splits=cv_folds_for_rfe, shuffle=True, random_state=random_state)

    # 使用RFECV自动选择特征数
    selector_rfe = RFECV(estimator_rfe, step=5, cv=skf_cv, scoring='roc_auc', min_features_to_select=5)
    selector_rfe.fit(X_scaled, y_all)

    # --- 获取选定特征 ---
    selected_top_feature_indices = selector_rfe.support_
    X_selected_after_rfe = X_scaled[:, selected_top_feature_indices]

    # --- 获取特征排名信息 ---
    feature_ranking = selector_rfe.ranking_
    ranked_idx_by_importance = np.argsort(feature_ranking)

    # --- 准备最终使用的特征矩阵和名称 (RFE后) ---
    X_selected_after_rfe = X_scaled[:, selected_top_feature_indices]
    original_indices_of_selected = np.where(selected_top_feature_indices)[0]
    selected_feature_names_final = [selected_feature_names_variance[i] for i in original_indices_of_selected]

    print(
        f"RFE (XGBoost) 特征选择完成，并自动选择了 {len(original_indices_of_selected)} 个特征.")

    # 修复索引错误：安全地访问cv_results_
    n_features_selected = selector_rfe.n_features_
    cv_results_scores = selector_rfe.cv_results_['mean_test_score']

    # 确保索引在有效范围内
    if n_features_selected > 0 and len(cv_results_scores) >= n_features_selected:
        score_index = n_features_selected - 1  # 0-based indexing
        performance_score = cv_results_scores[score_index]
        print(f"这些特征在交叉验证(AUC)下的表现约为: {performance_score:.4f}")
    else:
        # 如果索引仍然无效，使用最后一个可用的分数
        performance_score = cv_results_scores[-1] if len(cv_results_scores) > 0 else 0.5
        print(f"无法直接获取选定特征数的性能，使用最近可用的性能分数: {performance_score:.4f}")

    save_object(ranked_idx_by_importance, "./rfe_feature_ranking.pkl")

    # --- 新增: 递归特征添加 (RFA) ---
    if use_rfa:
        print(f"启动 RFA 阶段，自动确定最优特征数...")

        # 使用 RFE 选出的特征作为 RFA 的输入
        X_input_for_rfa = X_selected_after_rfe
        feature_names_input_for_rfa = selected_feature_names_final

        # 定义用于 RFA 的评估器 (可以与 RFE 不同)
        estimator_rfa = xgb.XGBClassifier(**xgb_params)  # 使用相同的 XGBoost 参数

        # 执行 RFA
        try:
            selected_indices_rfa, rfa_scores_history = recursive_feature_addition(
                estimator_rfa, X_input_for_rfa, y_all,
                cv=StratifiedKFold(n_splits=cv_folds_for_rfe, shuffle=True, random_state=random_state),
                scoring='roc_auc'
            )

            # 根据 RFA 结果更新最终特征
            X_selected = X_input_for_rfa[:, selected_indices_rfa]
            selected_feature_names_final = [feature_names_input_for_rfa[i] for i in selected_indices_rfa]

            print(f"RFA 完成，最终选择了 {len(selected_feature_names_final)} 个特征。")
            print("RFA 选择的特征列表:")
            for name in selected_feature_names_final:
                print(f"  - {name}")

        except Exception as e:
            print(f"RFA 过程中发生错误: {e}")
            print("回退到 RFE 选择的特征。")
            X_selected = X_selected_after_rfe  # 回退
    else:
        X_selected = X_selected_after_rfe

    print(f"最终特征选择阶段完成，剩余特征数: {X_selected.shape[1]}")

    # ----- 6. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # ----- 7. 过采样 (使用 WGAN-GP 替换 ADASYN/GAN) -----
    print(f"正在进行 WGAN-GP (含自注意力+动态学习率) 过采样 ...")
    # 检查类别分布
    unique_classes, class_counts = np.unique(y_train_full, return_counts=True)
    print(f"WGAN-GP前训练集分布: {dict(zip(unique_classes, class_counts))}")

    # 应用 WGAN-GP 采样
    try:
        # 注意：这里传递的是 numpy arrays
        X_resampled, y_resampled = wgangp_resample(X_train_full, y_train_full, **wgangp_config)
    except Exception as e:
        print(f"WGAN-GP采样失败: {e}. 尝试使用原始不平衡数据继续训练。")
        X_resampled, y_resampled = X_train_full, y_train_full

    # ----- 8. 训练基模型 (改为 XGBoost) -----
    print("训练 XGBoost 基模型 ...")
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )


    class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            # For binary classification, Booster returns probabilities for class 1.
            # We need to stack them with 1-probs for class 0 to form [prob_0, prob_1].
            return np.column_stack([1 - probs, probs])

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost.pkl")
    print("已训练基模型：XGBoost")

    # ----- 9. 训练级联修正器 (现在也使用 GradientBoostingClassifier) -----
    print("识别基模型误分类样本，训练级联修正器 (使用 GradientBoostingClassifier)...")
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:

        # 使用 GradientBoostingClassifier 作为元模型
        meta_clf = GradientBoostingClassifier(**meta_gbm_params)
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器 (GradientBoostingClassifier) 训练完成，困难样本数={len(X_hard)}")

        # --- 新增部分：也可以对元模型做个简单的CV评估---
        try:
            meta_cv_scores = cross_val_score(meta_clf, X_hard, y_hard, cv=min(3, len(X_hard) // 2),
                                             scoring='f1')  # 至少两折
            meta_mean_cv_score = np.mean(meta_cv_scores)
            print(f"Meta-model CV F1 Score on hard examples: {meta_mean_cv_score:.4f}")
        except Exception as e:
            print(f"无法计算 Meta-model 的 CV 分数: {e}")

    else:
        # 如果没有足够的困难样本，仍初始化一个默认的 GradientBoostingClassifier 模型以防万一
        print("未发现足够的误分类样本，但仍初始化默认的 GradientBoostingClassifier 修正器。")
        meta_clf = GradientBoostingClassifier(**meta_gbm_params)
        meta_clf.fit(X_resampled, y_resampled)  # Fit on all data

    # Save the meta model object directly instead of its internal booster
    save_object(meta_clf, "./meta_model_gbm.pkl") # 保存路径也做了相应更改

    # ----- 10. 阈值选择（修改为代价敏感） -----
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    # 使用修改后的阈值搜索函数
    best_thresh = cost_sensitive_threshold(y_holdout, holdout_probas, **threshold_weights)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    f1_2_final = fbeta_score(y_holdout, y_pred_holdout, beta=1.2, zero_division=0)
    # 计算最终加权分数
    final_score = (
            threshold_weights['recall_weight'] * recall +
            threshold_weights['auc_weight'] * auc +
            threshold_weights['precision_weight'] * precision
    )

    print(f"最佳阈值 (基于加权分数)={best_thresh:.4f}")
    print(f"使用该阈值的性能指标:")
    print(f"  Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}")
    print(f"  F1={f1:.4f}, F1.2={f1_2_final:.4f}")
    print(
        f"  最终加权分数 (Recall*{threshold_weights['recall_weight']} + AUC*{threshold_weights['auc_weight']} + Precision*{threshold_weights['precision_weight']}) = {final_score:.5f}")

    # ----- 11. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "variance_selector": selector_variance,
        "tree_feature_ranking": ranked_idx_by_importance,  # 保留 RFE 排名
        "selected_feature_names": selected_feature_names_final,
        "base_models": wrapped_xgb_model,
        "base_types": ["XGBoost"],
        "meta_nb": meta_clf,  # Now stores an GradientBoostingClassifier instance
        "threshold": float(best_thresh),
        "use_rfa": use_rfa,  # 保存配置
        "threshold_weights": threshold_weights  # 保存阈值权重配置
    }
    save_object(model_dict, "./model.pkl")
    print("已保存完整模型 ./model.pkl")
    print("=" * 60)




