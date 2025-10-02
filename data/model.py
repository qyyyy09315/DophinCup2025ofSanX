import os
import pickle
import warnings

import numpy as np
import pandas as pd

# 导入 KNNImputer 和 特征选择工具
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
# 导入交叉验证相关的工具
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
        probas_base = base_model.predict(X).astype(float)

    preds = (probas_base >= threshold).astype(int)
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


def f1_2_threshold(y_true, probas):
    thresholds = np.linspace(0, 1, 101)
    best_t, best_f1_2 = 0.5, -1
    beta = 1.2

    for t in thresholds:
        preds = (probas >= t).astype(int)
        try:
            f1_2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        except ValueError:
            f1_2 = 0.0

        if f1_2 > best_f1_2:
            best_f1_2, best_t = f1_2, t
    return best_t


# --- WGAN-GP 相关定义 (含自适应注意力机制) ---

class AdaptiveAttention(nn.Module):
    """自适应注意力模块"""

    def __init__(self, data_dim):
        super(AdaptiveAttention, self).__init__()
        self.data_dim = data_dim
        # 查询、键、值的线性变换
        self.query = nn.Linear(data_dim, data_dim)
        self.key = nn.Linear(data_dim, data_dim)
        self.value = nn.Linear(data_dim, data_dim)
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([data_dim])).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        # x shape: (batch_size, data_dim)
        # Q, K, V shape: (batch_size, data_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数 (batch_size, data_dim) x (data_dim, batch_size) -> (batch_size, batch_size)
        # 为了简化，我们计算每个样本对自身的注意力，然后广播到所有特征
        # 更标准的做法是处理序列数据，这里我们将其视为单步序列
        # 我们将输入x扩展为(batch_size, 1, data_dim) 来模拟序列
        x_seq = x.unsqueeze(1)  # (batch_size, 1, data_dim)
        Q_seq = Q.unsqueeze(1)  # (batch_size, 1, data_dim)
        K_seq = K.unsqueeze(1)  # (batch_size, 1, data_dim)
        V_seq = V.unsqueeze(1)  # (batch_size, 1, data_dim)

        # 注意力权重 (batch_size, 1, 1)
        # 简化处理：计算整个样本的全局注意力权重
        attention_weights = torch.matmul(Q_seq, K_seq.transpose(-2, -1)) / self.scale  # (batch_size, 1, 1)
        attention_weights = torch.softmax(attention_weights, dim=-1)  # (batch_size, 1, 1)

        # 应用注意力权重到值 (batch_size, 1, data_dim)
        attended_values = torch.matmul(attention_weights, V_seq)  # (batch_size, 1, data_dim)

        # 压缩回 (batch_size, data_dim)
        attended_values = attended_values.squeeze(1)  # (batch_size, data_dim)

        # 残差连接和层归一化 (简化版，无层归一化)
        out = x + attended_values
        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.data_dim = data_dim
        # 注意力模块
        self.attention = AdaptiveAttention(data_dim)

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
        # 应用自适应注意力
        attended_output = self.attention(raw_output)
        return attended_output


class Critic(nn.Module):  # 判别器更名为Critic
    def __init__(self, data_dim):
        super(Critic, self).__init__()
        self.data_dim = data_dim
        # 注意力模块
        self.attention = AdaptiveAttention(data_dim)

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
        # 应用自适应注意力
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
        "开始训练：Variance Filter -> Balanced Random Forest (Select 80 features) -> WGAN-GP (含自适应注意力+动态学习率) -> XGBoost -> 级联 Logistic Regression (Tree-based 特征权重)")
    print("-> 已将 DNN 特征加权替换为 RFE (LogisticRegression L2)")
    print("-> 新增按特征重要性排序后选取 Top 90% 的特征")
    print("-> 阈值调优方法已修改为 F1.2")
    print("-> 关键修改: 采样方法由 ADASYN 改为 WGAN-GP")  # <-- 更新日志
    print("-> 关键修改: 级联修正器由 GaussianNB 改为带 L2 正则化的 LogisticRegression")
    print("-> 关键改进: 缺失值填充方法由均值填充改为KNN填充 (n_neighbors=5)")
    print("-> 关键改进: 在 RFE 阶段加入了交叉验证来评估特征子集性能")
    print("-> 新增功能: WGAN-GP中加入动态学习率调度 (ExponentialLR)") # <-- 新增日志
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
    top_percentile_to_select = 0.9
    num_features_to_select_brf = 90
    cv_folds_for_rfe = 3

    # WGAN-GP 配置
    wgangp_config = {
        'latent_dim': 100,
        'epochs': 300,  # 可根据需要调整
        'batch_size': 64,
        'lr': 0.0001,  # WGAN-GP 推荐较小的学习率
        'lambda_gp': 10,  # 梯度惩罚系数
        'n_critic': 5,  # 训练几次判别器才训练一次生成器
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr_decay_rate': 0.99 # 新增配置项
    }

    # XGBoost 超参
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

    # ---- 4. 平衡随机森林特征选择 ----
    print(f"使用平衡随机森林进行特征选择，选择 {num_features_to_select_brf} 个特征...")
    brf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1,
                                 class_weight='balanced')
    brf_selector = SelectFromModel(brf, max_features=num_features_to_select_brf, threshold=-np.inf)
    X_brf_selected = brf_selector.fit_transform(X_scaled, y_all)
    selected_feature_indices_brf = brf_selector.get_support(indices=True)
    selected_feature_names_brf = [selected_feature_names_variance[i] for i in selected_feature_indices_brf]
    print(f"平衡随机森林特征选择完成，剩余特征数: {X_brf_selected.shape[1]}")

    save_object(brf_selector, "./brf_feature_selector.pkl")

    # ----- 5. Tree-based 特征权重 (替代DNN) -> 改为 RFE with Cross-Validation-----
    print("使用 RFE (LogisticRegression L2) 结合交叉验证获取特征排名并选择 Top 特征 ...")
    estimator_rfe = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=random_state,
                                       max_iter=1000)

    skf_cv = StratifiedKFold(n_splits=cv_folds_for_rfe, shuffle=True, random_state=random_state)

    rfe_num_features_to_select = int(top_percentile_to_select * X_brf_selected.shape[1])
    if rfe_num_features_to_select <= 0:
        rfe_num_features_to_select = 1

    print(f"目标选定特征数: {rfe_num_features_to_select}")

    selector_rfe = RFE(estimator_rfe, n_features_to_select=rfe_num_features_to_select, step=5)
    selector_rfe.fit(X_brf_selected, y_all)

    # --- 新增部分：在选定特征上进行交叉验证评分 ---
    selected_top_feature_indices = selector_rfe.support_
    X_rfe_selected = X_brf_selected[:, selected_top_feature_indices]

    cv_scores = cross_val_score(estimator_rfe, X_rfe_selected, y_all, cv=skf_cv, scoring='roc_auc')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(
        f"RFE所选特征 ({rfe_num_features_to_select}个) 的交叉验证平均 AUC 得分: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")

    feature_ranking = selector_rfe.ranking_
    ranked_idx_by_importance = np.argsort(feature_ranking)

    X_selected = X_brf_selected[:, selected_top_feature_indices]
    original_brf_indices_of_selected = np.where(selected_top_feature_indices)[0]
    selected_feature_names_final = [selected_feature_names_brf[i] for i in original_brf_indices_of_selected]

    print(
        f"RFE 特征加权完成，并选择了 Top {top_percentile_to_select * 100}% ({len(original_brf_indices_of_selected)}/{X_brf_selected.shape[1]}) 的特征.")
    print(f"这些特征在交叉验证(AUC)下的表现约为: {mean_cv_score:.4f}")

    save_object(ranked_idx_by_importance, "./rfe_feature_ranking.pkl")

    # ----- 6. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # ----- 7. 过采样 (使用 WGAN-GP 替换 ADASYN/GAN) -----
    print(f"正在进行 WGAN-GP (含自适应注意力+动态学习率) 过采样 ...")
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
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost.pkl")
    print("已训练基模型：XGBoost")

    # ----- 9. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器 (使用带 L2 正则化的 Logistic Regression)...")
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:
        meta_clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=random_state,
                                      max_iter=1000)
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器 (Logistic Regression) 训练完成，困难样本数={len(X_hard)}")

        # --- 新增部分：也可以对元模型做个简单的CV评估---
        try:
            meta_cv_scores = cross_val_score(meta_clf, X_hard, y_hard, cv=min(3, len(X_hard) // 2),
                                             scoring='f1')  # 至少两折
            meta_mean_cv_score = np.mean(meta_cv_scores)
            print(f"Meta-model CV F1 Score on hard examples: {meta_mean_cv_score:.4f}")
        except Exception as e:
            print(f"无法计算 Meta-model 的 CV 分数: {e}")

    else:
        meta_clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=random_state,
                                      max_iter=1000)
        meta_clf.fit(X_resampled, y_resampled)
        print("未发现足够的误分类样本，使用全部重采样数据训练修正器 (Logistic Regression)")

    # ----- 10. 阈值选择（修改为 F1.2） -----
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    best_thresh = f1_2_threshold(y_holdout, holdout_probas)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    f1_2_final = fbeta_score(y_holdout, y_pred_holdout, beta=1.2, zero_division=0)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"最佳阈值 (基于 F1.2)={best_thresh:.4f}, F1={f1:.4f}, F1.2={f1_2_final:.4f}")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, FinalScore={final_score:.5f}")

    # ----- 11. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "variance_selector": selector_variance,
        "brf_feature_selector": brf_selector,
        "tree_feature_ranking": ranked_idx_by_importance,
        "selected_feature_names": selected_feature_names_final,
        "base_models": wrapped_xgb_model,
        "base_types": ["XGBoost"],
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model.pkl")
    print("已保存完整模型 ./model.pkl")
    print("=" * 60)




