import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import chi2
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# 设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)


# ------------------- Focal Loss（保留兼容性） -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss.clamp(max=50))
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# ------------------- EarlyStopping（保留兼容性） -------------------
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = np.inf if monitor == 'loss' else -np.inf
        self.wait = 0

    def step(self, value):
        stop = False
        if self.monitor == 'loss':
            if (self.best - value) > self.min_delta:
                self.best = value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
        else:
            if (value - self.best) > self.min_delta:
                self.best = value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
        return stop


# ------------------- 对比学习编码网络 -------------------
class ContrastiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)  # 归一化输出，便于对比学习
        )

    def forward(self, x):
        return self.encoder(x)


# ------------------- NT-Xent 对比损失函数 -------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (batch_size, output_dim) — 同一样本的两个增强视图
        """
        batch_size = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)  # (2*batch_size, output_dim)
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # 掩码对角线（自身）
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # 构造标签：每个样本的正样本是其配对增强视图
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z.device)  # 正样本索引

        loss = self.criterion(sim_matrix, labels)
        return loss


# ------------------- 数据增强函数（简单噪声注入） -------------------
def add_noise(x, noise_factor=0.1):
    noise = torch.randn_like(x) * noise_factor
    return x + noise


# ------------------- 训练对比学习编码器（新增早停） -------------------
def train_contrastive_encoder(X_train_scaled, device, epochs=1000, batch_size=256, lr=1e-3):
    input_dim = X_train_scaled.shape[1]
    model = ContrastiveNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NTXentLoss(temperature=0.5)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=30, min_delta=1e-5, monitor='loss')

    # 转换为张量
    X_tensor = torch.FloatTensor(X_train_scaled).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            x1 = add_noise(x)
            x2 = add_noise(x)

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        # ✅ 早停检查
        if early_stopping.step(avg_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"Contrastive Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    model.eval()
    return model


# ------------------- 级联模型类（支持动态阈值）-------------------
class CascadedModel:
    def __init__(self, imputer, non_constant_features, scaler, contrastive_encoder, classifiers, threshold=0.5):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.scaler = scaler
        self.contrastive_encoder = contrastive_encoder  # 替代 pca
        self.classifiers = classifiers  # [adaboost, xgboost]
        self.threshold = threshold
        self.device = next(contrastive_encoder.parameters()).device

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_scaled = self.scaler.transform(X_masked)
        return X_scaled

    def _encode_features(self, X_scaled):
        self.contrastive_encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            emb = self.contrastive_encoder(X_tensor).cpu().numpy()
        return emb

    def predict(self, X_raw):
        proba = self.predict_proba(X_raw)[:, 1]
        return (proba >= self.threshold).astype(int).ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self._encode_features(X_scaled)
        probas = [clf.predict_proba(emb)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])

    def set_threshold(self, threshold):
        self.threshold = threshold


# ------------------- 阈值搜索函数（优化 G-Mean）-------------------
def find_optimal_threshold(y_true, y_proba, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_threshold = 0.5
    best_gmean = 0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        gmean = np.sqrt(rec * prec) if rec > 0 and prec > 0 else 0
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = th
    return best_threshold, best_gmean


# ------------------- 计算类别特异性特征重要性（近似方法） -------------------
def compute_class_specific_feature_importance(X, y, model, feature_names=None):
    proba = model.predict_proba(X)[:, 1]  # 正类概率
    importances = model.feature_importances_  # 全局重要性

    # 获取正负样本索引
    pos_idx = y == 1
    neg_idx = y == 0

    if pos_idx.sum() == 0 or neg_idx.sum() == 0:
        print("某一类样本为空，无法计算类别特异性重要性")
        return None, None

    # 对正样本：用预测概率加权特征重要性（概率越高，该样本的特征越“重要”）
    pos_weighted_importance = np.mean(
        proba[pos_idx][:, np.newaxis] * importances, axis=0
    )
    # 对负样本：用 (1 - 预测概率) 加权（预测为负的概率越高，特征越重要）
    neg_weighted_importance = np.mean(
        (1 - proba[neg_idx])[:, np.newaxis] * importances, axis=0
    )

    # 归一化
    pos_weighted_importance /= np.sum(pos_weighted_importance) + 1e-8
    neg_weighted_importance /= np.sum(neg_weighted_importance) + 1e-8

    if feature_names is not None:
        pos_df = pd.DataFrame({
            'feature': feature_names,
            'importance': pos_weighted_importance
        }).sort_values('importance', ascending=False)
        neg_df = pd.DataFrame({
            'feature': feature_names,
            'importance': neg_weighted_importance
        }).sort_values('importance', ascending=False)
        return pos_df, neg_df
    else:
        return pos_weighted_importance, neg_weighted_importance


# ------------------- 主流程（无贝叶斯优化）-------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("CUDA 不可用，将使用 CPU 训练。")
        device = torch.device("cpu")

    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    # 划分训练集（含验证集）和测试集
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=SEED, stratify=y_train_all)

    # 填补缺失值
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)

    # 删除常量特征（基于原始训练集）✅ 保留这一部分，作为固定特征集
    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    # ========== ✅ 新增：卡方检验特征筛选 ==========
    print("\n[Feature Selection] 使用卡方检验筛选显著相关特征...")

    # 对连续特征进行离散化（用于卡方检验）
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_non_constant)

    # 卡方检验（要求非负整数输入）
    chi2_scores, p_values = chi2(X_train_discretized, y_train)

    # 选择 p < 0.05 的特征
    chi2_selected = p_values < 0.05
    print(f"卡方检验显著特征数量: {chi2_selected.sum()} / {len(chi2_selected)}")

    # 构建最终特征掩码（在 non_constant_features 基础上进一步筛选）
    # 注意：chi2_selected 是针对 non_constant_features 子集的，需映射回原始特征空间
    final_selected_mask = np.zeros_like(non_constant_features, dtype=bool)
    final_selected_mask[non_constant_features] = chi2_selected  # 关键：映射回原维度

    # 如果卡方检验未选中任何特征，则回退到 non_constant_features
    if not np.any(final_selected_mask):
        print("⚠️  警告：卡方检验未选中任何特征，回退到非恒定特征。")
        final_selected_mask = non_constant_features

    print(f"最终用于建模的特征数量: {final_selected_mask.sum()}")

    # 应用卡方筛选后的特征
    X_train_selected = X_train_imp[:, final_selected_mask]

    # ========== 新采样策略：ADASYN 过采样 + TomekLinks 欠采样 ========== ✅ 修改点
    print("\n[Sampling] 第一阶段：使用 ADASYN 对少数类过采样...")

    # 自动平衡采样（也可指定 sampling_strategy）
    adasyn = ADASYN(random_state=SEED)
    X_ada, y_ada = adasyn.fit_resample(X_train_selected, y_train)

    print(f"ADASYN后: {len(y_ada)} 样本 (正例: {y_ada.sum()})")

    print("\n[Sampling] 第二阶段：使用 TomekLinks 清理边界噪声样本...")
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X_ada, y_ada)

    print(f"TomekLinks后: {len(y_resampled)} 样本 (正例: {y_resampled.sum()})")

    # ========== ✅ 使用 final_selected_mask 代替 non_constant_features ==========
    X_resampled_selected = X_resampled  # 已经是筛选后的特征

    # 标准化（基于重采样后数据拟合）
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled_selected)

    # 预处理验证集（使用相同的 final_selected_mask 掩码）
    X_val_imp = imputer.transform(X_val)
    X_val_selected = X_val_imp[:, final_selected_mask]
    X_val_scaled = scaler.transform(X_val_selected)

    # ========== ✅ 新增：训练对比学习编码器 ==========
    print("\n[Contrastive Learning] 开始训练对比学习编码器...")
    contrastive_encoder = train_contrastive_encoder(X_resampled_scaled, device, epochs=1000, batch_size=256, lr=1e-3)

    # 使用对比学习编码器提取特征
    def encode_with_contrastive(encoder, X, device):
        encoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            emb = encoder(X_tensor).cpu().numpy()
        return emb

    X_emb_train = encode_with_contrastive(contrastive_encoder, X_resampled_scaled, device)
    X_emb_val = encode_with_contrastive(contrastive_encoder, X_val_scaled, device)

    print(f"[Contrastive] 编码后特征维度: {X_emb_train.shape[1]}")

    # ========== 直接设定模型参数（替代贝叶斯优化）==========
    # AdaBoost 参数（常用设置）
    adaboost_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'random_state': SEED
    }

    # XGBoost 参数（常用设置）
    xgboost_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }

    print("\n[Training] 使用固定参数训练 AdaBoost...")
    adaboost = AdaBoostClassifier(**adaboost_params)
    adaboost.fit(X_emb_train, y_resampled)

    print("\n[Training] 使用固定参数训练 XGBoost...")
    xgboost = XGBClassifier(**xgboost_params)
    xgboost.fit(X_emb_train, y_resampled)

    # ========== 在验证集上搜索最优阈值 ==========
    y_proba_val_adaboost = adaboost.predict_proba(X_emb_val)[:, 1]
    y_proba_val_xgboost = xgboost.predict_proba(X_emb_val)[:, 1]
    y_proba_val = (y_proba_val_adaboost + y_proba_val_xgboost) / 2

    best_threshold, best_gmean = find_optimal_threshold(y_val, y_proba_val)
    print(f"\n[Threshold Search] 最佳阈值: {best_threshold:.4f}, G-Mean: {best_gmean:.6f}")

    # ========== 计算类别特异性特征重要性 ==========
    print("\n" + "="*60)
    print("计算类别特异性特征重要性（近似方法）...")
    print("="*60)

    # 获取对比学习后的“特征名”（虚拟）
    contrastive_feature_names = [f"Contrastive_Feature_{i+1}" for i in range(X_emb_train.shape[1])]

    # 对 AdaBoost
    print("\n[AdaBoost] 类别特异性特征重要性（基于对比学习特征）:")
    pos_imp_adaboost, neg_imp_adaboost = compute_class_specific_feature_importance(
        X_emb_val, y_val, adaboost, feature_names=contrastive_feature_names
    )
    if pos_imp_adaboost is not None:
        print("\n→ 正样本重要特征 (Top 5):")
        print(pos_imp_adaboost.head(5))
        print("\n→ 负样本重要特征 (Top 5):")
        print(neg_imp_adaboost.head(5))

    # 对 XGBoost
    print("\n[XGBoost] 类别特异性特征重要性（基于对比学习特征）:")
    pos_imp_xgboost, neg_imp_xgboost = compute_class_specific_feature_importance(
        X_emb_val, y_val, xgboost, feature_names=contrastive_feature_names
    )
    if pos_imp_xgboost is not None:
        print("\n→ 正样本重要特征 (Top 5):")
        print(pos_imp_xgboost.head(5))
        print("\n→ 负样本重要特征 (Top 5):")
        print(neg_imp_xgboost.head(5))

    # ========== 预测测试集 ==========
    X_test_imp = imputer.transform(X_test)
    X_test_selected = X_test_imp[:, final_selected_mask]  # ✅ 使用 final_selected_mask
    X_test_scaled = scaler.transform(X_test_selected)
    X_emb_test = encode_with_contrastive(contrastive_encoder, X_test_scaled, device)

    y_proba_adaboost = adaboost.predict_proba(X_emb_test)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_emb_test)[:, 1]
    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= best_threshold).astype(int)

    # ========== 评估指标 ==========
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print("=" * 50)
    print("最终评估结果：")
    print(f"Recall:    {recall:.6f}")
    print(f"AUC:       {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"F1-Score:  {f1:.6f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"使用阈值: {best_threshold:.4f}（基于验证集 G-Mean 优化）")
    print("=" * 50)

    # ========== 保存模型与预处理器 ==========
    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=final_selected_mask,  # ✅ 保存最终筛选后的特征掩码
        scaler=scaler,
        contrastive_encoder=contrastive_encoder,
        classifiers=[adaboost, xgboost],
        threshold=best_threshold
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(final_selected_mask, "selected_features_mask.pkl")  # ✅ 保存最终特征掩码
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")      # ✅ 保存 ADASYN 采样器
    joblib.dump(tomek, "tomek_links_sampler.pkl")  # ✅ 保存 TomekLinks 采样器
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")
    # 保存对比学习模型（需额外保存为 .pt）
    torch.save(contrastive_encoder.state_dict(), "contrastive_encoder.pth")
    joblib.dump(adaboost_params, "adaboost_params.pkl")
    joblib.dump(xgboost_params, "xgboost_params.pkl")

    print("✅ 模型和预处理器及参数已保存。")


if __name__ == "__main__":
    main()