import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA  # 新增 PCA

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


# ------------------- Focal Loss（添加 epsilon 防止数值下溢） -------------------
# 保留 FocalLoss 类，虽然 PCA 版本不使用，但为兼容性保留
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss.clamp(max=50))  # 防止梯度爆炸
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# ------------------- EarlyStopping（保留，虽然 PCA 版本不使用，但为兼容性保留） -------------------
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


# ------------------- 级联模型类（支持动态阈值）-------------------
class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, scaler, pca, classifiers, threshold=0.5):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.scaler = scaler
        self.pca = pca  # 替换 feature_extractor 为 pca
        self.classifiers = classifiers  # [adaboost, xgboost]
        self.threshold = threshold  # 动态设定的分类阈值

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_scaled = self.scaler.transform(X_sel)
        return X_scaled

    def predict(self, X_raw):
        proba = self.predict_proba(X_raw)[:, 1]
        return (proba >= self.threshold).astype(int).ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.pca.transform(X_scaled)  # 使用 PCA 降维
        probas = [clf.predict_proba(emb)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])

    def set_threshold(self, threshold):
        self.threshold = threshold


# ------------------- 阈值搜索函数（优化 G-Mean，兼顾 Recall 和 Precision）-------------------
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


# ------------------- 主流程 -------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，将使用 CPU 训练。")

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

    # 删除常量特征
    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    # VarianceThreshold
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)

    # ADASYN 过采样（仅在训练集上）
    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_selected, y_train)

    # 标准化
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # 预处理验证集（用于 PCA 适配 & 阈值搜索）
    X_val_imp = imputer.transform(X_val)
    X_val_non_constant = X_val_imp[:, non_constant_features]
    X_val_selected = selector.transform(X_val_non_constant)
    X_val_scaled = scaler.transform(X_val_selected)

    # 使用 PCA 进行降维（替代原来的神经网络特征提取器）
    # 保留95%的方差或指定 n_components，这里我们指定为 50（可根据数据调整）
    n_components = min(50, X_resampled_scaled.shape[1])  # 最多50维，或特征数
    pca = PCA(n_components=n_components, random_state=SEED)
    X_emb_train = pca.fit_transform(X_resampled_scaled)
    X_emb_val = pca.transform(X_val_scaled)

    print(f"[PCA] 降维后特征维度: {X_emb_train.shape[1]}")
    print(f"[PCA] 解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")

    # 计算样本权重（风险敏感：正样本权重更高）
    sample_weights = np.ones(len(y_resampled))
    sample_weights[y_resampled == 1] = 2.0  # 风险敏感系数

    # AdaBoost 分类器（使用样本权重）
    adaboost = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=SEED
    )
    adaboost.fit(X_emb_train, y_resampled, sample_weight=sample_weights)

    # XGBoost 分类器（使用样本权重）
    xgboost = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgboost.fit(X_emb_train, y_resampled, sample_weight=sample_weights)

    # 在验证集上搜索最优阈值（优化 G-Mean）
    y_proba_val_adaboost = adaboost.predict_proba(X_emb_val)[:, 1]
    y_proba_val_xgboost = xgboost.predict_proba(X_emb_val)[:, 1]
    y_proba_val = (y_proba_val_adaboost + y_proba_val_xgboost) / 2

    best_threshold, best_gmean = find_optimal_threshold(y_val, y_proba_val)
    print(f"[Threshold Search] 最佳阈值: {best_threshold:.4f}, G-Mean: {best_gmean:.6f}")

    # 预测测试集
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_scaled = scaler.transform(X_test_selected)
    X_emb_test = pca.transform(X_test_scaled)

    y_proba_adaboost = adaboost.predict_proba(X_emb_test)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_emb_test)[:, 1]
    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= best_threshold).astype(int)

    # 评估指标
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

    # 保存模型与预处理器（包含阈值）
    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        scaler=scaler,
        pca=pca,  # 替换为 pca
        classifiers=[adaboost, xgboost],
        threshold=best_threshold
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")
    joblib.dump(pca, "pca.pkl")  # 新增保存 PCA

    print("✅ 模型和预处理器已保存。")


if __name__ == "__main__":
    main()