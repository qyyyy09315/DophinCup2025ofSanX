import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
# 从 combine 模块导入 SMOTETomek
from imblearn.combine import SMOTETomek # 或者 from imblearn.combine import SMOTEENN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier # 导入 CatBoost
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
# 导入 LightGBM
import lightgbm as lgb

# 设置环境变量 (根据你的实际CPU核心数调整)
os.environ["LOKY_MAX_CPU_COUNT"] = "32"  # 假设你使用的是32核CPU
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- 硬编码的最佳超参数 (保留用于模型初始化，但不再进行网格搜索) ---
ADA_BEST_PARAMS = {
    'learning_rate': 0.4262213204002109,
    'n_estimators': 393
}

XGB_BEST_PARAMS = {
    'colsample_bytree': 0.7123738038749523,
    'learning_rate': 0.17280882494747454,
    'max_depth': 11,
    'n_estimators': 356,
    'subsample': 0.9208787923016158
}

# --- 固定的预处理和模型参数 ---
FIXED_PARAMS = {
    'inter_intra_threshold': 0.001,  # 固定 InterIntraDistanceSelector 阈值
    # 'adasyn_n_neighbors': 5,         # 移除 ADASYN 参数
    'lgb_top_ratio': 0.8             # 固定 LightGBM 选择特征比例
}

# --- 显式指定的最终分类概率阈值 ---
FINAL_PROBABILITY_THRESHOLD = 0.1

# --------------------------


class InterIntraDistanceSelector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("目前只支持二分类任务")

        class_0_indices = (y == classes[0])
        class_1_indices = (y == classes[1])
        X_0 = X[class_0_indices]
        X_1 = X[class_1_indices]

        intra_class_var_0 = np.var(X_0, axis=0)
        intra_class_var_1 = np.var(X_1, axis=0)
        intra_class_variance_sum = intra_class_var_0 + intra_class_var_1

        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        inter_class_distance_sq = (mean_1 - mean_0) ** 2

        scores = np.divide(inter_class_distance_sq, intra_class_variance_sum,
                           out=np.zeros_like(inter_class_distance_sq),
                           where=intra_class_variance_sum != 0)

        self.selected_features_ = scores > self.threshold
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class FeatureSelector: # 重命名类以反映通用性
    """用于包装特征选择掩码的标准类"""

    def __init__(self, selected_features_mask):
        self.selected_features_ = selected_features_mask

    def transform(self, X):
        return X[:, self.selected_features_]


def train_and_evaluate_fixed_params(X_train_all, y_train_all, X_test, y_test):
    """
    使用固定参数进行数据预处理、特征选择、模型训练和评估。
    """
    print("[Process] 开始使用固定参数进行处理和评估...")

    # --- 1. 初始预处理 (在整个训练集上执行) ---
    print("[Preprocessing] 执行初始预处理步骤...")
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)

    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)
    print(f"[Preprocessing] 初始预处理完成。特征数: {X_train_selected.shape[1]}")

    # --- 2. 应用 InterIntraDistanceSelector (使用固定参数) ---
    print(f"[Feature Selection] 应用 InterIntraDistanceSelector (threshold={FIXED_PARAMS['inter_intra_threshold']})...")
    inter_intra_selector = InterIntraDistanceSelector(threshold=FIXED_PARAMS['inter_intra_threshold'])
    X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
    print(f"[Feature Selection] InterIntraDistanceSelector 选择后特征数: {X_train_inter_intra_selected.shape[1]}")

    if X_train_inter_intra_selected.shape[1] < 2:
        raise ValueError("InterIntraDistanceSelector 选择的特征数过少 (<2)，无法继续。")

    # --- 3. 应用 LightGBM 特征选择 (使用固定参数) ---
    print(f"[Feature Selection] 应用 LightGBM 特征选择 (top_ratio={FIXED_PARAMS['lgb_top_ratio']})...")
    # 使用 LightGBM 进行特征重要性评估
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        random_state=SEED,
        verbose=-1, # 静默训练
        max_depth=10
    )
    lgb_model.fit(X_train_inter_intra_selected, y_train_all)
    feature_importances = lgb_model.feature_importances_
    num_features_to_select = max(1, int(FIXED_PARAMS['lgb_top_ratio'] * X_train_inter_intra_selected.shape[1]))
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_indices = sorted_indices[:num_features_to_select]
    lgb_selected_features_mask = np.zeros(X_train_inter_intra_selected.shape[1], dtype=bool)
    lgb_selected_features_mask[top_indices] = True
    # 使用通用的 FeatureSelector 类
    feature_selector = FeatureSelector(lgb_selected_features_mask)

    X_train_lgb_selected = feature_selector.transform(X_train_inter_intra_selected)
    print(f"[Feature Selection] LightGBM 选择后特征数: {X_train_lgb_selected.shape[1]}")

    # --- 4. 标准化 ---
    print("[Preprocessing] 标准化特征...")
    scaler = StandardScaler()
    X_train_lgb_selected_scaled = scaler.fit_transform(X_train_lgb_selected)

    # --- 5. 过采样 (SMOTETomek) ---
    print(f"[Resampling] 应用 SMOTETomek 过采样...")
    # 使用 SMOTETomek 进行过采样和欠采样
    smote_tomek = SMOTETomek(random_state=SEED)
    X_resampled_scaled, y_resampled = smote_tomek.fit_resample(X_train_lgb_selected_scaled, y_train_all)
    print(f"[Resampling] SMOTETomek 采样后样本数: {X_resampled_scaled.shape[0]}")

    # --- 6. 模型训练 ---
    print("[Training] 训练 AdaBoost 模型...")
    adaboost = AdaBoostClassifier(
        n_estimators=ADA_BEST_PARAMS['n_estimators'],
        learning_rate=ADA_BEST_PARAMS['learning_rate'],
        random_state=SEED
    )
    adaboost.fit(X_resampled_scaled, y_resampled)

    print("[Training] 训练 XGBoost 模型...")
    xgboost = XGBClassifier(
        n_estimators=XGB_BEST_PARAMS['n_estimators'],
        max_depth=XGB_BEST_PARAMS['max_depth'],
        learning_rate=XGB_BEST_PARAMS['learning_rate'],
        subsample=XGB_BEST_PARAMS['subsample'],
        colsample_bytree=XGB_BEST_PARAMS['colsample_bytree'],
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgboost.fit(X_resampled_scaled, y_resampled)

    print("[Training] 训练 CatBoost 模型...")
    # CatBoost 通常能很好地处理类别特征和不平衡数据，这里使用一些基础调优参数
    catboost = CatBoostClassifier(
        iterations=500, # 可根据需要调整
        depth=13,       # 可根据需要调整
        learning_rate=0.8, # 可根据需要调整
        loss_function='Logloss',
        eval_metric='Precision',
        random_seed=SEED,
        verbose=0 # 设置为 0 以避免训练过程中的大量输出
    )
    catboost.fit(X_resampled_scaled, y_resampled, silent=True) # silent=True 也用于抑制输出


    # --- 7. 测试集预处理 ---
    print("[Evaluation] 对测试集应用相同的预处理步骤...")
    # 应用初始预处理
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)

    # 应用 InterIntraDistanceSelector
    X_test_inter_intra_selected = inter_intra_selector.transform(X_test_selected)

    # 应用 LightGBM 特征选择
    X_test_lgb_selected = feature_selector.transform(X_test_inter_intra_selected)

    # 标准化
    X_test_scaled = scaler.transform(X_test_lgb_selected)

    # --- 8. 测试集预测 ---
    print("[Evaluation] 在测试集上进行预测...")
    y_proba_adaboost = adaboost.predict_proba(X_test_scaled)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_test_scaled)[:, 1]
    y_proba_catboost = catboost.predict_proba(X_test_scaled)[:, 1]

    # --- 9. 集成概率 (平均三个模型的概率) ---
    print("[Evaluation] 集成三个模型的预测概率...")
    y_proba = (y_proba_adaboost + y_proba_xgboost + y_proba_catboost) / 3

    # --- 10. 使用显式指定的阈值进行分类 ---
    print(f"[Evaluation] 使用阈值 {FINAL_PROBABILITY_THRESHOLD} 进行最终分类...")
    y_pred = (y_proba >= FINAL_PROBABILITY_THRESHOLD).astype(int)

    # --- 11. 评估 ---
    print("[Evaluation] 计算测试集评估指标...")
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0) # 防止 Precision 为 0/0 时警告
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"\n[Final Results] 测试集评估结果 (阈值={FINAL_PROBABILITY_THRESHOLD}):")
    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    return {
        'recall': recall,
        'auc': auc,
        'precision': precision,
        'final_score': final_score,
        'y_proba': y_proba,
        'y_pred': y_pred
    }


def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"显存已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    else:
        print("CUDA 不可用，将使用 CPU 训练。")

    # 打印并行设置
    print(f"设置 LOKY_MAX_CPU_COUNT 为: {os.environ.get('LOKY_MAX_CPU_COUNT', '未设置')}")

    # --- 数据加载 ---
    print("正在加载数据 'clean.csv'...")
    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
        print("已移除 'company_id' 列。")
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values
    print(f"数据加载完成。特征矩阵形状: {X.shape}, 目标向量形状: {y.shape}")

    if np.isnan(y).any():
        print("检测到目标变量 'y' 中存在 NaN，正在移除对应的样本...")
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        print(f"移除 NaN 后，数据集大小: X={X.shape}, y={y.shape}")

    # --- 数据划分 ---
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.05, random_state=SEED, stratify=y)
    print(f"数据集划分完成: 训练集 {X_train_all.shape}, 测试集 {X_test.shape}")

    # --- 使用固定参数进行处理和评估 ---
    results = train_and_evaluate_fixed_params(X_train_all, y_train_all, X_test, y_test)


if __name__ == "__main__":
    main()



