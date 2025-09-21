import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier

# 设置环境变量 (根据你的实际CPU核心数调整)
os.environ["LOKY_MAX_CPU_COUNT"] = "32"  # 假设你使用的是32核CPU
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- 硬编码的最佳超参数 ---
# 来自 AdaBoost 随机搜索的最佳参数
ADA_BEST_PARAMS = {
    'learning_rate': 0.4262213204002109,
    'n_estimators': 393
}

# 来自 XGBoost 随机搜索的最佳参数
XGB_BEST_PARAMS = {
    'colsample_bytree': 0.7123738038749523,
    'learning_rate': 0.17280882494747454,
    'max_depth': 11,
    'n_estimators': 356,
    'subsample': 0.9208787923016158
}


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


class BrfSelector:
    """用于包装 Balanced Random Forest 选择的特征掩码的标准类"""

    def __init__(self, selected_features_mask):
        self.selected_features_ = selected_features_mask

    def transform(self, X):
        return X[:, self.selected_features_]


def perform_grid_search(X_train_all, y_train_all, X_test, y_test, param_grid, cv_folds=3):
    """
    对 ADASYN 和特征选择器进行网格搜索。
    """
    print(f"[Grid Search] 开始网格搜索，参数网格: {param_grid}")
    best_score = -np.inf
    best_params = None

    # --- 第一步：预处理 (在整个训练集上执行一次) ---
    print("[Preprocessing] 执行初始预处理步骤...")
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)

    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)
    print(f"[Preprocessing] 初始预处理完成。特征数: {X_train_selected.shape[1]}")

    # --- 网格搜索主循环 ---
    # 遍历 InterIntraDistanceSelector 的 threshold
    for inter_intra_threshold in param_grid['inter_intra_threshold']:
        print(f"\n[Grid Search] 尝试 InterIntraDistanceSelector.threshold = {inter_intra_threshold}")
        inter_intra_selector = InterIntraDistanceSelector(threshold=inter_intra_threshold)
        try:
            X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
        except Exception as e:
            print(f"[Grid Search] InterIntraDistanceSelector 失败 (threshold={inter_intra_threshold}): {e}")
            continue

        print(f"[Grid Search] InterIntraDistanceSelector 选择后特征数: {X_train_inter_intra_selected.shape[1]}")
        if X_train_inter_intra_selected.shape[1] < 2:
            print("[Grid Search] 特征数过少 (<2)，跳过此阈值。")
            continue

        # 遍历 ADASYN n_neighbors 和 BRF top_ratio
        for n_neighbors in param_grid['adasyn_n_neighbors']:
            for top_ratio in param_grid['brf_top_ratio']:
                print(f"  -> 尝试 ADASYN.n_neighbors = {n_neighbors}, BRF top_ratio = {top_ratio}")

                # --- 交叉验证 ---
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
                fold_scores = []

                for fold, (train_index, val_index) in enumerate(skf.split(X_train_inter_intra_selected, y_train_all)):
                    # print(f"    -> Fold {fold+1}/{cv_folds}")

                    X_fold_train = X_train_inter_intra_selected[train_index]
                    y_fold_train = y_train_all[train_index]
                    X_fold_val = X_train_inter_intra_selected[val_index]
                    y_fold_val = y_train_all[val_index]

                    # --- 特征选择 (BalancedRandomForest) ---
                    try:
                        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
                        brf.fit(X_fold_train, y_fold_train)
                        feature_importances = brf.feature_importances_
                        num_features_to_select = max(1, int(top_ratio * X_fold_train.shape[1]))
                        sorted_indices = np.argsort(feature_importances)[::-1]
                        top_indices = sorted_indices[:num_features_to_select]
                        brf_selected_features_mask = np.zeros(X_fold_train.shape[1], dtype=bool)
                        brf_selected_features_mask[top_indices] = True
                        brf_selector = BrfSelector(brf_selected_features_mask)

                        X_fold_train_brf = brf_selector.transform(X_fold_train)
                        X_fold_val_brf = brf_selector.transform(X_fold_val)
                    except Exception as e:
                        # print(f"      -> BRF 特征选择失败: {e}")
                        fold_scores.append(0)  # 或者跳过此 fold
                        continue

                    # --- 标准化 ---
                    try:
                        scaler = StandardScaler()
                        X_fold_train_scaled = scaler.fit_transform(X_fold_train_brf)
                        X_fold_val_scaled = scaler.transform(X_fold_val_brf)
                    except Exception as e:
                        # print(f"      -> 标准化失败: {e}")
                        fold_scores.append(0)
                        continue

                    # --- 过采样 (ADASYN) ---
                    try:
                        adasyn = ADASYN(random_state=SEED, n_neighbors=n_neighbors)
                        X_fold_resampled_scaled, y_fold_resampled = adasyn.fit_resample(X_fold_train_scaled,
                                                                                        y_fold_train)
                    except Exception as e:
                        # print(f"      -> ADASYN 过采样失败: {e}")
                        fold_scores.append(0)
                        continue

                    # --- 模型训练与预测 ---
                    try:
                        # AdaBoost
                        adaboost = AdaBoostClassifier(
                            n_estimators=ADA_BEST_PARAMS['n_estimators'],
                            learning_rate=ADA_BEST_PARAMS['learning_rate'],
                            random_state=SEED
                        )
                        adaboost.fit(X_fold_resampled_scaled, y_fold_resampled)
                        y_proba_adaboost = adaboost.predict_proba(X_fold_val_scaled)[:, 1]

                        # XGBoost
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
                        xgboost.fit(X_fold_resampled_scaled, y_fold_resampled)
                        y_proba_xgboost = xgboost.predict_proba(X_fold_val_scaled)[:, 1]

                        # 平均概率
                        y_proba_fold = (y_proba_adaboost + y_proba_xgboost) / 2

                        # 计算 AUC
                        fold_auc = roc_auc_score(y_fold_val, y_proba_fold)
                        fold_scores.append(fold_auc)
                        # print(f"      -> Fold {fold+1} AUC: {fold_auc:.4f}")

                    except Exception as e:
                        # print(f"      -> 模型训练或预测失败: {e}")
                        fold_scores.append(0)
                        continue

                # --- 计算平均得分 ---
                if fold_scores:
                    avg_score = np.mean(fold_scores)
                    print(f"  -> 平均 AUC (CV): {avg_score:.6f}")
                else:
                    avg_score = -np.inf
                    print(f"  -> 所有 folds 失败，平均 AUC: {avg_score}")

                # --- 更新最佳参数 ---
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'inter_intra_threshold': inter_intra_threshold,
                        'adasyn_n_neighbors': n_neighbors,
                        'brf_top_ratio': top_ratio
                    }
                    print(f"  -> [New Best] AUC: {best_score:.6f}, Params: {best_params}")

    print(f"\n[Grid Search] 网格搜索完成。")
    print(f"[Grid Search] 最佳 AUC (CV): {best_score:.6f}")
    print(f"[Grid Search] 最佳参数: {best_params}")

    # --- 使用最佳参数在完整训练集上训练并评估 ---
    if best_params:
        print(f"\n[Evaluation] 使用最佳参数在完整训练集上训练最终模型并评估...")
        final_auc, final_recall, final_precision, final_score = evaluate_with_best_params(
            X_train_all, y_train_all, X_test, y_test, best_params, imputer, non_constant_features, selector
        )
        print(f"\n[Final Evaluation] 测试集结果:")
        print(f"Recall: {final_recall:.6f}")
        print(f"AUC: {final_auc:.6f}")
        print(f"Precision: {final_precision:.6f}")
        print(f"Final Score: {final_score:.4f}")

    return best_params, best_score


def evaluate_with_best_params(X_train_all, y_train_all, X_test, y_test, best_params, imputer, non_constant_features,
                              selector):
    """
    使用最佳参数在完整训练集上训练模型，并在测试集上评估。
    """
    # --- 1. 应用初始预处理 ---
    X_train_imp = imputer.transform(X_train_all)
    X_train_non_constant = X_train_imp[:, non_constant_features]
    X_train_selected = selector.transform(X_train_non_constant)

    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)

    # --- 2. 应用 InterIntraDistanceSelector ---
    inter_intra_selector = InterIntraDistanceSelector(threshold=best_params['inter_intra_threshold'])
    X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
    X_test_inter_intra_selected = inter_intra_selector.transform(X_test_selected)

    # --- 3. 应用 BalancedRandomForest 特征选择 ---
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    brf.fit(X_train_inter_intra_selected, y_train_all)
    feature_importances = brf.feature_importances_
    num_features_to_select = max(1, int(best_params['brf_top_ratio'] * X_train_inter_intra_selected.shape[1]))
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_indices = sorted_indices[:num_features_to_select]
    brf_selected_features_mask = np.zeros(X_train_inter_intra_selected.shape[1], dtype=bool)
    brf_selected_features_mask[top_indices] = True
    brf_selector = BrfSelector(brf_selected_features_mask)

    X_train_brf_selected = brf_selector.transform(X_train_inter_intra_selected)
    X_test_brf_selected = brf_selector.transform(X_test_inter_intra_selected)

    # --- 4. 标准化 ---
    scaler = StandardScaler()
    X_train_brf_selected_scaled = scaler.fit_transform(X_train_brf_selected)
    X_test_scaled = scaler.transform(X_test_brf_selected)

    # --- 5. 过采样 (ADASYN) ---
    adasyn = ADASYN(random_state=SEED, n_neighbors=best_params['adasyn_n_neighbors'])
    X_resampled_scaled, y_resampled = adasyn.fit_resample(X_train_brf_selected_scaled, y_train_all)

    # --- 6. 模型训练 ---
    adaboost = AdaBoostClassifier(
        n_estimators=ADA_BEST_PARAMS['n_estimators'],
        learning_rate=ADA_BEST_PARAMS['learning_rate'],
        random_state=SEED
    )
    adaboost.fit(X_resampled_scaled, y_resampled)

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

    # --- 7. 测试集预测与评估 ---
    y_proba_adaboost = adaboost.predict_proba(X_test_scaled)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_test_scaled)[:, 1]
    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= 0.15).astype(int)  # 使用您原始代码中的阈值

    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    # --- 8. 保存模型 (可选) ---
    # 这里可以添加保存最终模型的代码，类似于原始 main 函数中的 joblib.dump 部分
    # 例如:
    # joblib.dump(imputer, "best_imputer.pkl")
    # joblib.dump(selector, "best_variance_threshold_selector.pkl")
    # ... 为所有组件保存带有 'best_' 前缀的文件
    # print("\n[Model Saving] 最佳模型和预处理器已保存。")

    return auc, recall, precision, final_score


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
    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    if np.isnan(y).any():
        print("检测到目标变量 'y' 中存在 NaN，正在移除对应的样本...")
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        print(f"移除 NaN 后，数据集大小: X={X.shape}, y={y.shape}")

    # --- 数据划分 ---
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    print(f"数据集划分完成: 训练集 {X_train_all.shape}, 测试集 {X_test.shape}")

    # --- 定义参数网格 ---
    param_grid = {
        'inter_intra_threshold': [0.0001, 0.001, 0.0005],  # 示例网格，可根据需要调整
        'adasyn_n_neighbors': [3, 5, 7],  # 示例网格，可根据需要调整
        'brf_top_ratio': [0.7, 0.8, 0.9]  # 选择 60%, 80%, 100% 的特征
    }

    # --- 执行网格搜索 ---
    best_params, best_score = perform_grid_search(X_train_all, y_train_all, X_test, y_test, param_grid, cv_folds=3)


if __name__ == "__main__":
    main()



