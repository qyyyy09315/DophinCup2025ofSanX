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
from sklearn.metrics import recall_score, precision_score, roc_auc_score, make_scorer
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.ensemble import BalancedRandomForestClassifier


os.environ["LOKY_MAX_CPU_COUNT"] = "16"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def custom_score_func(y_true, y_proba):
    """计算自定义的加权分数: 0.3 * Recall + 0.5 * AUC + 0.2 * Precision"""
    y_pred = (y_proba >= 0.1).astype(int)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    final_score = 0.3 * recall + 0.5 * auc + 0.2 * precision
    return final_score

# 修复 v2: 修正函数签名以匹配 make_scorer(needs_proba=True) 的调用方式
# make_scorer 当 needs_proba=True 时，会调用 scorer_func(y_true, y_proba, ...)
def scorer_func(y_true, y_proba, **kwargs):
    """适配 make_scorer(needs_proba=True) 的函数签名"""
    # y_true 是真实的标签
    # y_proba 是模型预测的概率 (对于二分类，通常是正类的概率)
    # **kwargs 用于接收并忽略 scikit-learn 可能传递的其他参数
    return custom_score_func(y_true, y_proba)

# 使用 needs_proba=True 创建评分器
custom_scorer = make_scorer(scorer_func, needs_proba=True)


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


class ImprovedKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        centers = []
        centers.append(X[np.random.randint(n_samples)])

        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers.append(X[j])
                    break
        centers = np.array(centers)

        for iteration in range(self.max_iter):
            distances = cdist(X, centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            center_shift = np.linalg.norm(new_centers - centers)

            centers = new_centers

            if center_shift < self.tol:
                print(f"K-means 收敛于第 {iteration + 1} 次迭代")
                break
        else:
            print(f"K-means 达到最大迭代次数 {self.max_iter}")

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans has not been fitted yet.")
        X = np.asarray(X)
        distances = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


class BrfSelector:
    """用于包装 Balanced Random Forest 选择的特征掩码的标准类"""
    def __init__(self, selected_features_mask):
        self.selected_features_ = selected_features_mask

    def transform(self, X):
        return X[:, self.selected_features_]


class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, inter_intra_selector, brf_selector, kmeans, scaler, classifiers):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.inter_intra_selector = inter_intra_selector
        self.brf_selector = brf_selector
        self.kmeans = kmeans
        self.scaler = scaler
        self.classifiers = classifiers

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_inter_intra = self.inter_intra_selector.transform(X_sel)
        X_brf_selected = self.brf_selector.transform(X_inter_intra)
        X_scaled = self.scaler.transform(X_brf_selected)
        return X_scaled

    def predict(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        preds = [clf.predict(X_scaled) for clf in self.classifiers]
        final_pred = np.round(np.mean(preds, axis=0)).astype(int)
        return final_pred.ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        probas = [clf.predict_proba(X_scaled)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])


def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"显存已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    else:
        print("CUDA 不可用，将使用 CPU 训练。")

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

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)

    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)

    inter_intra_selector = InterIntraDistanceSelector(threshold=0.001)
    X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
    print(f"[Feature Selection] 原始特征数: {X_train_selected.shape[1]}, 选择后特征数 (类间类内): {X_train_inter_intra_selected.shape[1]}")

    print("[Feature Selection] 训练 Balanced Random Forest 以获取特征重要性...")
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    brf.fit(X_train_inter_intra_selected, y_train_all)

    feature_importances = brf.feature_importances_
    num_features_to_select = max(1, int(0.8 * X_train_inter_intra_selected.shape[1]))
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_indices = sorted_indices[:num_features_to_select]
    brf_selected_features_mask = np.zeros(X_train_inter_intra_selected.shape[1], dtype=bool)
    brf_selected_features_mask[top_indices] = True

    X_train_brf_selected = X_train_inter_intra_selected[:, brf_selected_features_mask]
    print(f"[Feature Selection] 类间类内选择后特征数: {X_train_inter_intra_selected.shape[1]}, BRF 选择后特征数: {X_train_brf_selected.shape[1]}")

    brf_selector = BrfSelector(brf_selected_features_mask)

    kmeans = ImprovedKMeans(n_clusters=2, random_state=SEED)
    cluster_labels = kmeans.fit_predict(X_train_brf_selected)
    print(f"[KMeans] 聚类中心:\n{kmeans.cluster_centers_}")

    pre_scaler = StandardScaler()
    X_train_brf_selected_scaled = pre_scaler.fit_transform(X_train_brf_selected)
    scaler = pre_scaler

    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    X_resampled_scaled, y_resampled = adasyn.fit_resample(X_train_brf_selected_scaled, y_train_all)

    print("[Hyperparameter Tuning] 开始随机搜索超参数...")

    print("[Hyperparameter Tuning] 调优 AdaBoost...")
    ada_param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.5),
    }
    ada_base = AdaBoostClassifier(random_state=SEED)
    ada_search = RandomizedSearchCV(
        estimator=ada_base,
        param_distributions=ada_param_dist,
        n_iter=20,
        scoring=custom_scorer, # 使用修复后的自定义评分器
        cv=3,
        n_jobs=-1,
        random_state=SEED,
        verbose=1
    )
    ada_search.fit(X_resampled_scaled, y_resampled)
    best_adaboost = ada_search.best_estimator_
    print(f"[Hyperparameter Tuning] AdaBoost 最佳参数: {ada_search.best_params_}")
    print(f"[Hyperparameter Tuning] AdaBoost 最佳分数: {ada_search.best_score_:.4f}")

    print("[Hyperparameter Tuning] 调优 XGBoost...")
    xgb_param_dist = {
        'n_estimators': randint(200, 600),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }
    xgb_base = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric='logloss')
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=30,
        scoring=custom_scorer, # 使用修复后的自定义评分器
        cv=3,
        n_jobs=-1,
        random_state=SEED,
        verbose=1
    )
    xgb_search.fit(X_resampled_scaled, y_resampled)
    best_xgboost = xgb_search.best_estimator_
    print(f"[Hyperparameter Tuning] XGBoost 最佳参数: {xgb_search.best_params_}")
    print(f"[Hyperparameter Tuning] XGBoost 最佳分数: {xgb_search.best_score_:.4f}")

    adaboost = best_adaboost
    xgboost = best_xgboost

    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_inter_intra_selected = inter_intra_selector.transform(X_test_selected)
    X_test_brf_selected = brf_selector.transform(X_test_inter_intra_selected)
    X_test_scaled = scaler.transform(X_test_brf_selected)

    y_proba_adaboost = adaboost.predict_proba(X_test_scaled)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_test_scaled)[:, 1]

    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= 0.2).astype(int)

    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        inter_intra_selector=inter_intra_selector,
        brf_selector=brf_selector,
        kmeans=kmeans,
        scaler=scaler,
        classifiers=[adaboost, xgboost]
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(inter_intra_selector, "inter_intra_distance_selector.pkl")
    joblib.dump(brf_selector, "brf_selector.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")

    print("模型和预处理器已保存。")
    print("[Final Model] 模型训练完成 (未使用深度网络和 GPU)，已进行超参数调优")



if __name__ == "__main__":
    main()
