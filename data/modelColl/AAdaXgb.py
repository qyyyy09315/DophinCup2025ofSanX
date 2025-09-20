import os
import random
import joblib
import numpy as np
import pandas as pd
import torch  # 保留，以防其他部分用到，但本次不使用
import torch.nn as nn  # 保留，以防其他部分用到，但本次不使用
import torch.nn.functional as F  # 保留，以防其他部分用到，但本次不使用
# from sklearn.model_selection import train_test_split # 已注释，因为未使用
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
# from torch.utils.data import DataLoader, TensorDataset # 已注释，因为不使用深度学习
# from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # 已注释
# import torch.cuda.amp as amp  # 混合精度 # 已注释
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split  # 重新引入，因为实际使用了

# 设置随机种子
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 已注释，因为不再使用GPU
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # 根据 CPU 核心数调整
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# torch.manual_seed(SEED) # 已注释
# if torch.cuda.is_available(): # 已注释
#     torch.cuda.manual_seed_all(SEED) # 已注释
#     torch.backends.cudnn.deterministic = False  # 已注释
#     torch.backends.cudnn.benchmark = True  # 已注释
# torch.set_num_threads(16) # 已注释


# ------------------- 类间类内距离特征选择器 -------------------
class InterIntraDistanceSelector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y):
        """
        基于类间类内距离进行特征选择。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("目前只支持二分类任务")

        class_0_indices = (y == classes[0])
        class_1_indices = (y == classes[1])
        X_0 = X[class_0_indices]
        X_1 = X[class_1_indices]

        # 计算每个特征的类内方差之和
        intra_class_var_0 = np.var(X_0, axis=0)
        intra_class_var_1 = np.var(X_1, axis=0)
        intra_class_variance_sum = intra_class_var_0 + intra_class_var_1

        # 计算每个特征的类间距离平方
        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        inter_class_distance_sq = (mean_1 - mean_0) ** 2

        # 计算判别分数 (类间距离 / 类内方差和)
        # 避免除以零
        scores = np.divide(inter_class_distance_sq, intra_class_variance_sum,
                           out=np.zeros_like(inter_class_distance_sq),
                           where=intra_class_variance_sum != 0)

        # 选择分数高于阈值的特征
        self.selected_features_ = scores > self.threshold
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# ------------------- 改进的K-means聚类器 -------------------
class ImprovedKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """
        使用改进的策略拟合 K-means。
        """
        X = np.asarray(X)
        np.random.seed(self.random_state)

        # 1. 初始化聚类中心 (使用标准 K-means++ 初始化)
        n_samples, n_features = X.shape
        centers = []
        # 随机选择第一个中心
        centers.append(X[np.random.randint(n_samples)])

        for _ in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])
            # 选择下一个中心的概率与距离平方成正比
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers.append(X[j])
                    break
        centers = np.array(centers)

        # 迭代优化
        for iteration in range(self.max_iter):
            # 分配每个点到最近的中心
            distances = cdist(X, centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # 计算中心变化
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
        """
        预测样本所属的簇。
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans has not been fitted yet.")
        X = np.asarray(X)
        distances = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


# ------------------- 级联模型类 (修改后) -------------------
# 移除了与 FeatureExtractor 和 emb_imputer 相关的属性和逻辑
class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, inter_intra_selector, kmeans, scaler, classifiers):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.inter_intra_selector = inter_intra_selector
        self.kmeans = kmeans
        self.scaler = scaler
        self.classifiers = classifiers

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_inter_intra = self.inter_intra_selector.transform(X_sel)
        # 注意：K-means聚类主要用于识别离群点，这里我们不直接用它变换数据，
        # 而是假设它在训练时帮助识别了离群点并进行了处理（例如过滤或加权）。
        # 如果需要在预测时也应用聚类，逻辑会更复杂。
        X_scaled = self.scaler.transform(X_inter_intra)
        return X_scaled

    def predict(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        # 不再通过 FeatureExtractor，直接使用 X_scaled
        preds = [clf.predict(X_scaled) for clf in self.classifiers]
        final_pred = np.round(np.mean(preds, axis=0)).astype(int)
        return final_pred.ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        # 不再通过 FeatureExtractor，直接使用 X_scaled
        probas = [clf.predict_proba(X_scaled)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])


# ------------------- 主流程 -------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())  # 保留打印，但实际不使用
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

    # 处理 y 中可能存在的 NaN (根据之前的对话)
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

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 新增：基于类间类内距离的特征选择
    inter_intra_selector = InterIntraDistanceSelector(threshold=0.001)  # 可调整阈值
    X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
    print(
        f"[Feature Selection] 原始特征数: {X_train_selected.shape[1]}, 选择后特征数: {X_train_inter_intra_selected.shape[1]}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 新增：使用改进的 K-means 识别离群点 (在训练集上)
    # 这里我们简单地拟合K-means并打印中心，实际应用中可以用来过滤或加权样本
    kmeans = ImprovedKMeans(n_clusters=2, random_state=SEED)
    # 假设我们先用原始数据拟合K-means来识别离群点
    # 或者用选择后的特征拟合
    cluster_labels = kmeans.fit_predict(X_train_inter_intra_selected)
    print(f"[KMeans] 聚类中心:\n{kmeans.cluster_centers_}")
    # 这里可以添加逻辑来处理离群点，例如：
    # 1. 找到每个簇的马氏距离
    # 2. 标记距离簇中心过远的点为离群点
    # 3. 在后续训练中过滤或降低这些点的权重
    # 为简化，我们这里只做拟合，不改变训练数据
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 数据标准化 (在 ADASYN 之前)
    pre_scaler = StandardScaler()
    X_train_inter_intra_selected_scaled = pre_scaler.fit_transform(X_train_inter_intra_selected)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    # X_resampled, y_resampled = adasyn.fit_resample(X_train_inter_intra_selected, y_train_all)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 修改：对标准化后的数据进行 ADASYN
    X_resampled_scaled, y_resampled = adasyn.fit_resample(X_train_inter_intra_selected_scaled, y_train_all)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 移除：不再需要 StandardScaler，因为 ADASYN 后的数据已经是标准化的
    # scaler = StandardScaler()
    # X_resampled_scaled = scaler.fit_transform(X_resampled)
    # 使用 pre_scaler 的副本作为最终的 scaler 保存
    scaler = pre_scaler  # 重要：确保测试集使用相同的 scaler
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 移除：深度网络特征提取部分
    # input_dim = X_resampled_scaled.shape[1]
    # feature_extractor = FeatureExtractor(...)
    # feature_extractor.fit(...)
    # X_emb_train = feature_extractor.transform(...)
    # emb_imputer = SimpleImputer(...)
    # X_emb_train_clean = emb_imputer.fit_transform(X_emb_train)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 修改：直接使用 ADASYN 后的数据训练分类器
    # adaboost = AdaBoostClassifier(...)
    # adaboost.fit(X_emb_train_clean, y_resampled)
    adaboost = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=SEED
    )
    adaboost.fit(X_resampled_scaled, y_resampled)  # <<< 使用 ADASYN 后的数据训练
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # xgboost = XGBClassifier(...)
    # xgboost.fit(X_emb_train_clean, y_resampled)
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
    xgboost.fit(X_resampled_scaled, y_resampled)  # <<< 使用 ADASYN 后的数据训练
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 测试阶段预处理
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_inter_intra_selected = inter_intra_selector.transform(X_test_selected)  # <<< 新增：测试集特征选择
    # 注意：测试集不应用 K-means 聚类处理
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # X_test_scaled = scaler.transform(X_test_inter_intra_selected)
    # 修改：使用与训练集相同的 scaler (pre_scaler) 进行标准化
    X_test_scaled = scaler.transform(X_test_inter_intra_selected)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 移除：深度网络特征提取部分
    # X_emb_test = feature_extractor.transform(X_test_scaled)
    # X_emb_test_clean = emb_imputer.transform(X_emb_test)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 修改：直接使用标准化后的测试数据进行预测
    # y_proba_adaboost = adaboost.predict_proba(X_emb_test_clean)[:, 1]
    # y_proba_xgboost = xgboost.predict_proba(X_emb_test_clean)[:, 1]
    y_proba_adaboost = adaboost.predict_proba(X_test_scaled)[:, 1]  # <<< 直接使用测试集
    y_proba_xgboost = xgboost.predict_proba(X_test_scaled)[:, 1]  # <<< 直接使用测试集
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= 0.5).astype(int)

    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # cascaded = CascadedModel(...) # 修改：移除 feature_extractor 和 emb_imputer
    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        inter_intra_selector=inter_intra_selector,  # <<< 保存选择器
        kmeans=kmeans,  # <<< 保存聚类器
        scaler=scaler,  # <<< 保存 scaler (即 pre_scaler)
        classifiers=[adaboost, xgboost]
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(inter_intra_selector, "inter_intra_distance_selector.pkl")  # <<< 保存选择器
    joblib.dump(kmeans, "kmeans_model.pkl")  # <<< 保存聚类器
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")  # <<< 保存 scaler (即 pre_scaler)
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # joblib.dump(emb_imputer, "emb_imputer.pkl") # 移除
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print("模型和预处理器已保存。")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # if torch.cuda.is_available(): ... # 移除或注释掉与 GPU 相关的最终打印
    print("[Final Model] 模型训练完成 (未使用深度网络和 GPU)")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    main()