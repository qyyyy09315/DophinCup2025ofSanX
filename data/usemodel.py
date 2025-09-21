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


# 以上填写对象的定义
test_data = pd.read_csv(r'testClean.csv')

# 2. 加载完整的级联模型（已包含所有预处理步骤）
cascaded_model = joblib.load('best_cascaded_model.pkl')

# 3. 应用预处理和模型预测
# 删除 'company_id' 列（如果存在）
X_test = test_data.drop(columns=['company_id'], errors='ignore')

# 直接使用级联模型进行预测
y_proba = cascaded_model.predict_proba(X_test)[:, 1]  # 类别1的概率
y_pred = cascaded_model.predict(X_test)  # 类别预测

# 4. 创建结果数据框
results_df = pd.DataFrame({
    'uuid': test_data['company_id'],  # 保留原始company_id
    'proba': y_proba,
    'prediction': y_pred
})

# 5. 保存结果
results_df.to_csv(r'C:\Users\YKSHb\Desktop\submit_template.csv', index=False)

print("预测结果已保存到 submit_template.csv")
