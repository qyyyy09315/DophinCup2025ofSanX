import multiprocessing
import os
import pickle
import random
import time
import warnings

# --- 导入绘图库 ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# --- 导入先进的特征工程库 ---
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, f_classif, \
    mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (recall_score, precision_score, roc_auc_score, f1_score, accuracy_score,
                             precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)

# --- 导入聚类和异常检测用于特征工程 ---
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# --- 导入稀疏矩阵支持 ---
from scipy import sparse
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# 环境配置
NUM_CORES = multiprocessing.cpu_count()
print(f"检测到 {NUM_CORES} 个CPU核心")
os.environ["LOKY_MAX_CPU_COUNT"] = str(NUM_CORES)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 神经网络模型定义 ---
class SimpleNN(nn.Module):
    """一个简单的前馈神经网络"""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出一个logit
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiScaleNNEnsemble:
    """多尺度神经网络集群"""

    def __init__(self, base_model_class, model_params_list, device='cpu'):
        self.base_model_class = base_model_class
        self.model_params_list = model_params_list  # 每个模型的参数字典列表
        self.device = device
        self.models = []
        self.weights = []  # 存储每个模型的权重

    def fit(self, X_train_list, y_train, X_val_list, y_val, epochs=50, batch_size=1024, lr=0.001):
        """训练所有模型"""
        self.models = []
        self.weights = []
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        for i, (X_train, X_val, model_params) in enumerate(zip(X_train_list, X_val_list, self.model_params_list)):
            print(f"训练模型 {i + 1}/{len(self.model_params_list)}...")
            input_dim = X_train.shape[1]
            model = self.base_model_class(input_dim, **model_params).to(self.device)

            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            best_auc = 0
            for epoch in range(epochs):
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # 在验证集上评估以确定权重
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(
                    model(torch.tensor(X_val, dtype=torch.float32).to(self.device))).cpu().numpy().flatten()
            try:
                val_auc = roc_auc_score(y_val, val_preds)
            except:
                val_auc = 0.5  # 防止AUC计算错误
            print(f"  模型 {i + 1} 验证集 AUC: {val_auc:.4f}")

            self.models.append(model)
            self.weights.append(val_auc)  # 使用AUC作为权重

    def predict_proba(self, X_test_list):
        """预测概率，返回正类概率"""
        if not self.models:
            raise ValueError("模型尚未训练，请先调用 fit 方法。")

        all_probs = []
        normalized_weights = np.array(self.weights) / np.sum(self.weights)  # 归一化权重

        for model, X_test, weight in zip(self.models, X_test_list, normalized_weights):
            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                probs = torch.sigmoid(model(test_tensor)).cpu().numpy().flatten()
            all_probs.append(probs * weight)  # 加权概率

        # 对所有加权概率求和得到最终概率
        final_probs = np.sum(np.array(all_probs), axis=0)
        # 确保概率在 [0, 1] 范围内
        final_probs = np.clip(final_probs, 0, 1)
        return final_probs

    def predict(self, X_test_list, threshold=0.5):
        """预测类别"""
        probs = self.predict_proba(X_test_list)
        return (probs >= threshold).astype(int)


def moo_smotetomek_func(X, y):
    """
    使用一组固定的参数应用 SMOTETomek 进行重采样。
    """
    print("  -> 应用固定参数的 SMOTETomek...")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTETomek.")
        return X, y

    try:
        if sparse.issparse(X):
            print("    -> 将稀疏矩阵转换为密集矩阵以进行 SMOTETomek...")
            X_dense = X.toarray()
        else:
            X_dense = X

        smotetomek = SMOTETomek(
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=SEED, n_jobs=-1),
            tomek=TomekLinks(sampling_strategy='majority', n_jobs=-1),
            random_state=SEED
        )

        X_resampled_dense, y_resampled = smotetomek.fit_resample(X_dense, y)

        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled_dense.shape[0]}")
        return X_resampled_dense, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y


class AdvancedFeatureEngineer:
    """
    先进的特征工程管道，包含：
    1. 多种预处理技术
    2. 统计特征生成
    3. 聚类特征
    4. 异常检测特征
    5. 降维技术组合
    6. 智能特征选择
    """

    def __init__(self,
                 # 预处理参数
                 scaler_type='robust',  # 'standard', 'robust', 'quantile', 'power', 'minmax'
                 power_method='yeo-johnson',  # PowerTransformer method

                 # 统计特征参数
                 create_statistical_features=True,
                 rolling_windows=[3, 5, 10],  # 滚动统计窗口

                 # 聚类特征参数
                 create_cluster_features=True,
                 n_clusters_kmeans=10,
                 dbscan_eps=0.5,
                 dbscan_min_samples=5,

                 # 异常检测特征参数
                 create_anomaly_features=True,
                 isolation_contamination=0.1,
                 lof_n_neighbors=20,

                 # 降维参数
                 use_multiple_decomposition=True,
                 svd_components=50,
                 ica_components=30,
                 fa_components=20,

                 # 特征选择参数
                 variance_threshold=0.01,
                 univariate_k_best=1000,
                 rf_n_features_to_select=800,

                 # 输出格式
                 force_sparse_output=False):  # 改为False，因为高级特征工程通常产生密集特征

        # 存储参数
        self.scaler_type = scaler_type
        self.power_method = power_method
        self.create_statistical_features = create_statistical_features
        self.rolling_windows = rolling_windows
        self.create_cluster_features = create_cluster_features
        self.n_clusters_kmeans = n_clusters_kmeans
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.create_anomaly_features = create_anomaly_features
        self.isolation_contamination = isolation_contamination
        self.lof_n_neighbors = lof_n_neighbors
        self.use_multiple_decomposition = use_multiple_decomposition
        self.svd_components = svd_components
        self.ica_components = ica_components
        self.fa_components = fa_components
        self.variance_threshold = variance_threshold
        self.univariate_k_best = univariate_k_best
        self.rf_n_features_to_select = rf_n_features_to_select
        self.force_sparse_output = force_sparse_output

        # 初始化组件
        self.imputer_num = None
        self.scaler = None
        self.power_transformer = None
        self.kmeans = None
        self.dbscan = None
        self.isolation_forest = None
        self.lof = None
        self.svd = None
        self.ica = None
        self.fa = None
        self.variance_selector = None
        self.univariate_selector = None
        self.rf_selector = None

        # 记录特征信息
        self.numerical_features = None
        self.original_feature_count = None
        self.feature_names = []

    def _create_statistical_features(self, X_dense):
        """创建统计特征"""
        if not self.create_statistical_features:
            return X_dense

        print("  -> 创建统计特征...")
        stat_features = []

        # 基础统计特征
        stat_features.append(np.mean(X_dense, axis=1, keepdims=True))  # 行均值
        stat_features.append(np.std(X_dense, axis=1, keepdims=True))  # 行标准差
        stat_features.append(np.max(X_dense, axis=1, keepdims=True))  # 行最大值
        stat_features.append(np.min(X_dense, axis=1, keepdims=True))  # 行最小值
        stat_features.append(np.median(X_dense, axis=1, keepdims=True))  # 行中位数

        # 高级统计特征
        stat_features.append(skew(X_dense, axis=1).reshape(-1, 1))  # 偏度
        stat_features.append(kurtosis(X_dense, axis=1).reshape(-1, 1))  # 峰度

        # 百分位数特征
        stat_features.append(np.percentile(X_dense, 25, axis=1, keepdims=True))  # 25%分位数
        stat_features.append(np.percentile(X_dense, 75, axis=1, keepdims=True))  # 75%分位数

        # 变异系数 (标准差/均值)
        mean_vals = np.mean(X_dense, axis=1, keepdims=True)
        std_vals = np.std(X_dense, axis=1, keepdims=True)
        cv = np.divide(std_vals, mean_vals + 1e-8)  # 避免除零
        stat_features.append(cv)

        # 合并统计特征
        stat_features_array = np.concatenate(stat_features, axis=1)

        # 添加特征名称
        stat_names = ['row_mean', 'row_std', 'row_max', 'row_min', 'row_median',
                      'row_skew', 'row_kurtosis', 'row_q25', 'row_q75', 'row_cv']
        self.feature_names.extend(stat_names)

        print(f"    -> 创建了 {stat_features_array.shape[1]} 个统计特征")
        return np.concatenate([X_dense, stat_features_array], axis=1)

    def _create_cluster_features(self, X_dense):
        """创建聚类特征"""
        if not self.create_cluster_features:
            return X_dense

        print("  -> 创建聚类特征...")
        cluster_features = []

        # K-Means 聚类
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters_kmeans,
                                 random_state=SEED, n_init=10)
            kmeans_labels = self.kmeans.fit_predict(X_dense)
        else:
            kmeans_labels = self.kmeans.predict(X_dense)

        # K-Means 距离特征
        kmeans_distances = self.kmeans.transform(X_dense)
        cluster_features.append(kmeans_distances)

        # DBSCAN 聚类 (仅在训练时)
        if self.dbscan is None:
            self.dbscan = DBSCAN(eps=self.dbscan_eps,
                                 min_samples=self.dbscan_min_samples, n_jobs=-1)
            dbscan_labels = self.dbscan.fit_predict(X_dense)
        else:
            # DBSCAN 没有 predict 方法，需要重新拟合或使用其他方法
            # 这里简化处理，跳过测试集的DBSCAN标签
            dbscan_labels = np.zeros(X_dense.shape[0])

        # 添加聚类标签作为特征 (one-hot编码可能更好，但这里简化)
        cluster_features.append(kmeans_labels.reshape(-1, 1))
        cluster_features.append(dbscan_labels.reshape(-1, 1))

        # 合并聚类特征
        cluster_features_array = np.concatenate(cluster_features, axis=1)

        # 添加特征名称
        cluster_names = [f'kmeans_dist_{i}' for i in range(self.n_clusters_kmeans)]
        cluster_names.extend(['kmeans_label', 'dbscan_label'])
        self.feature_names.extend(cluster_names)

        print(f"    -> 创建了 {cluster_features_array.shape[1]} 个聚类特征")
        return np.concatenate([X_dense, cluster_features_array], axis=1)

    def _create_anomaly_features(self, X_dense):
        """创建异常检测特征"""
        if not self.create_anomaly_features:
            return X_dense

        print("  -> 创建异常检测特征...")
        anomaly_features = []

        # Isolation Forest
        if self.isolation_forest is None:
            self.isolation_forest = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=SEED, n_jobs=-1)
            iso_scores = self.isolation_forest.fit(X_dense).decision_function(X_dense)
        else:
            iso_scores = self.isolation_forest.decision_function(X_dense)

        anomaly_features.append(iso_scores.reshape(-1, 1))

        # Local Outlier Factor
        if self.lof is None:
            self.lof = LocalOutlierFactor(
                n_neighbors=self.lof_n_neighbors,
                novelty=True, n_jobs=-1)
            self.lof.fit(X_dense)
            lof_scores = self.lof.decision_function(X_dense)
        else:
            lof_scores = self.lof.decision_function(X_dense)

        anomaly_features.append(lof_scores.reshape(-1, 1))

        # 合并异常检测特征
        anomaly_features_array = np.concatenate(anomaly_features, axis=1)

        # 添加特征名称
        anomaly_names = ['isolation_score', 'lof_score']
        self.feature_names.extend(anomaly_names)

        print(f"    -> 创建了 {anomaly_features_array.shape[1]} 个异常检测特征")
        return np.concatenate([X_dense, anomaly_features_array], axis=1)

    def fit(self, X, y=None):
        """拟合特征工程管道"""
        print("开始高级特征工程拟合...")

        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X_dense = X.values
        elif sparse.issparse(X):
            X_dense = X.toarray()
            self.numerical_features = list(range(X.shape[1]))
        else:
            X_dense = X
            self.numerical_features = list(range(X.shape[1]))

        self.original_feature_count = X_dense.shape[1]
        print(f"原始特征数: {self.original_feature_count}")

        # 1. 缺失值处理
        print("步骤 1: 缺失值处理...")
        self.imputer_num = SimpleImputer(strategy='median')
        X_dense = self.imputer_num.fit_transform(X_dense)

        # 2. 数据预处理和变换
        print("步骤 2: 数据预处理...")

        # 选择缩放器
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=SEED)
        elif self.scaler_type == 'power':
            self.power_transformer = PowerTransformer(method=self.power_method, standardize=True)
            X_dense = self.power_transformer.fit_transform(X_dense)
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()

        X_dense = self.scaler.fit_transform(X_dense)

        # 3. 高级特征工程
        print("步骤 3: 高级特征工程...")
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)

        print(f"特征工程后特征数: {X_dense.shape[1]}")

        # 4. 多重降维
        if self.use_multiple_decomposition:
            print("步骤 4: 多重降维...")

            # TruncatedSVD
            n_svd = min(self.svd_components, X_dense.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
            X_svd = self.svd.fit_transform(X_dense)

            # FastICA
            n_ica = min(self.ica_components, X_dense.shape[1])
            self.ica = FastICA(n_components=n_ica, random_state=SEED, max_iter=1000)
            X_ica = self.ica.fit_transform(X_dense)

            # Factor Analysis
            n_fa = min(self.fa_components, X_dense.shape[1])
            self.fa = FactorAnalysis(n_components=n_fa, random_state=SEED)
            X_fa = self.fa.fit_transform(X_dense)

            # 组合降维特征
            X_decomposed = np.concatenate([X_svd, X_ica, X_fa], axis=1)
            print(f"  -> SVD: {X_svd.shape[1]}, ICA: {X_ica.shape[1]}, FA: {X_fa.shape[1]}")

            # 将原始特征和降维特征结合
            X_dense = np.concatenate([X_dense, X_decomposed], axis=1)
            print(f"降维后总特征数: {X_dense.shape[1]}")

        # 5. 智能特征选择
        print("步骤 5: 智能特征选择...")

        # 方差过滤
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_dense = self.variance_selector.fit_transform(X_dense)
        print(f"  -> 方差过滤后特征数: {X_dense.shape[1]}")

        # 单变量特征选择
        if y is not None and self.univariate_k_best:
            k_best = min(self.univariate_k_best, X_dense.shape[1])
            self.univariate_selector = SelectKBest(score_func=f_classif, k=k_best)
            X_dense = self.univariate_selector.fit_transform(X_dense, y)
            print(f"  -> 单变量选择后特征数: {X_dense.shape[1]}")

        # 随机森林特征选择
        if y is not None and self.rf_n_features_to_select:
            n_rf_features = min(self.rf_n_features_to_select, X_dense.shape[1])
            temp_rf = BalancedRandomForestClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, random_state=SEED)
            self.rf_selector = SelectFromModel(
                temp_rf, max_features=n_rf_features, threshold=-np.inf)
            X_dense = self.rf_selector.fit_transform(X_dense, y)
            print(f"  -> 随机森林选择后特征数: {X_dense.shape[1]}")

        print("特征工程拟合完成！")
        return self

    def transform(self, X):
        """应用特征工程变换"""
        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            X_dense = X.values
        elif sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 1. 缺失值处理
        X_dense = self.imputer_num.transform(X_dense)

        # 2. 数据预处理
        if self.power_transformer is not None:
            X_dense = self.power_transformer.transform(X_dense)
        X_dense = self.scaler.transform(X_dense)

        # 3. 特征工程 (注意：聚类和异常检测需要特殊处理)
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)

        # 4. 多重降维
        if self.use_multiple_decomposition:
            X_svd = self.svd.transform(X_dense)
            X_ica = self.ica.transform(X_dense)
            X_fa = self.fa.transform(X_dense)
            X_decomposed = np.concatenate([X_svd, X_ica, X_fa], axis=1)
            X_dense = np.concatenate([X_dense, X_decomposed], axis=1)

        # 5. 特征选择
        X_dense = self.variance_selector.transform(X_dense)
        if self.univariate_selector is not None:
            X_dense = self.univariate_selector.transform(X_dense)
        if self.rf_selector is not None:
            X_dense = self.rf_selector.transform(X_dense)

        # 输出格式控制
        if self.force_sparse_output:
            return sparse.csr_matrix(X_dense)
        else:
            return X_dense

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)


# 其他函数保持不变
def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 {filename}")


def find_best_f1_threshold(y_true, y_proba):
    """在验证集上寻找最佳阈值以优化 F1 分数"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_f_score = f_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print(f"最佳 F1 阈值: {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f_score:.4f}")

    return best_threshold


def find_threshold_for_max_recall(y_true, y_proba, min_precision=0.4):
    """寻找能获得最高召回率且精确率不低于 min_precision 的阈值"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid_indices) == 0:
        print(f"警告: 没有阈值能满足精确率 >= {min_precision}。返回默认阈值 0.5。")
        return 0.5

    best_valid_idx = valid_indices[np.argmax(recalls[valid_indices])]
    best_threshold = thresholds[best_valid_idx]
    best_precision = precisions[best_valid_idx]
    best_recall = recalls[best_valid_idx]

    print(f"高召回率阈值 (Precision >= {min_precision}): {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    return best_threshold


def evaluate_and_plot(y_true, y_proba, threshold_f1, threshold_hr, model_name="Model"):
    """评估模型并绘制 ROC, PR, Confusion Matrix"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='b', lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")

    # Confusion Matrix
    y_pred_f1 = (y_proba >= threshold_f1).astype(int)
    cm = confusion_matrix(y_true, y_pred_f1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title(f'Confusion Matrix (Threshold={threshold_f1:.4f})')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    print(f"\n=== {model_name} 详细评估报告 ===")
    print(classification_report(y_true, y_pred_f1, target_names=['Negative', 'Positive']))
    print("-" * 40)


def train_and_evaluate_advanced(X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer_params=None):
    """使用先进特征工程的训练评估流程，模型替换为多尺度神经网络集群"""
    print("\n=== 构建先进特征工程管道 (多尺度神经网络集群) ===")

    # 特征工程 - 创建多个不同配置的特征工程器，形成多尺度输入
    print("\n=== 创建多尺度特征工程器 ===")
    fe_params_list = [
        # 基础配置
        feature_engineer_params,
        # 配置2：更强的降维
        {**feature_engineer_params, 'svd_components': 50, 'ica_components': 20, 'fa_components': 10,
         'univariate_k_best': 800, 'rf_n_features_to_select': 500},
        # 配置3：更少的降维，保留更多原始特征
        {**feature_engineer_params, 'svd_components': 100, 'ica_components': 70, 'fa_components': 50,
         'univariate_k_best': 1800, 'rf_n_features_to_select': 1200},
    ]

    feature_engineers = []
    X_train_engineered_list = []
    X_val_engineered_list = []
    X_test_engineered_list = []

    for i, params in enumerate(fe_params_list):
        print(f"\n--- 拟合特征工程器 {i + 1}/{len(fe_params_list)} ---")
        fe = AdvancedFeatureEngineer(**params)
        fe.fit(X_train, y_train)
        feature_engineers.append(fe)

        X_train_engineered_list.append(fe.transform(X_train))
        X_val_engineered_list.append(fe.transform(X_val))
        X_test_engineered_list.append(fe.transform(X_test))
        print(f"特征工程器 {i + 1} 输出特征数: {X_train_engineered_list[-1].shape[1]}")

    # 2. 应用 SMOTETomek (对第一个特征工程器的结果应用，或对每个都应用，这里简化)
    # 为了简化，我们只对第一个主要特征工程的结果应用重采样
    print("\n=== 应用 SMOTETomek 重采样 (基于第一个特征工程器) ===")
    X_train_resampled, y_train_resampled = moo_smotetomek_func(X_train_engineered_list[0], y_train)

    # 更新列表中的第一个元素
    X_train_engineered_list[0] = X_train_resampled

    # 3. 定义神经网络模型集群
    print("\n=== 定义神经网络模型集群 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ensemble_model_params_list = [
        {'hidden_dims': [256, 128, 64], 'dropout_rate': 0.3},
        {'hidden_dims': [128, 64], 'dropout_rate': 0.2},
        {'hidden_dims': [512, 256, 128, 64], 'dropout_rate': 0.4},
        {'hidden_dims': [64, 32], 'dropout_rate': 0.1},
    ]

    classifier = MultiScaleNNEnsemble(SimpleNN, ensemble_model_params_list, device=device)

    # 4. 模型训练
    print("\n=== 开始模型集群训练 ===")
    start_time = time.time()
    classifier.fit(X_train_engineered_list, y_train_resampled, X_val_engineered_list, y_val, epochs=50, batch_size=1024,
                   lr=0.001)
    end_time = time.time()
    print(f"模型集群训练耗时: {end_time - start_time:.2f} 秒")

    # 5. 保存模型 (保存整个特征工程器列表和分类器)
    trained_pipeline = {
        'feature_engineers': feature_engineers,
        'classifier': classifier
    }
    save_model(trained_pipeline, 'advanced_feature_engineering_nn_ensemble_model.pkl')

    # 6. 验证集阈值选择
    print("\n=== 验证集阈值选择 ===")
    val_proba = classifier.predict_proba(X_val_engineered_list)

    threshold_f1 = find_best_f1_threshold(y_val, val_proba)
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # 7. 测试集评估
    print("\n=== 测试集评估 ===")
    test_proba = classifier.predict_proba(X_test_engineered_list)

    # F1优化阈值评估
    print("\n--- 使用 F1 优化阈值评估 ---")
    y_pred_f1 = (test_proba >= threshold_f1).astype(int)
    recall_f1 = recall_score(y_test, y_pred_f1)
    auc_f1 = roc_auc_score(y_test, test_proba)
    precision_f1 = precision_score(y_test, y_pred_f1)
    f1_f1 = f1_score(y_test, y_pred_f1)
    accuracy_f1 = accuracy_score(y_test, y_pred_f1)
    print(f"阈值: {threshold_f1:.4f}")
    print(f"Recall: {recall_f1:.4f}")
    print(f"AUC: {auc_f1:.4f}")
    print(f"Precision: {precision_f1:.4f}")
    print(f"F1 Score: {f1_f1:.4f}")
    print(f"Accuracy: {accuracy_f1:.4f}")
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision_f1 + 50 * auc_f1 + 30 * recall_f1:.4f}")

    # 高召回率阈值评估
    print("\n--- 使用高召回率阈值评估 ---")
    y_pred_hr = (test_proba >= threshold_high_recall).astype(int)
    recall_hr = recall_score(y_test, y_pred_hr)
    auc_hr = roc_auc_score(y_test, test_proba)
    precision_hr = precision_score(y_test, y_pred_hr)
    f1_hr = f1_score(y_test, y_pred_hr)
    accuracy_hr = accuracy_score(y_test, y_pred_hr)
    print(f"阈值: {threshold_high_recall:.4f}")
    print(f"Recall: {recall_hr:.4f}")
    print(f"AUC: {auc_hr:.4f}")
    print(f"Precision: {precision_hr:.4f}")
    print(f"F1 Score: {f1_hr:.4f}")
    print(f"Accuracy: {accuracy_hr:.4f}")
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision_hr + 50 * auc_hr + 30 * recall_hr:.4f}")

    # 可视化评估结果
    print("\n=== 绘制评估图表 ===")
    evaluate_and_plot(y_test, test_proba, threshold_f1, threshold_high_recall,
                      model_name="Advanced Feature Engineering NN Ensemble")

    return {
        'recall': recall_f1,
        'auc': auc_f1,
        'precision': precision_f1,
        'f1': f1_f1,
        'accuracy': accuracy_f1,
        'y_proba': test_proba,
        'y_pred': y_pred_f1,
        'threshold': threshold_f1,
        'high_recall_results': {
            'recall': recall_hr,
            'precision': precision_hr,
            'f1': f1_hr,
            'threshold': threshold_high_recall
        },
        'trained_model': trained_pipeline  # 保存的是包含特征工程器和分类器的字典
    }


def main():
    # 检查CUDA可用性
    print("CUDA 可用性 (PyTorch):", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # 配置先进特征工程参数 (基础配置)
    feature_engineer_params = {
        # 预处理配置
        'scaler_type': 'robust',  # 使用鲁棒缩放，对异常值更稳健
        'power_method': 'yeo-johnson',  # Yeo-Johnson变换处理偏态分布

        # 统计特征配置
        'create_statistical_features': True,
        'rolling_windows': [3, 5, 10],

        # 聚类特征配置
        'create_cluster_features': True,
        'n_clusters_kmeans': 15,  # 增加聚类数量
        'dbscan_eps': 0.3,
        'dbscan_min_samples': 5,

        # 异常检测特征配置
        'create_anomaly_features': True,
        'isolation_contamination': 0.05,  # 假设5%的数据是异常值
        'lof_n_neighbors': 20,

        # 多重降维配置
        'use_multiple_decomposition': True,
        'svd_components': 80,  # 增加SVD成分
        'ica_components': 50,  # 增加ICA成分
        'fa_components': 30,  # 增加FA成分

        # 特征选择配置
        'variance_threshold': 0.01,
        'univariate_k_best': 1500,  # 增加单变量选择的特征数
        'rf_n_features_to_select': 1000,  # 最终保留1000个特征

        # 输出格式
        'force_sparse_output': False  # 使用密集矩阵，便于高级特征工程
    }

    # 数据加载
    print("\n=== 数据加载 ===")
    try:
        data = pd.read_csv("clean.csv", low_memory=False)
        if "company_id" in data.columns:
            data = data.drop(columns=["company_id"])
        if "target" not in data.columns:
            raise ValueError("数据中必须包含'target'列")

        # 分离特征和目标
        y = data["target"].values.astype(int)
        feature_data = data.drop(columns=["target"])
        X = feature_data.values

        print(f"数据加载成功: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
        print(f"正样本比例: {np.mean(y):.4f}")
        print(f"使用的特征工程参数基础配置: {feature_engineer_params}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 移除y中的NaN
    if np.isnan(y).any():
        print("移除y中的NaN...")
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    # 数据划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y)
    print(f"\n初步数据划分: 临时集={X_temp.shape[0]}, 测试集={X_test.shape[0]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=SEED, stratify=y_temp)
    print(f"最终数据划分: 训练集={X_train.shape[0]}, 验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")

    # 启动先进特征工程训练流程 (使用修改后的函数)
    print("\n=== 启动先进特征工程训练流程 (多尺度神经网络集群) ===")
    results = train_and_evaluate_advanced(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_engineer_params=feature_engineer_params
    )

    # 保存预测结果
    if 'y_proba' in results and 'y_pred' in results:
        pd.DataFrame({
            'true_label': y_test,
            'pred_prob': results['y_proba'],
            'pred_label_f1_optimized': results['y_pred']
        }).to_csv("advanced_predictions_nn_ensemble.csv", index=False)
        print("\n预测结果已保存到 advanced_predictions_nn_ensemble.csv")

    # 打印最终总结
    print("\n" + "=" * 50)
    print("先进特征工程模型训练完成 (多尺度神经网络集群)！")
    print("=" * 50)
    print(f"最佳F1阈值: {results['threshold']:.4f}")
    print(f"测试集性能:")
    print(f"  - Precision: {results['precision']:.4f}")
    print(f"  - Recall: {results['recall']:.4f}")
    print(f"  - F1-Score: {results['f1']:.4f}")
    print(f"  - AUC-ROC: {results['auc']:.4f}")
    print(f"  - Accuracy: {results['accuracy']:.4f}")

    test_score = 20 * results['precision'] + 50 * results['auc'] + 30 * results['recall']
    print(f"  - 综合得分: {test_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()




