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

# --- 导入 XGBoost ---
from xgboost import XGBClassifier

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
        # 新增：用于记录嵌入特征组
        self.embedding_groups = {}
        # 新增：用于记录向量特征列名
        self.vector_feature_names = []
        # 新增：记录经过所有工程处理后的特征数
        self.final_feature_count_after_engineering_ = None
        # 新增：记录在fit阶段的特征数，用于后续一致性检查
        self.fit_feature_count_before_imputer_ = None
        self.fit_feature_count_after_imputer_ = None

        # Flag to indicate if the estimator is fitted
        self._fitted = False

    def _identify_embedding_groups(self, feature_names):
        """识别并分组 *_emb_* 形式的特征"""
        print("  -> 识别嵌入特征组...")
        emb_pattern = "_emb_"  # 修改为新的模式
        emb_dict = {}

        for feat in feature_names:
            if emb_pattern in feat:
                # 提取前缀 (例如 'industry_emb_0' -> 'industry')
                parts = feat.split(emb_pattern)
                if len(parts) >= 2:
                    prefix = emb_pattern.join(parts[:-1])  # 处理 'a_emb_b_emb_0' 的情况
                    suffix = parts[-1]
                    # 确保后缀是数字
                    if suffix.isdigit():
                        if prefix not in emb_dict:
                            emb_dict[prefix] = []
                        emb_dict[prefix].append(feat)

        # 按照编号排序每个组内的特征
        for prefix, feats in emb_dict.items():
            # 根据后缀数字排序
            feats.sort(key=lambda x: int(x.split('_')[-1]))
            self.embedding_groups[prefix] = feats
            print(f"    -> 发现嵌入组 '{prefix}': 包含 {len(feats)} 个特征")

    def _process_embedding_features(self, X_df):
        """处理嵌入特征：将 *_emb_* 组合并为向量特征，并保留原始向量和范数"""
        if not self.embedding_groups:
            print("  -> 未发现嵌入特征组，跳过处理。")
            return X_df

        print("  -> 处理嵌入特征组...")
        X_processed_df = X_df.copy()

        for prefix, emb_features in self.embedding_groups.items():
            print(f"    -> 处理组 '{prefix}' ({len(emb_features)} 个特征)...")
            # 提取嵌入向量
            emb_vectors = X_processed_df[emb_features].values

            # 1. 为新特征命名 (例如 'industry_emb_vector')
            new_vector_feature_name = f"{prefix}_emb_vector"
            new_norm_feature_name = f"{prefix}_emb_norm"

            # 2. 将向量作为新列添加 (注意：这会创建一个包含数组的列)
            #    这里我们将向量本身作为一个特征存储
            X_processed_df[new_vector_feature_name] = list(emb_vectors)

            # 3. 计算向量的L2范数作为另一个特征
            vector_norms = np.linalg.norm(emb_vectors, axis=1)
            X_processed_df[new_norm_feature_name] = vector_norms

            # 4. 删除原始的嵌入特征列
            X_processed_df = X_processed_df.drop(columns=emb_features)
            print(f"    -> 合并为特征 '{new_vector_feature_name}' 和 '{new_norm_feature_name}' 并移除原始列")
            # 记录新创建的向量特征列名
            self.vector_feature_names.append(new_vector_feature_name)

        print(f"  -> 嵌入特征处理完成，剩余特征数: {X_processed_df.shape[1]}")
        return X_processed_df

    def _impute_vector_columns(self, X_df):
        """使用零向量填充向量列中的缺失值"""
        if not self.vector_feature_names:
            print("  -> 未发现向量特征列，跳过向量缺失值填充。")
            return X_df

        print("  -> 使用零向量填充向量列缺失值...")
        X_imputed_df = X_df.copy()

        for vec_col_name in self.vector_feature_names:
            # 获取该列的第一个非空元素来确定向量维度
            first_valid_entry = X_imputed_df[vec_col_name].dropna().iloc[0] if not X_imputed_df[
                vec_col_name].dropna().empty else None
            if first_valid_entry is not None:
                # 确保第一个有效条目是数组或列表，然后获取其长度
                if isinstance(first_valid_entry, (list, np.ndarray)):
                    vector_dim = len(first_valid_entry)
                else:
                    # 如果第一个有效条目不是列表或数组（理论上不应该发生），跳过
                    print(
                        f"    -> 警告: 列 '{vec_col_name}' 的第一个有效条目不是列表或数组: {type(first_valid_entry)}，跳过填充。")
                    continue
                zero_vector = np.zeros(vector_dim)

                # 定义一个更安全的填充函数
                def fill_na_with_zero_vector(x):
                    # 检查是否为标量缺失值
                    try:
                        # 对于标量值，pd.isna 是安全的
                        if not isinstance(x, (list, np.ndarray)):
                            if pd.isna(x):
                                return zero_vector
                            else:
                                return x

                        # 对于数组或列表
                        x_array = np.asarray(x)

                        # 检查是否为空数组
                        if x_array.size == 0:
                            return zero_vector

                        # 检查是否包含所有NaN
                        if x_array.dtype.kind in ['f', 'c']:  # 浮点数或复数
                            if np.all(np.isnan(x_array)):
                                return zero_vector

                        # 如果是有效向量，保留原值
                        return x

                    except Exception as e:
                        print(f"      -> 处理值时出错: {e}, 使用零向量填充")
                        return zero_vector

                # 应用填充函数
                X_imputed_df[vec_col_name] = X_imputed_df[vec_col_name].apply(fill_na_with_zero_vector)
                print(f"    -> 列 '{vec_col_name}' 已用维度为 {vector_dim} 的零向量填充缺失值/空向量/全NaN向量。")
            else:
                print(f"    -> 警告: 列 '{vec_col_name}' 没有有效数据来确定向量维度，跳过填充。")
        return X_imputed_df

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

    def _create_cluster_features(self, X_dense, training_phase=False):
        """创建聚类特征"""
        if not self.create_cluster_features:
            return X_dense

        print("  -> 创建聚类特征...")
        cluster_features = []

        # K-Means 聚类
        if training_phase and self.kmeans is None:
            # Fit on training data
            self.kmeans = KMeans(n_clusters=self.n_clusters_kmeans,
                                 random_state=SEED, n_init=10)
            kmeans_labels = self.kmeans.fit_predict(X_dense)
        elif not training_phase and self.kmeans is not None:
            # Predict on non-training data using already fitted model
            kmeans_labels = self.kmeans.predict(X_dense)
        else:
            # This can happen if called during inference without fitting or other edge cases.
            # For simplicity, we'll just add dummy labels here.
            print("Warning: KMeans was not properly fitted. Adding dummy cluster features.")
            kmeans_labels = np.zeros(X_dense.shape[0])

        # K-Means 距离特征 - Always transform based on fitted centroids
        if self.kmeans is not None:
            kmeans_distances = self.kmeans.transform(X_dense)
            cluster_features.append(kmeans_distances)

            # Add label only once per call
            cluster_features.append(kmeans_labels.reshape(-1, 1))

        # DBSCAN 聚类
        if training_phase and self.dbscan is None:
            # Only fit/predict on training set due to lack of .predict()
            self.dbscan = DBSCAN(eps=self.dbscan_eps,
                                 min_samples=self.dbscan_min_samples, n_jobs=-1)
            dbscan_labels = self.dbscan.fit_predict(X_dense)
        elif not training_phase and self.dbscan is not None:
            # Inference phase uses previously computed clusters (dummy prediction approach).
            # As DBSCAN has no direct predict(), re-fitting might be needed but that's expensive.
            # Here we simply assign all points to one default cluster (-1 often means noise).
            print("Info: Using dummy assignment for DBSCAN predictions during transformation/inference.")
            # A simple workaround could involve nearest neighbor search against core samples,
            # but for now let's assume a constant value like -1 which indicates noise/outliers.
            # Alternatively you may want to store indices from original clustering step.
            # Let's go with assigning everything to most common class found in training.
            unique_labels_in_training = np.unique(self.dbscan.labels_)
            mode_label = unique_labels_in_training[
                np.argmax(np.bincount(self.dbscan.labels_[self.dbscan.labels_ >= 0]))] if any(
                self.dbscan.labels_ >= 0) else -1
            dbscan_labels = np.full(shape=X_dense.shape[0], fill_value=mode_label)
        else:
            # Handle case where DBSCAN wasn't used at all (either intentionally disabled or error state)
            print("Warning: DBSCAN was not properly initialized/fitted. Assigning placeholder values.")
            dbscan_labels = np.zeros(X_dense.shape[0])

        # Append both distance matrix and predicted/assigned cluster labels
        if hasattr(self, 'kmeans') and self.kmeans is not None:
            pass  # Already appended above when computing distances

        cluster_features.append(dbscan_labels.reshape(-1, 1))

        # 合并聚类特征
        cluster_features_array = np.concatenate(cluster_features, axis=1)

        # 添加特征名称
        cluster_names = [f'kmeans_dist_{i}' for i in range(self.n_clusters_kmeans)] if self.kmeans is not None else []
        cluster_names.extend(['kmeans_label', 'dbscan_label'])
        self.feature_names.extend(cluster_names)

        print(f"    -> 创建了 {cluster_features_array.shape[1]} 个聚类特征")
        return np.concatenate([X_dense, cluster_features_array], axis=1)

    def _create_anomaly_features(self, X_dense, training_phase=False):
        """创建异常检测特征"""
        if not self.create_anomaly_features:
            return X_dense

        print("  -> 创建异常检测特征...")
        anomaly_features = []

        # Isolation Forest
        if training_phase and self.isolation_forest is None:
            self.isolation_forest = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=SEED, n_jobs=-1)
            iso_scores = self.isolation_forest.fit(X_dense).decision_function(X_dense)
        elif not training_phase and self.isolation_forest is not None:
            iso_scores = self.isolation_forest.decision_function(X_dense)
        else:
            print("Warning: IsolationForest not fitted correctly; adding zeros instead.")
            iso_scores = np.zeros((X_dense.shape[0],))

        anomaly_features.append(iso_scores.reshape(-1, 1))

        # Local Outlier Factor
        if training_phase and self.lof is None:
            self.lof = LocalOutlierFactor(
                n_neighbors=self.lof_n_neighbors,
                novelty=True, n_jobs=-1)
            self.lof.fit(X_dense)
            lof_scores = self.lof.decision_function(X_dense)
        elif not training_phase and self.lof is not None:
            lof_scores = self.lof.decision_function(X_dense)
        else:
            print("Warning: LOF not fitted correctly; adding zeros instead.")
            lof_scores = np.zeros((X_dense.shape[0],))

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
            # 新增：识别嵌入特征组
            self._identify_embedding_groups(X.columns.tolist())
            # 新增：处理嵌入特征
            X_processed_df = self._process_embedding_features(X)
            # 新增：填充向量列缺失值
            X_imputed_df = self._impute_vector_columns(X_processed_df)

            # 分离数值列和向量列
            self.numerical_features = X_imputed_df.select_dtypes(include=[np.number]).columns.tolist()
            vector_columns = [col for col in self.vector_feature_names if col in X_imputed_df.columns]

            # 将向量列展开为数值列
            vector_data_list = []
            for col in vector_columns:
                if len(X_imputed_df[col]) > 0:
                    # 获取第一个有效向量来确定维度
                    sample_vector = None
                    for vec in X_imputed_df[col]:
                        if isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
                            sample_vector = vec
                            break

                    if sample_vector is not None:
                        expanded_df = pd.DataFrame(X_imputed_df[col].tolist(),
                                                   columns=[f"{col}_dim_{i}" for i in range(len(sample_vector))])
                        vector_data_list.append(expanded_df)

            if vector_data_list:
                vector_data_df = pd.concat(vector_data_list, axis=1)
                numerical_data_df = X_imputed_df[self.numerical_features]
                # 重置索引以避免合并时的索引冲突
                numerical_data_df = numerical_data_df.reset_index(drop=True)
                vector_data_df = vector_data_df.reset_index(drop=True)
                X_final_df = pd.concat([numerical_data_df, vector_data_df], axis=1)
            else:
                X_final_df = X_imputed_df[self.numerical_features]

            X_dense = X_final_df.values

        elif sparse.issparse(X):
            X_dense = X.toarray()
            self.numerical_features = list(range(X.shape[1]))
        else:
            X_dense = X
            self.numerical_features = list(range(X.shape[1]))

        self.original_feature_count = X_dense.shape[1]
        print(f"原始特征数: {self.original_feature_count}")

        # 1. 缺失值处理 (现在处理的是纯数值型数组)
        print("步骤 1: 缺失值处理...")
        self.imputer_num = SimpleImputer(strategy='median')

        # 记录 imputer 期望的特征数 before fit
        self.fit_feature_count_before_imputer_ = X_dense.shape[1]
        print(f"  -> SimpleImputer 拟合前特征数: {self.fit_feature_count_before_imputer_}")

        X_dense = self.imputer_num.fit_transform(X_dense)

        # 记录 imputer 期望的特征数 after fit
        self.fit_feature_count_after_imputer_ = X_dense.shape[1]
        print(f"  -> SimpleImputer 拟合后特征数: {self.fit_feature_count_after_imputer_}")

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
        X_dense = self._create_cluster_features(X_dense, training_phase=True)  # Indicate this is training phase
        X_dense = self._create_anomaly_features(X_dense, training_phase=True)  # Same logic applies here too.

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

        # 记录最终特征数
        self.final_feature_count_after_engineering_ = X_dense.shape[1]
        print(f"特征工程拟合完成！最终特征数: {self.final_feature_count_after_engineering_}")
        self._fitted = True
        return self

    def transform(self, X):
        """应用特征工程变换"""
        if not self._fitted:
            raise RuntimeError(
                "This AdvancedFeatureEngineer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            # 新增：处理嵌入特征 (使用在fit中识别的组)
            X_processed_df = self._process_embedding_features(X)
            # 新增：填充向量列缺失值
            X_imputed_df = self._impute_vector_columns(X_processed_df)

            # 分离数值列和向量列
            # 注意：这里使用 self.numerical_features，它是在 fit 时确定的
            numerical_data_df = X_imputed_df.reindex(columns=self.numerical_features, fill_value=0)  # 安全地选择列
            vector_columns = [col for col in self.vector_feature_names if col in X_imputed_df.columns]

            # 将向量列展开为数值列
            vector_data_list = []
            for col in vector_columns:
                if len(X_imputed_df[col]) > 0:
                    # 获取第一个有效向量来确定维度
                    sample_vector = None
                    for vec in X_imputed_df[col]:
                        if isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
                            sample_vector = vec
                            break

                    if sample_vector is not None:
                        expanded_df = pd.DataFrame(X_imputed_df[col].tolist(),
                                                   columns=[f"{col}_dim_{i}" for i in range(len(sample_vector))])
                        vector_data_list.append(expanded_df)

            if vector_data_list:
                vector_data_df = pd.concat(vector_data_list, axis=1)
                # 重置索引以避免合并时的索引冲突
                numerical_data_df = numerical_data_df.reset_index(drop=True)
                vector_data_df = vector_data_df.reset_index(drop=True)
                X_final_df = pd.concat([numerical_data_df, vector_data_df], axis=1)
            else:
                X_final_df = numerical_data_df  # 如果没有向量特征，就只用 numerical_data_df

            X_dense = X_final_df.values

        elif sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # --- 新增：维度检查点 1 ---
        print(f"  -> transform 阶段，经过 DataFrame 处理后的特征数: {X_dense.shape[1]}")

        # --- 新增：维度一致性检查 before imputer ---
        if self.fit_feature_count_before_imputer_ is not None:
            if X_dense.shape[1] != self.fit_feature_count_before_imputer_:
                print(
                    f"  -> transform 阶段，经过 DataFrame 处理后的特征数 ({X_dense.shape[1]}) 与 fit 阶段的 imputer 输入特征数 ({self.fit_feature_count_before_imputer_}) 不一致。")
                print(f"  -> 调整 X_dense 维度以匹配...")
                if X_dense.shape[1] > self.fit_feature_count_before_imputer_:
                    # 如果 transform 时特征数更多，裁剪
                    X_dense = X_dense[:, :self.fit_feature_count_before_imputer_]
                    print(f"  -> 裁剪至 {self.fit_feature_count_before_imputer_} 个特征")
                elif X_dense.shape[1] < self.fit_feature_count_before_imputer_:
                    # 如果 transform 时特征数更少，用零填充
                    zeros_to_add = self.fit_feature_count_before_imputer_ - X_dense.shape[1]
                    zeros = np.zeros((X_dense.shape[0], zeros_to_add))
                    X_dense = np.concatenate([X_dense, zeros], axis=1)
                    print(f"  -> 填充 {zeros_to_add} 个零特征列")
            else:
                print(f"  -> transform 阶段，经过 DataFrame 处理后的特征数校验通过: {X_dense.shape[1]}")

        # 1. 缺失值处理 (现在处理的是纯数值型数组)
        # --- 新增：维度检查点 2 ---
        print(f"  -> transform 阶段，准备进行 SimpleImputer transform 的特征数: {X_dense.shape[1]}")

        # --- 新增：维度一致性检查 during imputer ---
        if X_dense.shape[1] != self.fit_feature_count_after_imputer_:
            # Imputer 在 fit 时会根据输入调整，但 transform 时要求输入维度一致
            # 我们需要确保 X_dense 的列数与 fit 时一致
            print(
                f"  -> transform 阶段，准备进行 SimpleImputer transform 的特征数 ({X_dense.shape[1]}) 与 fit 阶段的 imputer 输出特征数 ({self.fit_feature_count_after_imputer_}) 不一致。")
            if X_dense.shape[1] != self.fit_feature_count_before_imputer_:
                print(
                    f"  -> 但是，它与 fit 阶段的 imputer 输入特征数 ({self.fit_feature_count_before_imputer_}) 相同，这可能意味着 imputer 没有改变特征数。")
                # 在这种情况下，我们假设 imputer 的输出特征数应该与输入特征数一致
                expected_output_features = self.fit_feature_count_before_imputer_
            else:
                expected_output_features = self.fit_feature_count_after_imputer_

            if X_dense.shape[1] != expected_output_features:
                print(f"  -> 调整 X_dense 维度以符合 imputer 期望...")
                if X_dense.shape[1] > expected_output_features:
                    X_dense = X_dense[:, :expected_output_features]
                    print(f"  -> 裁剪至 {expected_output_features} 个特征")
                elif X_dense.shape[1] < expected_output_features:
                    zeros_to_add = expected_output_features - X_dense.shape[1]
                    zeros = np.zeros((X_dense.shape[0], zeros_to_add))
                    X_dense = np.concatenate([X_dense, zeros], axis=1)
                    print(f"  -> 填充 {zeros_to_add} 个零特征列")

        X_dense = self.imputer_num.transform(X_dense)
        # --- 新增：维度检查点 3 ---
        print(f"  -> transform 阶段，经过 SimpleImputer transform 后的特征数: {X_dense.shape[1]}")

        # 2. 数据预处理
        if self.power_transformer is not None:
            X_dense = self.power_transformer.transform(X_dense)
        X_dense = self.scaler.transform(X_dense)

        # 3. 特征工程 (注意：聚类和异常检测需要特殊处理)
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense,
                                                training_phase=False)  # Apply learned models rather than refit them
        X_dense = self._create_anomaly_features(X_dense, training_phase=False)  # Again apply learned ones vs retraining

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

        # --- 新增：强制维度一致性检查 ---
        if self.final_feature_count_after_engineering_ is not None:
            if X_dense.shape[1] != self.final_feature_count_after_engineering_:
                raise ValueError(
                    f"transform 阶段最终特征数 ({X_dense.shape[1]}) 与 fit 阶段记录的特征数 "
                    f"({self.final_feature_count_after_engineering_}) 不一致。"
                    f"这通常是由于 fit 和 transform 阶段的处理逻辑不一致导致的。"
                )
            else:
                print(f"  -> transform 阶段最终特征数校验通过: {X_dense.shape[1]}")

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
    """使用先进特征工程的训练评估流程"""
    print("\n=== 构建先进特征工程管道 ===")

    # 特征工程
    feature_engineer = AdvancedFeatureEngineer(**feature_engineer_params)
    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # 1. 拟合特征工程器
    print("\n=== 拟合特征工程器 ===")
    feature_engineer.fit(X_train, y_train)
    X_train_engineered = feature_engineer.transform(X_train)

    # 2. 应用 SMOTETomek
    print("\n=== 应用 SMOTETomek 重采样 ===")
    X_train_resampled, y_train_resampled = moo_smotetomek_func(X_train_engineered, y_train)

    # 3. 模型训练
    print("\n=== 开始模型训练 ===")
    start_time = time.time()
    classifier.fit(X_train_resampled, y_train_resampled)
    end_time = time.time()
    print(f"模型训练耗时: {end_time - start_time:.2f} 秒")

    # 4. 保存模型
    trained_pipeline = Pipeline([
        ('feature_engineer', feature_engineer),
        ('classifier', classifier)
    ])
    save_model(trained_pipeline, 'advanced_feature_engineering_model.pkl')

    # 5. 验证集阈值选择
    print("\n=== 验证集阈值选择 ===")
    X_val_engineered = feature_engineer.transform(X_val)
    val_proba = classifier.predict_proba(X_val_engineered)[:, 1]

    threshold_f1 = find_best_f1_threshold(y_val, val_proba)
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # 6. 测试集评估
    print("\n=== 测试集评估 ===")
    X_test_engineered = feature_engineer.transform(X_test)
    test_proba = classifier.predict_proba(X_test_engineered)[:, 1]

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
                      model_name="Advanced Feature Engineering XGBoost")

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
        'trained_model': trained_pipeline
    }


def main():
    # 检查CUDA可用性
    print("CUDA 可用性 (PyTorch):", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # 配置先进特征工程参数
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
        import dask.dataframe as dd
        ddf = dd.read_csv('train_bert_embedded.csv')

        data = ddf.compute()
        # 假设你的数据文件是 train_bert_embedded.csv
        # data = pd.read_csv("train_bert_embedded.csv", engine='pyarrow')
        if "company_id" in data.columns:
            data = data.drop(columns=["company_id"])
        if "target" not in data.columns:
            raise ValueError("数据中必须包含'target'列")

        # 分离特征和目标
        y = data["target"].values.astype(int)
        feature_data = data.drop(columns=["target"])
        X = feature_data  # 传递DataFrame给特征工程器

        print(f"数据加载成功: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
        print(f"正样本比例: {np.mean(y):.4f}")
        print(f"使用的特征工程参数: {feature_engineer_params}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 移除y中的NaN
    if np.isnan(y).any():
        print("移除y中的NaN...")
        mask = ~np.isnan(y)
        X = X.iloc[mask]  # 使用iloc进行索引
        y = y[mask]

    # 数据划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y)
    print(f"\n初步数据划分: 临时集={X_temp.shape[0]}, 测试集={X_test.shape[0]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=SEED, stratify=y_temp)
    print(f"最终数据划分: 训练集={X_train.shape[0]}, 验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")

    # 启动先进特征工程训练流程
    print("\n=== 启动先进特征工程训练流程 ===")
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
        }).to_csv("advanced_predictions.csv", index=False)
        print("\n预测结果已保存到 advanced_predictions.csv")

    # 打印最终总结
    print("\n" + "=" * 50)
    print("特征工程训练完成！")
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



