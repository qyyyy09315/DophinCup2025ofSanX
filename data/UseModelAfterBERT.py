import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- 导入特征工程类 ---
# 这里需要包含您提供的 AdvancedFeatureEngineer 类的完整定义
# 为保持代码完整性，将类定义直接包含在此脚本中

import multiprocessing
import os
import random
import time
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, f_classif, \
    mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import sparse
from scipy.stats import skew, kurtosis


# 环境配置 - 最大化CPU利用率
NUM_CORES = multiprocessing.cpu_count()
print(f"检测到 {NUM_CORES} 个CPU核心")
os.environ["LOKY_MAX_CPU_COUNT"] = str(NUM_CORES)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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
                 rolling_windows=None,  # 滚动统计窗口

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
        self.rolling_windows = rolling_windows or [3, 5, 10]  # 默认值
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


    def _identify_embedding_groups(self, feature_names):
        """识别并分组 *_emb_* 形式的特征"""
        print("  -> 识别嵌入特征组...")
        emb_pattern = "_emb_"  # 修改为新的模式
        self.embedding_groups.clear()
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

        # 记录最终特征数
        self.final_feature_count_after_engineering_ = X_dense.shape[1]
        print(f"特征工程拟合完成！最终特征数: {self.final_feature_count_after_engineering_}")
        return self

    def transform(self, X):
        """应用特征工程变换"""
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
                print(f"  -> 尝试调整 X_dense 维度以匹配...")
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
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=42, n_jobs=-1),
            tomek=TomekLinks(sampling_strategy='majority', n_jobs=-1),
            random_state=42
        )

        X_resampled_dense, y_resampled = smotetomek.fit_resample(X_dense, y)

        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled_dense.shape[0]}")
        return X_resampled_dense, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y


def load_and_apply_model():
    # 1. 加载测试数据
    test_data_path = '../UseBert/train_bert_embedded.csv'  # 修改数据路径
    print(f"正在加载测试数据: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    print(f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1]}")

    # 2. 加载完整的训练管道 (特征工程器 + 分类器)
    # 请确保 'advanced_feature_engineering_model.pkl' 文件与脚本在同一目录下，或提供完整路径
    model_path = 'advanced_feature_engineering_model.pkl'
    print(f"正在加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        trained_pipeline = pickle.load(f)

    print("模型加载成功。")

    # 从加载的管道中提取特征工程器和分类器
    feature_engineer = trained_pipeline.named_steps['feature_engineer']
    classifier = trained_pipeline.named_steps['classifier']
    if not hasattr(feature_engineer, 'embedding_groups'):
        feature_engineer.embedding_groups = {}
        print("  -> 已修复模型: 添加缺失的embedding_groups属性")

    # 3. 准备测试特征并应用特征工程
    # 假设测试数据不包含 'target' 列，如果包含则移除
    feature_columns = [col for col in test_data.columns if col != 'target']
    if 'company_id' in feature_columns:
        X_test_raw = test_data[feature_columns].drop(columns=['company_id'])
        uuid_column = test_data['company_id']
    else:
        X_test_raw = test_data[feature_columns]
        uuid_column = test_data.index  # 如果没有company_id，使用索引

    print(f"用于特征工程的特征数: {X_test_raw.shape[1]}")

    print("正在进行特征工程变换...")
    X_test_engineered = feature_engineer.transform(X_test_raw)
    print(f"特征工程完成，变换后特征数: {X_test_engineered.shape[1]}")

    # 4. 应用模型预测
    print("正在进行预测...")
    # 获取预测概率
    y_proba = classifier.predict_proba(X_test_engineered)[:, 1]  # 获取正类概率
    print(f"预测完成，获得 {len(y_proba)} 个概率值。")

    # 5. 应用阈值进行分类
    # 使用与训练时相同的阈值选择方法或固定阈值
    # 这里我们使用一个示例阈值，实际应用中应根据验证集性能选择
    # 为了演示，假设我们使用0.5作为阈值，但在实际应用中应使用验证集找到的最佳阈值
    # 例如，如果在训练脚本中找到的阈值是 results['threshold']，则应加载并使用该值
    # 为了兼容性，这里先使用一个示例阈值

    # 为了更准确，我们可以根据模型性能报告选择一个阈值
    # 例如，在训练脚本中，最佳F1阈值是 find_best_f1_threshold 的返回值
    # 为了演示，我们假设一个阈值，实际使用时应从训练结果中获取
    # 如果有验证集上的最佳阈值文件，可以加载它
    # 例如，如果训练脚本保存了阈值到文件，可以加载
    # 或者，我们在这里使用一个常见的默认值，或者基于训练时的阈值
    # 由于我们无法直接访问训练时的阈值，这里使用一个示例值
    # 实际上，您应该在训练完成后将阈值保存到文件，然后在这里加载
    # 例如，您可以将 find_best_f1_threshold 的返回值保存为 'best_threshold.pkl'
    # 下面是一个示例，假设阈值为0.5，但实际应用中应使用训练时确定的阈值

    # 为了演示，我们使用0.5作为默认阈值，但更推荐使用训练时找到的最佳阈值
    # 假设我们知道最佳阈值是0.55（这需要从训练脚本中获取）
    threshold = 0.7  # 这应该替换为训练时找到的最佳阈值
    print(f"应用分类阈值: {threshold}")
    y_pred = (y_proba >= threshold).astype(int)
    print(f"分类完成。")

    # 6. 创建结果数据框
    results_df = pd.DataFrame({
        'uuid': uuid_column,
        'proba': y_proba,
        'prediction': y_pred
    })
    print("结果数据框创建成功。")

    # 7. 保存结果
    output_path = r'C:\Users\YKSHb\Desktop\submit_template.csv'  # 修改输出文件名
    print(f"正在保存结果到: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"预测完成，使用阈值 {threshold} 进行分类，结果已保存到 {output_path}")


if __name__ == "__main__":
    load_and_apply_model()



