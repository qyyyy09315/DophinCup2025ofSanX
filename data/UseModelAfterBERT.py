import multiprocessing
import os
import pickle
import random
import warnings

# --- 导入绘图库 ---
import numpy as np
import pandas as pd
# --- 导入稀疏矩阵支持 ---
from scipy import sparse
from scipy.stats import skew, kurtosis
# --- 导入聚类和异常检测用于特征工程 ---
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
# --- 导入先进的特征工程库 ---
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)

# --- 导入 XGBoost 和 CatBoost ---

warnings.filterwarnings('ignore')

# 环境配置
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
    5. 取消降维技术组合
    6. 取消智能特征选择
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

                 # 取消降维参数
                 use_multiple_decomposition=False,  # 设置为False以取消降维
                 svd_components=50,
                 ica_components=30,
                 fa_components=20,

                 # 取消特征选择参数
                 variance_threshold=0.01,
                 univariate_k_best=None,  # 设置为None以取消单变量选择
                 rf_n_features_to_select=None,  # 设置为None以取消随机森林选择

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
        self.use_multiple_decomposition = use_multiple_decomposition  # 降维已取消
        self.svd_components = svd_components
        self.ica_components = ica_components
        self.fa_components = fa_components
        self.variance_threshold = variance_threshold
        self.univariate_k_best = univariate_k_best  # 特征选择已取消
        self.rf_n_features_to_select = rf_n_features_to_select  # 特征选择已取消
        self.force_sparse_output = force_sparse_output

        # 初始化组件
        self.imputer_num = None
        self.scaler = None
        self.power_transformer = None
        self.kmeans = None
        self.dbscan = None
        self.isolation_forest = None
        self.lof = None
        # 取消降维组件
        self.svd = None
        self.ica = None
        self.fa = None
        # 取消特征选择组件
        self.variance_selector = None
        self.univariate_selector = None
        self.rf_selector = None

        # 记录特征信息
        self.original_feature_names = []  # 原始DataFrame列名
        self.final_feature_names_after_processing = []  # fit后经过所有预处理(包括emb展开)的特征名
        self.feature_names_for_modeling = []  # 最终传给模型的特征名
        self.embedding_groups = {}  # 记录*_emb_*形式的特征组

        # 新增：用于记录向量特征列名及其对应的展开维度名
        self.vector_expanded_feature_map = {}

        # 新增：记录经过所有工程处理后的特征数
        self.final_feature_count_after_engineering_ = None
        # 新增：记录在fit阶段的特征数，用于后续一致性检查
        self.fit_feature_count_before_imputer_ = None
        self.fit_feature_count_after_imputer_ = None

    def _identify_embedding_groups(self, feature_names):
        """识别并分组 *_emb_* 形式的特征"""
        print("  -> 识别嵌入特征组...")
        emb_pattern = "_emb_"
        emb_dict = {}

        for feat in feature_names:
            if emb_pattern in feat:
                parts = feat.split(emb_pattern)
                if len(parts) >= 2:
                    prefix = emb_pattern.join(parts[:-1])
                    suffix = parts[-1]
                    if suffix.isdigit():
                        if prefix not in emb_dict:
                            emb_dict[prefix] = []
                        emb_dict[prefix].append(feat)

        # 按照编号排序每个组内的特征
        for prefix, feats in emb_dict.items():
            feats.sort(key=lambda x: int(x.split('_')[-1]))
            self.embedding_groups[prefix] = feats
            print(f"    -> 发现嵌入组 '{prefix}': 包含 {len(feats)} 个特征")

    def _process_embedding_features_fit(self, X_df):
        """FIT专用: 处理嵌入特征：将 *_emb_* 组合并为向量特征，并保留原始向量和范数"""
        if not self.embedding_groups:
            print("  -> 未发现嵌入特征组，跳过处理。")
            return X_df

        print("  -> FIT: 处理嵌入特征组...")
        X_processed_df = X_df.copy()

        # 清空旧映射
        self.vector_expanded_feature_map = {}

        for prefix, emb_features in self.embedding_groups.items():
            print(f"    -> 处理组 '{prefix}' ({len(emb_features)} 个特征)...")

            # 提取嵌入向量
            emb_vectors = X_processed_df[emb_features].values

            # 为新特征命名
            new_vector_feature_name = f"{prefix}_emb_vector"
            new_norm_feature_name = f"{prefix}_emb_norm"

            # 将向量作为新列添加
            X_processed_df[new_vector_feature_name] = list(emb_vectors)

            # 计算向量的L2范数作为另一个特征
            vector_norms = np.linalg.norm(emb_vectors, axis=1)
            X_processed_df[new_norm_feature_name] = vector_norms

            # 删除原始的嵌入特征列
            X_processed_df = X_processed_df.drop(columns=emb_features)
            print(f"    -> 合并为特征 '{new_vector_feature_name}' 和 '{new_norm_feature_name}' 并移除原始列")

            # 记录展开的新特征名 (用于transform时重建DataFrame)
            # 关键改进：即使向量可能为空，也预先定义好维度名
            # 假设维度是固定的，基于原始emb_features的数量
            dim_names = [f"{prefix}_emb_dim_{i}" for i in range(len(emb_features))]
            self.vector_expanded_feature_map[new_vector_feature_name] = dim_names

        print(f"  -> FIT: 嵌入特征处理完成，剩余特征数: {X_processed_df.shape[1]}")
        return X_processed_df

    def _process_embedding_features_transform(self, X_df):
        """TRANSFORM专用: 处理嵌入特征"""
        if not self.embedding_groups:
            print("  -> TRANSFORM: 未发现嵌入特征组，跳过处理。")
            return X_df

        print("  -> TRANSFORM: 处理嵌入特征组...")
        X_processed_df = X_df.copy()

        for prefix, emb_features in self.embedding_groups.items():
            print(f"    -> TRANSFORM: 处理组 '{prefix}' ({len(emb_features)} 个特征)...")

            # 提取嵌入向量
            emb_vectors = X_processed_df[emb_features].values

            # 为新特征命名
            new_vector_feature_name = f"{prefix}_emb_vector"
            new_norm_feature_name = f"{prefix}_emb_norm"

            # 将向量作为新列添加
            X_processed_df[new_vector_feature_name] = list(emb_vectors)

            # 计算向量的L2范数作为另一个特征
            vector_norms = np.linalg.norm(emb_vectors, axis=1)
            X_processed_df[new_norm_feature_name] = vector_norms

            # 删除原始的嵌入特征列
            X_processed_df = X_processed_df.drop(columns=emb_features)
            print(f"    -> TRANSFORM: 合并为特征 '{new_vector_feature_name}' 和 '{new_norm_feature_name}' 并移除原始列")

        print(f"  -> TRANSFORM: 嵌入特征处理完成，剩余特征数: {X_processed_df.shape[1]}")
        return X_processed_df

    def _impute_vector_columns(self, X_df):
        """使用零向量填充向量列中的缺失值"""
        vector_feature_names = list(self.vector_expanded_feature_map.keys())  # 使用fit阶段学到的名字
        if not vector_feature_names:
            print("  -> 未发现向量特征列，跳过向量缺失值填充。")
            return X_df

        print("  -> 使用零向量填充向量列缺失值...")
        X_imputed_df = X_df.copy()

        for vec_col_name in vector_feature_names:
            if vec_col_name not in X_imputed_df.columns:
                print(f"    -> 警告: 列 '{vec_col_name}' 在当前数据集中不存在，跳过。")
                continue

            # 获取该列的第一个非空元素来确定向量维度
            # 关键改进：使用预先定义的维度名来确定维度
            expected_dims = self.vector_expanded_feature_map.get(vec_col_name, [])
            if not expected_dims:
                print(f"    -> 警告: 列 '{vec_col_name}' 没有预定义的维度信息，跳过填充。")
                continue

            vector_dim = len(expected_dims)
            zero_vector = np.zeros(vector_dim)

            def fill_na_with_zero_vector(x):
                try:
                    if not isinstance(x, (list, np.ndarray)):
                        if pd.isna(x):
                            return zero_vector
                        else:
                            # 如果不是数组但不是NA，尝试转换，失败则用零向量
                            try:
                                x_array = np.asarray(x)
                                if x_array.size == 0 or (
                                        x_array.dtype.kind in ['f', 'c'] and np.all(np.isnan(x_array))):
                                    return zero_vector
                                return x_array
                            except:
                                return zero_vector
                    x_array = np.asarray(x)
                    if x_array.size == 0:
                        return zero_vector
                    if x_array.dtype.kind in ['f', 'c']:
                        if np.all(np.isnan(x_array)):
                            return zero_vector
                    # 检查维度是否匹配
                    if x_array.shape[0] != vector_dim:
                        # 维度不匹配，用零向量填充或截断
                        if x_array.shape[0] < vector_dim:
                            padded = np.zeros(vector_dim)
                            padded[:x_array.shape[0]] = x_array
                            return padded
                        else:  # x_array.shape[0] > vector_dim
                            return x_array[:vector_dim]
                    return x_array
                except Exception as e:
                    print(f"      -> 处理值时出错: {e}, 使用零向量填充")
                    return zero_vector

            X_imputed_df[vec_col_name] = X_imputed_df[vec_col_name].apply(fill_na_with_zero_vector)
            print(
                f"    -> 列 '{vec_col_name}' 已用维度为 {vector_dim} 的零向量填充缺失值/空向量/全NaN向量/维度不匹配向量。")
        return X_imputed_df

    def _create_statistical_features(self, X_dense):
        """创建统计特征"""
        if not self.create_statistical_features:
            return X_dense

        print("  -> 创建统计特征...")
        stat_features = []

        stat_features.append(np.mean(X_dense, axis=1, keepdims=True))
        stat_features.append(np.std(X_dense, axis=1, keepdims=True))
        stat_features.append(np.max(X_dense, axis=1, keepdims=True))
        stat_features.append(np.min(X_dense, axis=1, keepdims=True))
        stat_features.append(np.median(X_dense, axis=1, keepdims=True))

        stat_features.append(skew(X_dense, axis=1).reshape(-1, 1))
        stat_features.append(kurtosis(X_dense, axis=1).reshape(-1, 1))

        stat_features.append(np.percentile(X_dense, 25, axis=1, keepdims=True))
        stat_features.append(np.percentile(X_dense, 75, axis=1, keepdims=True))

        mean_vals = np.mean(X_dense, axis=1, keepdims=True)
        std_vals = np.std(X_dense, axis=1, keepdims=True)
        cv = np.divide(std_vals, mean_vals + 1e-8)
        stat_features.append(cv)

        stat_features_array = np.concatenate(stat_features, axis=1)

        stat_names = ['row_mean', 'row_std', 'row_max', 'row_min', 'row_median',
                      'row_skew', 'row_kurtosis', 'row_q25', 'row_q75', 'row_cv']

        # 更新建模特征名列表
        self.feature_names_for_modeling.extend(stat_names)

        print(f"    -> 创建了 {stat_features_array.shape[1]} 个统计特征")
        return np.concatenate([X_dense, stat_features_array], axis=1)

    def _create_cluster_features(self, X_dense):
        """创建聚类特征"""
        if not self.create_cluster_features:
            return X_dense

        print("  -> 创建聚类特征...")
        cluster_features = []

        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_clusters_kmeans,
                                 random_state=SEED, n_init=10)
            kmeans_labels = self.kmeans.fit_predict(X_dense)
        else:
            kmeans_labels = self.kmeans.predict(X_dense)

        kmeans_distances = self.kmeans.transform(X_dense)
        cluster_features.append(kmeans_distances)

        if self.dbscan is None:
            self.dbscan = DBSCAN(eps=self.dbscan_eps,
                                 min_samples=self.dbscan_min_samples, n_jobs=-1)
            dbscan_labels = self.dbscan.fit_predict(X_dense)
        else:
            dbscan_labels = np.zeros(X_dense.shape[0])

        cluster_features.append(kmeans_labels.reshape(-1, 1))
        cluster_features.append(dbscan_labels.reshape(-1, 1))

        cluster_features_array = np.concatenate(cluster_features, axis=1)

        cluster_names = [f'kmeans_dist_{i}' for i in range(self.n_clusters_kmeans)]
        cluster_names.extend(['kmeans_label', 'dbscan_label'])

        # 更新建模特征名列表
        self.feature_names_for_modeling.extend(cluster_names)

        print(f"    -> 创建了 {cluster_features_array.shape[1]} 个聚类特征")
        return np.concatenate([X_dense, cluster_features_array], axis=1)

    def _create_anomaly_features(self, X_dense):
        """创建异常检测特征"""
        if not self.create_anomaly_features:
            return X_dense

        print("  -> 创建异常检测特征...")
        anomaly_features = []

        if self.isolation_forest is None:
            self.isolation_forest = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=SEED, n_jobs=-1)
            iso_scores = self.isolation_forest.fit(X_dense).decision_function(X_dense)
        else:
            iso_scores = self.isolation_forest.decision_function(X_dense)

        anomaly_features.append(iso_scores.reshape(-1, 1))

        if self.lof is None:
            self.lof = LocalOutlierFactor(
                n_neighbors=self.lof_n_neighbors,
                novelty=True, n_jobs=-1)
            self.lof.fit(X_dense)
            lof_scores = self.lof.decision_function(X_dense)
        else:
            lof_scores = self.lof.decision_function(X_dense)

        anomaly_features.append(lof_scores.reshape(-1, 1))

        anomaly_features_array = np.concatenate(anomaly_features, axis=1)

        anomaly_names = ['isolation_score', 'lof_score']

        # 更新建模特征名列表
        self.feature_names_for_modeling.extend(anomaly_names)

        print(f"    -> 创建了 {anomaly_features_array.shape[1]} 个异常检测特征")
        return np.concatenate([X_dense, anomaly_features_array], axis=1)

    def fit(self, X, y=None):
        """拟合特征工程管道"""
        print("开始高级特征工程拟合...")

        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            self.original_feature_names = X.columns.tolist()

            # Step 1: Identify embedding groups
            self._identify_embedding_groups(self.original_feature_names)

            # Step 2: Process embeddings (Fit specific logic)
            X_processed_df = self._process_embedding_features_fit(X)

            # Record names after processing embeddings but before expanding vectors
            post_emb_processing_names = X_processed_df.columns.tolist()

            # Step 3: Impute vector columns
            X_imputed_df = self._impute_vector_columns(X_processed_df)

            # Step 4: Expand vector features into individual numeric columns
            expanded_dfs = []
            other_cols = []

            print(f"  -> 开始展开向量特征...")
            for col in post_emb_processing_names:
                if col in self.vector_expanded_feature_map:  # It's a vector column to expand
                    print(f"    -> 展开向量列 '{col}'...")
                    vec_data = X_imputed_df[col]
                    expected_dims = self.vector_expanded_feature_map[col]
                    expected_dim_count = len(expected_dims)

                    if len(vec_data) > 0:
                        # Convert list of arrays/vectors to a list of lists for DataFrame constructor
                        # Ensure all vectors have the correct length
                        validated_vectors = []
                        for i, vec in enumerate(vec_data):
                            try:
                                vec_array = np.asarray(vec)
                                if vec_array.size == 0:
                                    validated_vectors.append(np.zeros(expected_dim_count))
                                elif vec_array.shape[0] != expected_dim_count:
                                    # Handle dimension mismatch
                                    if vec_array.shape[0] < expected_dim_count:
                                        padded = np.zeros(expected_dim_count)
                                        padded[:vec_array.shape[0]] = vec_array
                                        validated_vectors.append(padded)
                                    else:
                                        validated_vectors.append(vec_array[:expected_dim_count])
                                else:
                                    validated_vectors.append(vec_array)
                            except Exception as e:
                                print(f"      -> 警告: 处理第 {i} 个向量时出错 ({e})，使用零向量填充。")
                                validated_vectors.append(np.zeros(expected_dim_count))

                        try:
                            expanded_df = pd.DataFrame(validated_vectors, columns=expected_dims)
                            expanded_dfs.append(expanded_df)
                            print(f"      -> 成功展开 '{col}' 为 {len(expected_dims)} 个特征。")
                        except Exception as e:
                            print(f"      -> 错误: 无法从列 '{col}' 创建DataFrame ({e})，跳过该列。")
                            # Even if expansion fails, we should not crash.
                            # Add zero columns as placeholders if needed for consistency?
                            # For now, we just skip. The final concat will handle missing parts.
                    else:
                        print(f"      -> 警告: 向量列 '{col}' 为空，跳过展开。")
                        # Optionally add zero columns here too, but let's see if it's needed.
                else:
                    # Regular numeric or non-vector column
                    other_cols.append(col)

            # Combine all parts back together
            final_parts = []
            if other_cols:
                final_parts.append(X_imputed_df[other_cols])
                print(f"  -> 添加了 {len(other_cols)} 个非向量特征。")
            if expanded_dfs:
                final_parts.extend(expanded_dfs)
                total_expanded_features = sum(df.shape[1] for df in expanded_dfs)
                print(f"  -> 添加了 {len(expanded_dfs)} 个向量组，共 {total_expanded_features} 个展开特征。")

            if final_parts:
                X_final_df = pd.concat(final_parts, axis=1)
                # Ensure consistent order by sorting column names alphabetically
                sorted_cols = sorted(X_final_df.columns)
                X_final_df = X_final_df.reindex(columns=sorted_cols)
                print(f"  -> 合并后特征总数: {X_final_df.shape[1]} (行数: {X_final_df.shape[0]})")
            else:
                print("  -> 警告: 最终特征DataFrame为空！这将导致后续错误。")
                X_final_df = pd.DataFrame(index=X.index)  # At least keep the index

            # Store the final processed feature names that will go into imputer/scaler etc.
            self.final_feature_names_after_processing = X_final_df.columns.tolist()
            print(f"Final pre-imputation feature names count: {len(self.final_feature_names_after_processing)}")

            X_dense = X_final_df.values
            print(f"  -> 转换为 NumPy 数组，形状: {X_dense.shape}")

        elif sparse.issparse(X):
            X_dense = X.toarray()
            # For sparse input, we assume no special preprocessing like embeddings was needed
            self.final_feature_names_after_processing = [f"feature_{i}" for i in range(X_dense.shape[1])]
        else:
            X_dense = X
            self.final_feature_names_after_processing = [f"feature_{i}" for i in range(X_dense.shape[1])]

        self.original_feature_count = X_dense.shape[1]
        print(f"原始特征数 (转换为dense array后): {self.original_feature_count}")

        # 1. 缺失值处理
        print("步骤 1: 缺失值处理...")
        self.imputer_num = SimpleImputer(strategy='median')
        self.fit_feature_count_before_imputer_ = X_dense.shape[1]
        print(f"  -> SimpleImputer 拟合前特征数: {self.fit_feature_count_before_imputer_} (形状: {X_dense.shape})")
        if X_dense.shape[0] == 0:
            raise ValueError("错误：在SimpleImputer拟合前，特征矩阵行数为0。请检查数据预处理逻辑。")
        X_dense = self.imputer_num.fit_transform(X_dense)
        self.fit_feature_count_after_imputer_ = X_dense.shape[1]
        print(f"  -> SimpleImputer 拟合后特征数: {self.fit_feature_count_after_imputer_} (形状: {X_dense.shape})")

        # 2. 数据预处理和变换
        print("步骤 2: 数据预处理...")
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
        print(f"  -> 预处理后特征矩阵形状: {X_dense.shape}")

        # 3. 高级特征工程 - Initialize modeling feature names with base ones
        self.feature_names_for_modeling = self.final_feature_names_after_processing.copy()
        print("步骤 3: 高级特征工程...")
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)

        print(f"特征工程后特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        # 4. 取消多重降维
        if self.use_multiple_decomposition:
            print("步骤 4: 多重降维... (已取消)")
        else:
            print("步骤 4: 多重降维已取消")

        # 5. 取消智能特征选择
        print("步骤 5: 智能特征选择... (已取消)")

        # 记录最终特征数
        self.final_feature_count_after_engineering_ = X_dense.shape[1]
        print(f"特征工程拟合完成！最终特征数: {self.final_feature_count_after_engineering_} (形状: {X_dense.shape})")
        if X_dense.shape[0] == 0:
            raise ValueError("错误：特征工程完成后，特征矩阵行数为0。这是致命错误，请检查数据和特征工程逻辑。")
        return self

    def transform(self, X):
        """应用特征工程变换"""
        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            # Step 1: Process embeddings using fitted info (Transform specific logic)
            X_processed_df = self._process_embedding_features_transform(X)

            # Step 2: Impute vector columns based on what was learned during fit
            X_imputed_df = self._impute_vector_columns(X_processed_df)

            # Step 3: Expand vector features into individual numeric columns using mapping from fit
            expanded_dfs = []
            other_cols_from_fit = []

            current_post_process_names = set(X_imputed_df.columns)

            # Reconstruct structure similar to fit phase
            print(f"  -> Transform: 开始展开向量特征...")
            # Handle vector expansions first based on fit-time map
            for vec_col_name, expected_dims in self.vector_expanded_feature_map.items():
                if vec_col_name in X_imputed_df.columns:
                    print(f"    -> Transform: 展开向量列 '{vec_col_name}'...")
                    raw_vectors = X_imputed_df[vec_col_name].tolist()
                    expected_dim_count = len(expected_dims)

                    # Validate dimensions match expectation from fit
                    validated_vectors = []
                    for i, v in enumerate(raw_vectors):
                        try:
                            if isinstance(v, (list, np.ndarray)):
                                v_array = np.asarray(v)
                                if v_array.size == 0:
                                    validated_vectors.append(np.zeros(expected_dim_count))
                                elif v_array.shape[0] != expected_dim_count:
                                    # Handle dimension mismatch
                                    if v_array.shape[0] < expected_dim_count:
                                        padded = np.zeros(expected_dim_count)
                                        padded[:v_array.shape[0]] = v_array
                                        validated_vectors.append(padded)
                                    else:
                                        validated_vectors.append(v_array[:expected_dim_count])
                                else:
                                    validated_vectors.append(v_array)
                            else:
                                # If not array-like, try to convert or use zero vector
                                print(f"      -> 警告: 第 {i} 个元素不是数组 ({type(v)})，使用零向量。")
                                validated_vectors.append(np.zeros(expected_dim_count))
                        except Exception as e:
                            print(f"      -> 警告: 处理第 {i} 个向量时出错 ({e})，使用零向量。")
                            validated_vectors.append(np.zeros(expected_dim_count))

                    try:
                        expanded_df = pd.DataFrame(validated_vectors, columns=expected_dims)
                        expanded_dfs.append(expanded_df)
                        print(f"      -> Transform: 成功展开 '{vec_col_name}' 为 {len(expected_dims)} 个特征。")
                    except Exception as e:
                        print(f"      -> Transform: 错误: 无法从列 '{vec_col_name}' 创建DataFrame ({e})，添加零列。")
                        # Add zero columns as fallback
                        zero_df = pd.DataFrame(np.zeros((len(X), expected_dim_count)), columns=expected_dims)
                        expanded_dfs.append(zero_df)
                else:
                    print(f"    -> Transform: 警告: 向量列 '{vec_col_name}' 在transform数据中不存在。")

            # Now handle other columns that were present in fit
            for original_col_name in self.final_feature_names_after_processing:
                # Check if it's an expanded dimension name handled above
                parent_vec_found = False
                for _, dims in self.vector_expanded_feature_map.items():
                    if original_col_name in dims:
                        parent_vec_found = True
                        break

                if not parent_vec_found and original_col_name in current_post_process_names:
                    other_cols_from_fit.append(original_col_name)

            # Combine everything again
            final_parts_transform = []
            if other_cols_from_fit:
                final_parts_transform.append(X_imputed_df[other_cols_from_fit])
                print(f"  -> Transform: 添加了 {len(other_cols_from_fit)} 个非向量特征。")
            if expanded_dfs:
                final_parts_transform.extend(expanded_dfs)
                total_expanded_features = sum(df.shape[1] for df in expanded_dfs)
                print(f"  -> Transform: 添加了 {len(expanded_dfs)} 个向量组，共 {total_expanded_features} 个展开特征。")

            if final_parts_transform:
                X_final_df_transform = pd.concat(final_parts_transform, axis=1)
                # Align column order with fit stage result
                aligned_cols = [col for col in self.final_feature_names_after_processing
                                if col in X_final_df_transform.columns]
                missing_cols = [col for col in self.final_feature_names_after_processing
                                if col not in X_final_df_transform.columns]

                if missing_cols:
                    print(f"Warning: Missing columns in transform data compared to fit: {missing_cols[:5]}...")
                    # Add missing columns filled with zeros
                    for mc in missing_cols:
                        X_final_df_transform[mc] = 0.0

                # Final reordering to ensure exact match with fit output schema
                X_final_df_transform = X_final_df_transform.reindex(columns=self.final_feature_names_after_processing)
                print(
                    f"  -> Transform: 合并后特征总数: {X_final_df_transform.shape[1]} (行数: {X_final_df_transform.shape[0]})")
            else:
                print("  -> Transform: 警告: 最终特征DataFrame为空！这将导致后续错误。")
                # Fallback: Create empty dataframe matching expected schema
                X_final_df_transform = pd.DataFrame(columns=self.final_feature_names_after_processing, index=X.index)
                # Fill with zeros if needed
                if len(self.final_feature_names_after_processing) > 0:
                    X_final_df_transform = X_final_df_transform.fillna(0.0)

            X_dense = X_final_df_transform.values
            print(f"  -> Transform: 转换为 NumPy 数组，形状: {X_dense.shape}")

        elif sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Dimension check points remain largely unchanged since core issue fixed above
        print(
            f"  -> transform 阶段，准备进行 SimpleImputer transform 的特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        # Consistency checks adjusted slightly
        if X_dense.shape[1] != self.fit_feature_count_before_imputer_:
            print(
                f"  -> transform 阶段，准备进行 SimpleImputer transform 的特征数 ({X_dense.shape[1]}) "
                f"与 fit 阶段的 imputer 输入特征数 ({self.fit_feature_count_before_imputer_}) 不一致。"
            )

            # Adjust shape to match exactly what imputer expects
            required_shape = (X_dense.shape[0], self.fit_feature_count_before_imputer_)
            if X_dense.shape[0] == required_shape[0]:  # Rows must match
                if X_dense.shape[1] < required_shape[1]:
                    pad_width = required_shape[1] - X_dense.shape[1]
                    padding = np.zeros((required_shape[0], pad_width))
                    X_dense = np.hstack([X_dense, padding])
                    print(f"     -> Padding {pad_width} columns to reach correct width.")

                elif X_dense.shape[1] > required_shape[1]:
                    X_dense = X_dense[:, :required_shape[1]]
                    print(f"     -> Truncating to {required_shape[1]} columns to match.")

        print(f"  -> transform 阶段，SimpleImputer transform 前形状: {X_dense.shape}")
        if X_dense.shape[0] == 0:
            raise ValueError("错误：在SimpleImputer transform前，特征矩阵行数为0。")

        # Apply transformations step-by-step ensuring shapes align properly now
        X_dense = self.imputer_num.transform(X_dense)
        print(
            f"  -> transform 阶段，经过 SimpleImputer transform 后的特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        if self.power_transformer is not None:
            X_dense = self.power_transformer.transform(X_dense)
        X_dense = self.scaler.transform(X_dense)
        print(f"  -> transform 阶段，经过 Scaler transform 后的特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        # Recreate engineered features just like in fit
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)
        print(f"  -> transform 阶段，特征工程后特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        # Skip decomposition steps as per config
        if self.use_multiple_decomposition:
            print("  -> transform 阶段，多重降维... (已取消)")
        else:
            print("  -> transform 阶段，多重降维已取消")

        # Skip selection steps as per config
        print("  -> transform 阶段，特征选择... (已取消)")

        # Final consistency check enforced strictly
        if self.final_feature_count_after_engineering_ is not None:
            if X_dense.shape[1] != self.final_feature_count_after_engineering_:
                error_msg = (
                    f"transform 阶段最终特征数 ({X_dense.shape[1]}) 与 fit 阶段记录的特征数 "
                    f"({self.final_feature_count_after_engineering_}) 不一致。\n"
                    f"这通常是由于 fit 和 transform 阶段的处理逻辑不一致导致的。\n"
                    f"请检查是否有新增/遗漏字段或者类型变化等问题。"
                )
                raise ValueError(error_msg)
            else:
                print(f"  -> transform 阶段最终特征数校验通过: {X_dense.shape[1]}")

        # Output format control
        if self.force_sparse_output:
            return sparse.csr_matrix(X_dense)
        else:
            return X_dense

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)


# 以上填写对象
def load_model(model_path):
    """加载已保存的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"模型已从 {model_path} 加载。")
    return model


def predict_with_model(model, test_data_path, output_path):
    """
    使用加载的模型对新的测试数据进行预测。
    修改此函数以处理可能不是 sklearn Pipeline 的模型。

    参数:
    model: 通过pickle加载的完整训练模型（可能包含特征工程管道）。
    test_data_path (str): 待分类的 parquet 文件路径。
    output_path (str): 预测结果保存的 CSV 文件路径。
    """
    print(f"\n=== 开始加载测试数据 ===")
    # 1. 加载测试数据
    try:
        test_data = pd.read_parquet(test_data_path)
        print(f"测试数据加载成功: 特征数={test_data.shape[1]}, 样本数={test_data.shape[0]}")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return

    # 2. 处理 'company_id' 列
    if 'company_id' in test_data.columns:
        uuid_column = test_data['company_id']
        # 从特征中移除 'company_id'
        feature_data = test_data.drop(columns=['company_id'])
        print("已找到 'company_id' 列，将用作 uuid。")
    else:
        print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
        uuid_column = test_data.index
        feature_data = test_data

    # 3. 应用模型进行预测
    print(f"\n=== 开始进行预测 ===")
    try:
        # 尝试不同的模型结构
        # 情况 1: 模型是一个 sklearn Pipeline
        if hasattr(model, 'steps'):
            print("检测到模型为 sklearn Pipeline...")
            # 检查最后一步是否是分类器
            if hasattr(model.steps[-1][1], 'predict_proba'):
                print("Pipeline 最后一步是分类器，直接调用 predict_proba...")
                y_proba = model.predict_proba(feature_data)[:, 1]
            else:
                # 如果最后一步不是分类器，假设倒数第二步是特征工程，最后一步是分类器
                # 或者模型结构有问题，需要特殊处理
                print("Pipeline 最后一步不是分类器，尝试手动处理...")
                # 获取特征工程部分
                feature_engineer = model.named_steps.get('feature_engineer', None)
                if feature_engineer is None:
                    # 如果没有命名为 'feature_engineer' 的步骤，尝试获取第一个非分类器步骤
                    feature_engineer = None
                    for name, step in model.steps[:-1]:
                        if not hasattr(step, 'predict_proba'):  # 不是分类器
                            feature_engineer = step
                            break
                if feature_engineer is None:
                    raise ValueError("无法在 Pipeline 中找到特征工程步骤。")

                # 获取分类器部分
                classifier = model.steps[-1][1]
                if not hasattr(classifier, 'predict_proba'):
                    raise ValueError("Pipeline 的最后一步不是有效的分类器。")

                # 应用特征工程
                print("应用特征工程...")
                X_transformed = feature_engineer.transform(feature_data)
                print(f"特征工程完成，变换后形状: {X_transformed.shape}")

                # 应用分类器
                print("应用分类器进行预测...")
                y_proba = classifier.predict_proba(X_transformed)[:, 1]

        # 情况 2: 模型是一个包含 'feature_engineer' 和 'classifier' 属性的对象
        elif hasattr(model, 'feature_engineer') and hasattr(model, 'classifier'):
            print("检测到模型为自定义对象 (包含 feature_engineer 和 classifier)...")
            # 应用特征工程
            print("应用特征工程...")
            X_transformed = model.feature_engineer.transform(feature_data)
            print(f"特征工程完成，变换后形状: {X_transformed.shape}")

            # 应用分类器
            print("应用分类器进行预测...")
            y_proba = model.classifier.predict_proba(X_transformed)[:, 1]

        # 情况 3: 模型本身就是一个已经训练好的分类器，假设特征工程已在外部完成或数据无需工程
        elif hasattr(model, 'predict_proba'):
            print("检测到模型为直接的分类器...")
            y_proba = model.predict_proba(feature_data)[:, 1]

        else:
            raise ValueError("加载的模型对象格式未知，无法进行预测。请检查模型保存方式。")

        # 使用模型在训练时找到的最佳阈值进行分类
        # 假设模型 Pipeline 中存储了阈值，或者我们使用一个固定的阈值（例如 0.5）
        # 这里我们尝试从模型对象中获取，如果获取不到则使用默认值
        # 注意：原训练代码没有直接将阈值保存到模型对象中，
        # 因此我们需要在预测时重新定义或使用默认值。
        # 为了演示，我们在这里使用 0.7 作为默认阈值。
        # 在实际应用中，最好将训练时的最佳阈值也保存下来。

        # 尝试从模型或其属性中获取阈值（这部分依赖于训练脚本如何保存阈值）
        # 一种方法是在训练后将阈值作为模型对象的属性保存
        # 例如，在训练脚本的最后添加: trained_pipeline.threshold_ = threshold_f1
        # 如果没有这样做，我们使用默认阈值
        default_threshold = 0.7
        # 尝试从模型对象中获取阈值 (假设在训练后有这一步)
        threshold = getattr(model, 'threshold_', default_threshold)
        print(f"使用阈值 {threshold} 进行分类。")

        y_pred = (y_proba >= threshold).astype(int)
        print("预测完成。")
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 创建结果 DataFrame
    results_df = pd.DataFrame({
        'uuid': uuid_column,
        'proba': y_proba,
        'prediction': y_pred
    })
    print("结果数据框创建成功。")

    # 5. 保存结果
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n预测结果已保存到 {output_path}")
    except Exception as e:
        print(f"保存预测结果失败: {e}")


def main():
    """主函数"""
    # --- 配置区域 ---
    # 请根据实际情况修改以下路径
    model_path = '../UseBert/advanced_feature_engineering_ensemble_model.pkl'  # 训练脚本保存的模型文件路径
    test_data_path = '../UseBert/test_bert_embedded.parquet'  # 待分类的 parquet 文件路径
    output_path = r'C:\Users\YKSHb\Desktop\submit_template.csv'  # 预测结果保存路径
    # --- 配置区域结束 ---

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。请确保模型已训练并保存。")
        return

    if not os.path.exists(test_data_path):
        print(f"错误: 找不到测试数据文件 '{test_data_path}'。")
        return

    # 加载模型
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 进行预测
    predict_with_model(model, test_data_path, output_path)


if __name__ == "__main__":
    main()



