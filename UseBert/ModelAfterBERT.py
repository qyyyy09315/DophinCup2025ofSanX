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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
# --- 导入稀疏矩阵支持 ---
from scipy import sparse
from scipy.stats import skew, kurtosis
# --- 导入聚类和异常检测用于特征工程 ---
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
# --- 导入先进的特征工程库 ---
from sklearn.impute import SimpleImputer
from sklearn.metrics import (recall_score, precision_score, roc_auc_score, f1_score, accuracy_score,
                             precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)
# --- 导入 XGBoost 和 CatBoost ---
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
            first_valid_entry = X_imputed_df[vec_col_name].dropna().iloc[0] if not X_imputed_df[
                vec_col_name].dropna().empty else None
            if first_valid_entry is not None:
                if isinstance(first_valid_entry, (list, np.ndarray)):
                    vector_dim = len(first_valid_entry)
                else:
                    print(
                        f"    -> 警告: 列 '{vec_col_name}' 的第一个有效条目不是列表或数组: {type(first_valid_entry)}，跳过填充。")
                    continue

                zero_vector = np.zeros(vector_dim)

                def fill_na_with_zero_vector(x):
                    try:
                        if not isinstance(x, (list, np.ndarray)):
                            if pd.isna(x):
                                return zero_vector
                            else:
                                return x
                        x_array = np.asarray(x)
                        if x_array.size == 0:
                            return zero_vector
                        if x_array.dtype.kind in ['f', 'c']:
                            if np.all(np.isnan(x_array)):
                                return zero_vector
                        return x
                    except Exception as e:
                        print(f"      -> 处理值时出错: {e}, 使用零向量填充")
                        return zero_vector

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

            for col in post_emb_processing_names:
                if col in self.vector_expanded_feature_map:  # It's a vector column to expand
                    if len(X_imputed_df[col]) > 0:
                        sample_vector = None
                        for vec in X_imputed_df[col]:
                            if isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
                                sample_vector = vec
                                break
                        if sample_vector is not None:
                            expanded_df = pd.DataFrame(X_imputed_df[col].tolist(),
                                                       columns=self.vector_expanded_feature_map[col])
                            expanded_dfs.append(expanded_df)
                else:
                    # Regular numeric or non-vector column
                    other_cols.append(col)

            # Combine all parts back together
            final_parts = []
            if other_cols:
                final_parts.append(X_imputed_df[other_cols])
            if expanded_dfs:
                final_parts.extend(expanded_dfs)

            if final_parts:
                X_final_df = pd.concat(final_parts, axis=1)
                # Ensure consistent order by sorting column names alphabetically
                sorted_cols = sorted(X_final_df.columns)
                X_final_df = X_final_df.reindex(columns=sorted_cols)
            else:
                X_final_df = pd.DataFrame()  # Should rarely happen

            # Store the final processed feature names that will go into imputer/scaler etc.
            self.final_feature_names_after_processing = X_final_df.columns.tolist()
            print(f"Final pre-imputation feature count: {len(self.final_feature_names_after_processing)}")

            X_dense = X_final_df.values

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
        print(f"  -> SimpleImputer 拟合前特征数: {self.fit_feature_count_before_imputer_}")
        X_dense = self.imputer_num.fit_transform(X_dense)
        self.fit_feature_count_after_imputer_ = X_dense.shape[1]
        print(f"  -> SimpleImputer 拟合后特征数: {self.fit_feature_count_after_imputer_}")

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

        # 3. 高级特征工程 - Initialize modeling feature names with base ones
        self.feature_names_for_modeling = self.final_feature_names_after_processing.copy()
        print("步骤 3: 高级特征工程...")
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)

        print(f"特征工程后特征数: {X_dense.shape[1]}")

        # 4. 取消多重降维
        if self.use_multiple_decomposition:
            print("步骤 4: 多重降维... (已取消)")
        else:
            print("步骤 4: 多重降维已取消")

        # 5. 取消智能特征选择
        print("步骤 5: 智能特征选择... (已取消)")

        # 记录最终特征数
        self.final_feature_count_after_engineering_ = X_dense.shape[1]
        print(f"特征工程拟合完成！最终特征数: {self.final_feature_count_after_engineering_}")
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
            for original_col_name in self.final_feature_names_after_processing:
                # Check if it's an expanded dimension name
                parent_vec_found = False
                for vec_col, dims in self.vector_expanded_feature_map.items():
                    if original_col_name in dims:
                        parent_vec_found = True
                        # This is part of a vector expansion; handled later
                        break

                if parent_vec_found:
                    continue  # Will be added when its parent vector is found

                if original_col_name in current_post_process_names:
                    other_cols_from_fit.append(original_col_name)

            # Now handle vector expansions
            for vec_col_name, expected_dims in self.vector_expanded_feature_map.items():
                if vec_col_name in X_imputed_df.columns:
                    # Extract vectors and convert them correctly
                    raw_vectors = X_imputed_df[vec_col_name].tolist()

                    # Validate dimensions match expectation from fit
                    valid_vectors = []
                    for v in raw_vectors:
                        if isinstance(v, (list, np.ndarray)) and len(v) == len(expected_dims):
                            valid_vectors.append(list(v))
                        else:
                            # Use zero vector if invalid/mismatched
                            valid_vectors.append([0.0] * len(expected_dims))

                    expanded_df = pd.DataFrame(valid_vectors, columns=expected_dims)
                    expanded_dfs.append(expanded_df)

            # Combine everything again
            final_parts_transform = []
            if other_cols_from_fit:
                final_parts_transform.append(X_imputed_df[other_cols_from_fit])
            if expanded_dfs:
                final_parts_transform.extend(expanded_dfs)

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

            else:
                # Fallback: Create empty dataframe matching expected schema
                print("Warning: No features constructed in transform, creating dummy frame.")
                X_final_df_transform = pd.DataFrame(columns=self.final_feature_names_after_processing)
                # Fill with zeros if needed
                if len(self.final_feature_names_after_processing) > 0:
                    X_final_df_transform = X_final_df_transform.reindex(index=X.index).fillna(0.0)

            X_dense = X_final_df_transform.values

        elif sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Dimension check points remain largely unchanged since core issue fixed above
        print(f"  -> transform 阶段，准备进行 SimpleImputer transform 的特征数: {X_dense.shape[1]}")

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

        # Apply transformations step-by-step ensuring shapes align properly now
        X_dense = self.imputer_num.transform(X_dense)
        print(f"  -> transform 阶段，经过 SimpleImputer transform 后的特征数: {X_dense.shape[1]}")

        if self.power_transformer is not None:
            X_dense = self.power_transformer.transform(X_dense)
        X_dense = self.scaler.transform(X_dense)

        # Recreate engineered features just like in fit
        X_dense = self._create_statistical_features(X_dense)
        X_dense = self._create_cluster_features(X_dense)
        X_dense = self._create_anomaly_features(X_dense)

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


def find_threshold_for_max_recall(y_true, y_proba, min_precision=0.5):
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


def ensemble_predict_proba(models, X):
    """对多个模型的预测概率进行平均"""
    probas = []
    for model in models:
        proba = model.predict_proba(X)[:, 1]
        probas.append(proba)
    avg_proba = np.mean(probas, axis=0)
    return avg_proba


def train_and_evaluate_advanced(X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer_params=None):
    """使用先进特征工程的训练评估流程"""
    print("\n=== 构建先进特征工程管道 ===")

    # 特征工程
    feature_engineer = AdvancedFeatureEngineer(**feature_engineer_params)

    # XGBoost 模型 - 关注少数类
    xgb_classifier = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=1.0 / np.mean(y_train),  # 关注少数类
        verbosity=1
    )

    # CatBoost 模型 - 关注少数类
    cat_classifier = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.01,
        subsample=0.8,
        random_strength=0.8,
        random_seed=SEED,
        verbose=False,
        class_weights=[1.0, 1.0 / np.mean(y_train)],  # 关注少数类
        eval_metric='Logloss'
    )

    # 1. 拟合特征工程器
    print("\n=== 拟合特征工程器 ===")
    feature_engineer.fit(X_train, y_train)
    X_train_engineered = feature_engineer.transform(X_train)

    # 2. 模型训练
    print("\n=== 开始 XGBoost 模型训练 ===")
    start_time = time.time()
    xgb_classifier.fit(X_train_engineered, y_train)
    end_time = time.time()
    print(f"XGBoost 模型训练耗时: {end_time - start_time:.2f} 秒")

    print("\n=== 开始 CatBoost 模型训练 ===")
    start_time = time.time()
    cat_classifier.fit(X_train_engineered, y_train, eval_set=(feature_engineer.transform(X_val), y_val),
                       early_stopping_rounds=50)
    end_time = time.time()
    print(f"CatBoost 模型训练耗时: {end_time - start_time:.2f} 秒")

    # 3. 保存模型
    trained_pipeline = Pipeline([
        ('feature_engineer', feature_engineer),
        ('xgb_classifier', xgb_classifier),
        ('cat_classifier', cat_classifier)
    ])
    save_model(trained_pipeline, 'advanced_feature_engineering_ensemble_model.pkl')

    # 4. 验证集阈值选择
    print("\n=== 验证集阈值选择 ===")
    X_val_engineered = feature_engineer.transform(X_val)
    val_proba_xgb = xgb_classifier.predict_proba(X_val_engineered)[:, 1]
    val_proba_cat = cat_classifier.predict_proba(X_val_engineered)[:, 1]
    val_proba_ensemble = (val_proba_xgb + val_proba_cat) / 2.0

    threshold_f1 = find_best_f1_threshold(y_val, val_proba_ensemble)
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba_ensemble, min_precision=0.4)

    # 5. 测试集评估
    print("\n=== 测试集评估 ===")
    X_test_engineered = feature_engineer.transform(X_test)
    test_proba_xgb = xgb_classifier.predict_proba(X_test_engineered)[:, 1]
    test_proba_cat = cat_classifier.predict_proba(X_test_engineered)[:, 1]
    test_proba_ensemble = (test_proba_xgb + test_proba_cat) / 2.0

    # F1优化阈值评估
    print("\n--- 使用 F1 优化阈值评估 ---")
    y_pred_f1 = (test_proba_ensemble >= threshold_f1).astype(int)
    recall_f1 = recall_score(y_test, y_pred_f1)
    auc_f1 = roc_auc_score(y_test, test_proba_ensemble)
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
    y_pred_hr = (test_proba_ensemble >= threshold_high_recall).astype(int)
    recall_hr = recall_score(y_test, y_pred_hr)
    auc_hr = roc_auc_score(y_test, test_proba_ensemble)
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
    evaluate_and_plot(y_test, test_proba_ensemble, threshold_f1, threshold_high_recall,
                      model_name="Advanced Feature Engineering Ensemble (XGBoost & CatBoost)")

    return {
        'recall': recall_f1,
        'auc': auc_f1,
        'precision': precision_f1,
        'f1': f1_f1,
        'accuracy': accuracy_f1,
        'y_proba': test_proba_ensemble,
        'y_pred': y_pred_f1,
        'threshold': threshold_f1,
        'high_recall_results': {
            'recall': recall_hr,
            'precision': precision_hr,
            'f1': f1_hr,
            'threshold': threshold_high_recall
        },
        'trained_model': trained_pipeline,
        'individual_models': [xgb_classifier, cat_classifier]
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

        # 取消多重降维配置
        'use_multiple_decomposition': False,  # 关闭降维
        'svd_components': 80,  # 降维已取消
        'ica_components': 50,  # 降维已取消
        'fa_components': 30,  # 降维已取消

        # 取消特征选择配置
        'variance_threshold': 0.01,  # 特征选择已取消
        'univariate_k_best': None,  # 取消单变量选择
        'rf_n_features_to_select': None,  # 取消随机森林选择

        # 输出格式
        'force_sparse_output': False  # 使用密集矩阵，便于高级特征工程
    }

    # 数据加载
    print("\n=== 数据加载 ===")
    try:
        # 示例模拟一个小的数据集用于演示目的
        np.random.seed(42)
        num_samples = 1000
        num_regular_features = 10
        embed_dim = 5
        company_ids = np.arange(num_samples)
        targets = np.random.binomial(1, 0.1, size=num_samples)  # Make minority class rare

        regular_data = np.random.randn(num_samples, num_regular_features)
        industry_embeddings = np.random.randn(num_samples, embed_dim)
        sector_embeddings = np.random.randn(num_samples, embed_dim)

        df_dict = {'company_id': company_ids}
        for i in range(num_regular_features):
            df_dict[f'reg_feat_{i}'] = regular_data[:, i]

        for j in range(embed_dim):
            df_dict[f'industry_emb_{j}'] = industry_embeddings[:, j]
            df_dict[f'sector_emb_{j}'] = sector_embeddings[:, j]

        df_dict['target'] = targets

        data = pd.DataFrame(df_dict)
        #####

        # 实际读取方式如下所示，请替换为你自己的路径
        # import dask.dataframe as dd
        # ddf = dd.read_csv('train_bert_embedded.csv')
        # data = ddf.compute()

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
        X = X.iloc[mask]
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
