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
from catboost import CatBoostClassifier
# --- 导入稀疏矩阵支持 ---
from scipy import sparse
# --- 导入先进的特征工程库 ---
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # 导入 RandomForestClassifier
# --- 导入预处理 ---
from sklearn.impute import SimpleImputer
from sklearn.metrics import (recall_score, precision_score, roc_auc_score, f1_score, accuracy_score,
                             precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)
# --- 导入 XGBoost 和 CatBoost ---
from xgboost import XGBClassifier

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
    2. 使用随机森林进行降维（可选）
    3. 使用随机森林进行特征选择（可选，仅对非向量特征）
    """

    def __init__(self,
                 # 预处理参数
                 scaler_type='robust',  # 'standard', 'robust', 'quantile', 'power', 'minmax'
                 power_method='yeo-johnson',  # PowerTransformer method

                 # 降维参数 (使用随机森林)
                 use_rf_decomposition=False,  # 默认关闭降维
                 rf_n_components=50,  # 保留的特征数量
                 rf_random_state=SEED,

                 # 特征选择参数 (使用随机森林)
                 use_rf_selection=True,  # 启用随机森林特征选择
                 rf_n_features_to_select=100,  # 选择的特征数量，默认100
                 rf_selection_random_state=SEED,

                 # 输出格式
                 force_sparse_output=False):  # 改为False，因为高级特征工程通常产生密集特征

        # 存储参数
        self.scaler_type = scaler_type
        self.power_method = power_method
        self.use_rf_decomposition = use_rf_decomposition
        self.rf_n_components = rf_n_components
        self.rf_random_state = rf_random_state
        self.use_rf_selection = use_rf_selection
        self.rf_n_features_to_select = rf_n_features_to_select
        self.rf_selection_random_state = rf_selection_random_state
        self.force_sparse_output = force_sparse_output

        # 初始化组件
        self.imputer_num = None
        self.scaler = None
        self.power_transformer = None
        # 降维组件
        self.rf_regressor = None
        self.selected_rf_features_ = None
        # 特征选择组件
        self.rf_selector = None
        self.selected_features_indices_ = None  # 用于记录最终选择的特征索引

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
                            expanded_df = pd.DataFrame(validated_vectors, columns=expected_dims,
                                                       index=X_imputed_df.index)
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
        print("步骤 3: 高级特征工程... (无额外统计、聚类、异常检测步骤)")

        print(f"特征工程后特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

        # 4. 使用随机森林进行降维 (可选)
        if self.use_rf_decomposition and y is not None:
            print("步骤 4 (可选): 使用随机森林进行降维...")
            # 训练一个随机森林回归器来评估特征重要性
            # 使用 y 作为目标，因为它在 fit 阶段是可用的
            self.rf_regressor = RandomForestRegressor(
                n_estimators=100,
                max_features='sqrt',
                random_state=self.rf_random_state,
                n_jobs=-1
            )
            print(f"  -> 训练随机森林回归器以计算特征重要性...")

            # 关键修复：确保X_dense和y的行数一致
            if X_dense.shape[0] != len(y):
                print(f"  -> 警告：特征矩阵行数({X_dense.shape[0]})与标签向量长度({len(y)})不匹配")
                min_len = min(X_dense.shape[0], len(y))
                print(f"  -> 截断数据，保持行数一致: {min_len}")
                X_dense = X_dense[:min_len]
                y = y[:min_len]
                print(f"  -> 修复后 - 特征矩阵形状: {X_dense.shape}, 标签形状: {y.shape}")

            self.rf_regressor.fit(X_dense, y)

            # 获取特征重要性并排序
            importances = self.rf_regressor.feature_importances_
            indices = np.argsort(importances)[::-1]  # 降序排列

            # 选择前 n 个最重要的特征
            n_select = min(self.rf_n_components, len(importances))
            self.selected_rf_features_ = indices[:n_select]

            print(f"  -> 基于随机森林重要性选择 {n_select} 个特征...")
            X_dense = X_dense[:, self.selected_rf_features_]

            # 更新特征名
            selected_names = [self.feature_names_for_modeling[i] for i in self.selected_rf_features_]
            self.feature_names_for_modeling = selected_names

        else:
            print("步骤 4 (可选): 随机森林降维未启用或目标变量不可用")

        # 5. 使用随机森林进行特征选择 (可选)
        if self.use_rf_selection and y is not None:
            print("步骤 5 (可选): 使用随机森林进行特征选择...")

            # 识别非向量特征的索引
            non_vector_feature_indices = []
            non_vector_feature_names = []
            for i, name in enumerate(self.feature_names_for_modeling):
                # 检查是否是向量特征（以 _emb_vector 结尾）
                is_vector_feature = any(name.startswith(prefix) and name.endswith("_emb_vector")
                                        for prefix in self.vector_expanded_feature_map.keys())
                if not is_vector_feature:
                    non_vector_feature_indices.append(i)
                    non_vector_feature_names.append(name)

            if not non_vector_feature_indices:
                print("  -> 没有找到非向量特征，跳过特征选择。")
            else:
                print(f"  -> 识别出 {len(non_vector_feature_indices)} 个非向量特征用于选择。")

                # 提取非向量特征
                X_non_vector = X_dense[:, non_vector_feature_indices]

                # 训练随机森林分类器
                self.rf_selector = RandomForestClassifier(
                    n_estimators=100,
                    max_features='sqrt',
                    random_state=self.rf_selection_random_state,
                    n_jobs=-1,
                    class_weight='balanced'  # 处理不平衡数据
                )
                print(f"  -> 训练随机森林分类器以计算特征重要性...")

                # 再次检查X_non_vector和y的行数
                if X_non_vector.shape[0] != len(y):
                    print(f"  -> 警告：非向量特征矩阵行数({X_non_vector.shape[0]})与标签向量长度({len(y)})不匹配")
                    min_len = min(X_non_vector.shape[0], len(y))
                    print(f"  -> 截断数据，保持行数一致: {min_len}")
                    X_non_vector_selected = X_non_vector[:min_len]
                    y_selected = y[:min_len]
                else:
                    X_non_vector_selected = X_non_vector
                    y_selected = y

                self.rf_selector.fit(X_non_vector_selected, y_selected)

                # 获取特征重要性并排序
                importances = self.rf_selector.feature_importances_
                indices = np.argsort(importances)[::-1]  # 降序排列

                # 选择前 n 个最重要的特征 (非向量)
                n_select = min(self.rf_n_features_to_select, len(importances))
                selected_non_vector_indices_local = indices[:n_select]  # 本地索引

                # 转换为全局特征索引
                selected_non_vector_indices_global = [non_vector_feature_indices[i] for i in
                                                      selected_non_vector_indices_local]

                # 识别向量特征的全局索引
                vector_feature_indices_global = [i for i, name in enumerate(self.feature_names_for_modeling)
                                                 if any(name.startswith(prefix) and name.endswith("_emb_vector")
                                                        for prefix in self.vector_expanded_feature_map.keys())]

                # 合并最终选择的特征索引 (向量特征 + 选择的非向量特征)，并排序以保持顺序
                self.selected_features_indices_ = sorted(
                    vector_feature_indices_global + selected_non_vector_indices_global)

                print(f"  -> 保留所有 {len(vector_feature_indices_global)} 个向量特征。")
                print(f"  -> 基于随机森林重要性选择了 {len(selected_non_vector_indices_global)} 个非向量特征。")
                print(f"  -> 最终特征总数: {len(self.selected_features_indices_)}")

                # 应用特征选择
                X_dense = X_dense[:, self.selected_features_indices_]

                # 更新特征名
                selected_names = [self.feature_names_for_modeling[i] for i in self.selected_features_indices_]
                self.feature_names_for_modeling = selected_names

        else:
            print("步骤 5 (可选): 随机森林特征选择未启用或目标变量不可用")

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
                        expanded_df = pd.DataFrame(validated_vectors, columns=expected_dims, index=X_imputed_df.index)
                        expanded_dfs.append(expanded_df)
                        print(f"      -> Transform: 成功展开 '{vec_col_name}' 为 {len(expected_dims)} 个特征。")
                    except Exception as e:
                        print(f"      -> Transform: 错误: 无法从列 '{vec_col_name}' 创建DataFrame ({e})，添加零列。")
                        # Add zero columns as fallback
                        zero_df = pd.DataFrame(np.zeros((len(X), expected_dim_count)), columns=expected_dims,
                                               index=X.index)
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

        # Skip statistical, cluster, anomaly features as per config
        print("  -> transform 阶段，高级特征工程... (无额外统计、聚类、异常检测步骤)")

        # Apply random forest dimensionality reduction if fitted
        if self.use_rf_decomposition and self.selected_rf_features_ is not None:
            print("  -> transform 阶段，应用随机森林降维...")
            X_dense = X_dense[:, self.selected_rf_features_]
        else:
            print("  -> transform 阶段，随机森林降维未应用")

        # Apply random forest feature selection if fitted
        if self.use_rf_selection and self.selected_features_indices_ is not None:
            print("  -> transform 阶段，应用随机森林特征选择...")
            X_dense = X_dense[:, self.selected_features_indices_]
        else:
            print("  -> transform 阶段，随机森林特征选择未应用")

        print(f"  -> transform 阶段，特征工程后特征数: {X_dense.shape[1]} (形状: {X_dense.shape})")

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


# --- 其他函数保持不变 ---

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
        n_estimators=1000,
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
        iterations=1000,
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
    print(f"  -> 训练集特征工程后形状: {X_train_engineered.shape}")

    # 2. 模型训练
    print("\n=== 开始 XGBoost 模型训练 ===")
    print(f"  -> XGBoost 输入特征矩阵形状: {X_train_engineered.shape}, 标签形状: {y_train.shape}")

    # 关键修复：检查特征矩阵行数与标签向量长度是否匹配
    if X_train_engineered.shape[0] == 0:
        raise ValueError("致命错误：XGBoost训练数据特征矩阵行数为0！")

    # 新添加：检查特征矩阵行数与标签向量长度是否一致
    if X_train_engineered.shape[0] != len(y_train):
        print(f"警告：特征矩阵行数({X_train_engineered.shape[0]})与标签向量长度({len(y_train)})不匹配")
        # 尝试修复：截断较长的一方以匹配较短的一方
        min_len = min(X_train_engineered.shape[0], len(y_train))
        print(f"  -> 截断数据，保持行数一致: {min_len}")
        X_train_engineered = X_train_engineered[:min_len]
        y_train = y_train[:min_len]
        print(f"  -> 修复后 - 特征矩阵形状: {X_train_engineered.shape}, 标签形状: {y_train.shape}")

    start_time = time.time()
    xgb_classifier.fit(X_train_engineered, y_train)
    end_time = time.time()
    print(f"XGBoost 模型训练耗时: {end_time - start_time:.2f} 秒")

    print("\n=== 开始 CatBoost 模型训练 ===")
    print(f"  -> CatBoost 输入特征矩阵形状: {X_train_engineered.shape}, 标签形状: {y_train.shape}")
    start_time = time.time()

    # 对验证集也进行同样的检查和修复
    X_val_engineered_temp = feature_engineer.transform(X_val)
    if X_val_engineered_temp.shape[0] != len(y_val):
        print(f"警告：验证集特征矩阵行数({X_val_engineered_temp.shape[0]})与标签向量长度({len(y_val)})不匹配")
        min_len_val = min(X_val_engineered_temp.shape[0], len(y_val))
        print(f"  -> 截断验证集数据，保持行数一致: {min_len_val}")
        X_val_engineered_temp = X_val_engineered_temp[:min_len_val]
        y_val_temp = y_val[:min_len_val]
        print(f"  -> 修复后 - 验证集特征矩阵形状: {X_val_engineered_temp.shape}, 标签形状: {y_val_temp.shape}")
    else:
        y_val_temp = y_val

    cat_classifier.fit(X_train_engineered, y_train, eval_set=(X_val_engineered_temp, y_val_temp),
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
    # 使用之前修复过的验证集特征工程数据，而不是重新生成
    X_val_engineered = X_val_engineered_temp
    y_val_used = y_val_temp
    print(f"  -> 验证集特征工程后形状: {X_val_engineered.shape}")

    # 对验证集也进行同样的检查和修复（如果需要）
    if X_val_engineered.shape[0] != len(y_val_used):
        print(f"警告：验证集特征矩阵行数({X_val_engineered.shape[0]})与标签向量长度({len(y_val_used)})不匹配")
        min_len_val = min(X_val_engineered.shape[0], len(y_val_used))
        print(f"  -> 截断验证集数据，保持行数一致: {min_len_val}")
        X_val_engineered = X_val_engineered[:min_len_val]
        y_val_used = y_val_used[:min_len_val]
        print(f"  -> 修复后 - 验证集特征矩阵形状: {X_val_engineered.shape}, 标签形状: {y_val_used.shape}")

    val_proba_xgb = xgb_classifier.predict_proba(X_val_engineered)[:, 1]
    val_proba_cat = cat_classifier.predict_proba(X_val_engineered)[:, 1]
    val_proba_ensemble = (val_proba_xgb + val_proba_cat) / 2.0

    threshold_f1 = find_best_f1_threshold(y_val_used, val_proba_ensemble)
    threshold_high_recall = find_threshold_for_max_recall(y_val_used, val_proba_ensemble, min_precision=0.4)

    # 5. 测试集评估
    print("\n=== 测试集评估 ===")
    X_test_engineered = feature_engineer.transform(X_test)
    print(f"  -> 测试集特征工程后形状: {X_test_engineered.shape}")

    # 对测试集也进行同样的检查和修复
    if X_test_engineered.shape[0] != len(y_test):
        print(f"警告：测试集特征矩阵行数({X_test_engineered.shape[0]})与标签向量长度({len(y_test)})不匹配")
        min_len_test = min(X_test_engineered.shape[0], len(y_test))
        print(f"  -> 截断测试集数据，保持行数一致: {min_len_test}")
        X_test_engineered = X_test_engineered[:min_len_test]
        y_test_temp = y_test[:min_len_test]
        print(f"  -> 修复后 - 测试集特征矩阵形状: {X_test_engineered.shape}, 标签形状: {y_test_temp.shape}")
    else:
        y_test_temp = y_test

    test_proba_xgb = xgb_classifier.predict_proba(X_test_engineered)[:, 1]
    test_proba_cat = cat_classifier.predict_proba(X_test_engineered)[:, 1]
    test_proba_ensemble = (test_proba_xgb + test_proba_cat) / 2.0

    # F1优化阈值评估
    print("\n--- 使用 F1 优化阈值评估 ---")
    y_pred_f1 = (test_proba_ensemble >= threshold_f1).astype(int)
    recall_f1 = recall_score(y_test_temp, y_pred_f1)
    auc_f1 = roc_auc_score(y_test_temp, test_proba_ensemble)
    precision_f1 = precision_score(y_test_temp, y_pred_f1)
    f1_f1 = f1_score(y_test_temp, y_pred_f1)
    accuracy_f1 = accuracy_score(y_test_temp, y_pred_f1)
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
    recall_hr = recall_score(y_test_temp, y_pred_hr)
    auc_hr = roc_auc_score(y_test_temp, test_proba_ensemble)
    precision_hr = precision_score(y_test_temp, y_pred_hr)
    f1_hr = f1_score(y_test_temp, y_pred_hr)
    accuracy_hr = accuracy_score(y_test_temp, y_pred_hr)
    print(f"阈值: {threshold_high_recall:.4f}")
    print(f"Recall: {recall_hr:.4f}")
    print(f"AUC: {auc_hr:.4f}")
    print(f"Precision: {precision_hr:.4f}")
    print(f"F1 Score: {f1_hr:.4f}")
    print(f"Accuracy: {accuracy_hr:.4f}")
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision_hr + 50 * auc_hr + 30 * recall_hr:.4f}")

    # 可视化评估结果
    print("\n=== 绘制评估图表 ===")
    evaluate_and_plot(y_test_temp, test_proba_ensemble, threshold_f1, threshold_high_recall,
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

        # 降维配置 (使用随机森林) - 已关闭
        'use_rf_decomposition': False,  # 关闭降维
        'rf_n_components': 50,
        'rf_random_state': SEED,

        # 特征选择配置 (使用随机森林) - 已启用
        'use_rf_selection': True,  # 启用特征选择
        'rf_n_features_to_select': 150,  # 选择150个非向量特征
        'rf_selection_random_state': SEED,

        # 输出格式
        'force_sparse_output': False  # 使用密集矩阵，便于高级特征工程
    }

    # 数据加载
    print("\n=== 数据加载 ===")
    try:
        # 实际读取方式如下所示，请替换为你自己的路径
        data = pd.read_parquet('train_bert_enhanced_embedded_hier.parquet')


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
    try:
        results = train_and_evaluate_advanced(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_engineer_params=feature_engineer_params
        )
    except Exception as e:
        print(f"训练流程发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

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
