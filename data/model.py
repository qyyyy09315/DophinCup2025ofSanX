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
from imblearn.ensemble import BalancedRandomForestClassifier  # 用于特征选择
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
# --- 导入 TruncatedSVD 作为 PCA 的替代方案 (可选但推荐) ---
from sklearn.decomposition import TruncatedSVD
# --- 导入随机森林和特征选择器 ---
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# --- 导入 Pipeline ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures
# --- 导入 XGBoost 和 平衡随机森林 ---
from xgboost import XGBClassifier

# --- 导入稀疏矩阵支持 ---
from scipy import sparse

warnings.filterwarnings('ignore')

# 环境配置 - 最大化CPU利用率
NUM_CORES = multiprocessing.cpu_count()
print(f"检测到 {NUM_CORES} 个CPU核心")
os.environ["LOKY_MAX_CPU_COUNT"] = str(NUM_CORES)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# --- 固定参数的 SMOTETomek ---
def moo_smotetomek_func(X, y):
    """
    使用一组固定的参数应用 SMOTETomek 进行重采样。
    注意：imblearn 通常需要密集矩阵输入。
    """
    print("  -> 应用固定参数的 SMOTETomek...")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTETomek.")
        return X, y

    try:
        # 确保输入是密集矩阵，因为 SMOTETomek 可能不直接支持稀疏矩阵
        if sparse.issparse(X):
            print("    -> 将稀疏矩阵转换为密集矩阵以进行 SMOTETomek...")
            X_dense = X.toarray()
        else:
            X_dense = X

        # 使用固定的参数组合
        smotetomek = SMOTETomek(
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=SEED, n_jobs=-1),
            tomek=TomekLinks(sampling_strategy='majority', n_jobs=-1),
            random_state=SEED
        )

        X_resampled_dense, y_resampled = smotetomek.fit_resample(X_dense, y)

        # 如果原始输入是稀疏的，考虑是否将结果转回稀疏（但重采样后可能不稀疏）
        # 这里我们保持为密集矩阵，因为它可能已经不稀疏了
        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled_dense.shape[0]}")
        return X_resampled_dense, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y


# --- 更新 AdvancedPreprocessor 类 (调整步骤顺序并支持稀疏矩阵) ---
class AdvancedPreprocessor:
    """高级数据预处理管道，包含清洗、标准化、特征工程（分箱、交叉）、随机森林特征选择和降维 (支持稀疏矩阵)"""

    def __init__(self, n_bins=None, cross_degree=None, rf_n_features_to_select=None, use_truncated_svd=False,
                 svd_n_components=None, force_sparse_output=True):
        """
        初始化预处理器。
        :param n_bins: int or None, 分箱的数量。如果为 None，则不进行分箱。
        :param cross_degree: int or None, 特征交叉的度数。如果为 None，则不进行特征交叉。
        :param rf_n_features_to_select: int or float or None, 使用随机森林选择的特征数量。
                                        如果是 int，则选择该数量的特征。
                                        如果是 float (0 < x < 1)，则选择该比例的特征。
                                        如果为 None，则跳过随机森林特征选择。
        :param use_truncated_svd: bool, 是否使用 TruncatedSVD 替代 PCA。
        :param svd_n_components: int or float or None, TruncatedSVD 的 n_components。
                                 如果是 int，则保留该数量的成分。
                                 如果是 float (0 < x < 1)，则保留该比例的方差。
                                 如果为 None 且 use_truncated_svd=True，则默认为 100。
        :param force_sparse_output: bool, 是否强制输出为稀疏矩阵 (csr_matrix)。
        """
        self.imputer_num = None
        self.scaler = None
        self.winsorizer = None
        self.binner = None
        self.crosser = None
        self.rf_selector = None  # 新增：随机森林选择器
        self.svd = None  # 新增：TruncatedSVD (替代PCA)
        self.numerical_features = None
        self.n_bins = n_bins
        self.cross_degree = cross_degree
        self.rf_n_features_to_select = rf_n_features_to_select
        self.use_truncated_svd = use_truncated_svd
        self.svd_n_components = svd_n_components
        self.force_sparse_output = force_sparse_output  # 控制输出格式
        self.original_feature_count = None  # 记录原始特征数

    def fit(self, X, y=None):
        """拟合预处理器"""
        # 假设 X 是一个 DataFrame，或者需要提供列名列表 self.numerical_features
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        elif self.numerical_features is None:
            # 如果 X 是 numpy array 且未提供列名，则假设所有列都是数值型
            self.numerical_features = list(range(X.shape[1]))
            print("警告: 未提供列名，假设所有特征均为数值型。")

        self.original_feature_count = X.shape[1]
        print(f"原始特征数: {self.original_feature_count}")

        print("步骤 1/4: 缺失值处理...")
        # 1. 缺失值处理 - 使用中位数填充数值型特征
        self.imputer_num = SimpleImputer(strategy='median')
        X_imputed = X.copy()

        # 检查 numerical_features 是否在有效范围内
        valid_numerical_features = [f for f in self.numerical_features if f < self.original_feature_count]
        if len(valid_numerical_features) != len(self.numerical_features):
            print(f"警告: 发现无效的 numerical_features 索引，已过滤。有效特征数: {len(valid_numerical_features)}")
            self.numerical_features = valid_numerical_features

        if isinstance(X, np.ndarray) or sparse.issparse(X):
            # 对于 numpy array 或稀疏矩阵，需要处理列索引
            if sparse.issparse(X):
                # 稀疏矩阵需要先转换为密集来填充，然后可能再转回稀疏
                # 或者使用稀疏友好的 imputer (如 sklearn 0.24+ 的 SimpleImputer)
                # 这里我们简单处理：转换 -> 填充 -> 转回
                X_dense_for_impute = X.toarray()
                X_dense_for_impute[:, self.numerical_features] = self.imputer_num.fit_transform(
                    X_dense_for_impute[:, self.numerical_features])
                X_imputed = sparse.csr_matrix(X_dense_for_impute)
            else:  # numpy array
                X_imputed[:, self.numerical_features] = self.imputer_num.fit_transform(X[:, self.numerical_features])
        else:  # DataFrame
            X_imputed.loc[:, self.numerical_features] = self.imputer_num.fit_transform(X[self.numerical_features])

        print("步骤 2/4: 异常值处理 (Winsorization)...")
        # 2. 异常值处理 - Winsorization
        # QuantileTransformer 可以处理稀疏矩阵 (如果输入是稀疏的，输出通常也是稀疏的)
        self.winsorizer = QuantileTransformer(output_distribution='uniform', random_state=SEED,
                                              n_quantiles=min(1000, X_imputed.shape[0]))
        X_winsorized = X_imputed.copy()
        if isinstance(X_imputed, np.ndarray):
            X_winsorized[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[:, self.numerical_features])
        elif sparse.issparse(X_imputed):
            # QuantileTransformer 可以直接处理稀疏矩阵
            # 但通常要求输入是密集的或特定格式，这里我们转换
            X_winsorized_dense = X_imputed.toarray()
            X_winsorized_dense[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_winsorized_dense[:, self.numerical_features])
            X_winsorized = sparse.csr_matrix(X_winsorized_dense)
        else:  # DataFrame
            X_winsorized.loc[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[self.numerical_features])

        print("步骤 3/4: 特征转换与标准化...")
        # 3. 特征转换与标准化
        # StandardScaler 可以处理稀疏矩阵 (输出也是稀疏的)
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False 对稀疏矩阵是必需的
        X_scaled = self.scaler.fit_transform(X_winsorized)
        # 确保标准化后的数据是稀疏的 (如果输入是稀疏的)
        if not sparse.issparse(X_scaled) and (sparse.issparse(X_winsorized) or self.force_sparse_output):
            X_scaled = sparse.csr_matrix(X_scaled)
            print("    -> 标准化后数据已转换为稀疏矩阵 (csr_matrix)。")

        # --- 新增步骤: 特征分箱 ---
        if self.n_bins is not None and self.n_bins > 1:
            print(f"步骤 3.1: 特征分箱 (n_bins={self.n_bins})...")
            # KBinsDiscretizer 可以处理稀疏矩阵 (输出是稀疏的)
            # 使用 'quantile' 策略以获得更平衡的箱
            self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='quantile')
            # 注意: encode='onehot-dense' 会输出密集矩阵。如果希望保持稀疏，可以使用 'onehot' (但需要 sklearn 0.24+)
            # 这里为了兼容性暂时使用 'onehot-dense'，然后强制转回稀疏
            # --- 关键修改点：确保输入到 fit_transform 的是密集数组 ---
            if sparse.issparse(X_scaled):
                print("    -> 将稀疏矩阵 X_scaled 转换为密集数组以供 KBinsDiscretizer 使用...")
                X_scaled_dense = X_scaled.toarray()
            else:
                X_scaled_dense = X_scaled

            X_binned_dense = self.binner.fit_transform(X_scaled_dense)  # 输入是密集的 X_scaled_dense
            X_binned = sparse.csr_matrix(X_binned_dense)
            print(f"  -> 分箱后特征数: {X_binned.shape[1]}")
        else:
            X_binned = X_scaled  # 注意输入是 X_scaled
            print("  -> 跳过分箱步骤。")

        # --- 新增步骤: 特征交叉 ---
        if self.cross_degree is not None and self.cross_degree >= 2:
            print(f"步骤 3.2: 特征交叉 (degree={self.cross_degree})...")
            # PolynomialFeatures 可以处理稀疏矩阵 (输出是稀疏的)
            # 限制交互项数量，degree=2 通常足够
            self.crosser = PolynomialFeatures(degree=self.cross_degree, interaction_only=True, include_bias=False)
            X_crossed = self.crosser.fit_transform(X_binned)  # 注意输入是 X_binned
            # 确保交叉后的数据是稀疏的
            if not sparse.issparse(X_crossed):
                X_crossed = sparse.csr_matrix(X_crossed)
            print(f"  -> 交叉后特征数: {X_crossed.shape[1]}")
        else:
            X_crossed = X_binned  # 注意输入是 X_binned
            print("  -> 跳过特征交叉步骤。")

        # --- 新增步骤: 随机森林特征选择 (调整到交叉之后) ---
        if self.rf_n_features_to_select is not None:
            print(f"步骤 3.3: 随机森林特征选择 (目标特征数: {self.rf_n_features_to_select})...")
            # 使用一个相对快速的随机森林来评估特征重要性
            # 注意：输入是 X_crossed (已分箱和交叉)
            if isinstance(self.rf_n_features_to_select, int) and self.rf_n_features_to_select > X_crossed.shape[1]:
                print(
                    f"  -> 警告: 请求的特征数 ({self.rf_n_features_to_select}) 超过可用特征数 ({X_crossed.shape[1]})。将选择所有可用特征。")
                n_features_to_select = X_crossed.shape[1]
            elif isinstance(self.rf_n_features_to_select, float) and 0 < self.rf_n_features_to_select <= 1:
                n_features_to_select = int(self.rf_n_features_to_select * X_crossed.shape[1])
                print(f"  -> 根据比例 {self.rf_n_features_to_select}，将选择约 {n_features_to_select} 个特征。")
            else:
                n_features_to_select = self.rf_n_features_to_select

            # 使用一个快速的随机森林模型来评估特征重要性
            # --- 修改点: 使用 BalancedRandomForestClassifier ---
            temp_rf = BalancedRandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=SEED)
            # threshold=-np.inf 确保选出 max_features 个特征，即使重要性很低
            self.rf_selector = SelectFromModel(temp_rf, max_features=n_features_to_select, threshold=-np.inf)

            # SelectFromModel 可以处理稀疏矩阵
            X_rf_selected = self.rf_selector.fit_transform(X_crossed, y)  # 输入是 X_crossed
            # 确保选择后的数据是稀疏的
            if not sparse.issparse(X_rf_selected):
                X_rf_selected = sparse.csr_matrix(X_rf_selected)
            print(f"  -> 随机森林选择后特征数: {X_rf_selected.shape[1]}")
        else:
            X_rf_selected = X_crossed  # 输入是 X_crossed
            print("  -> 跳过随机森林特征选择步骤。")

        # --- 修改点: 使用 TruncatedSVD 降维 (输入是 X_rf_selected) ---
        if self.use_truncated_svd:
            print("步骤 4/4: TruncatedSVD 降维...")
            n_components_svd = self.svd_n_components if self.svd_n_components is not None else min(100,
                                                                                                   X_rf_selected.shape[
                                                                                                       1] - 1)
            # TruncatedSVD 可以直接处理稀疏矩阵
            self.svd = TruncatedSVD(n_components=n_components_svd, random_state=SEED)
            self.svd.fit(X_rf_selected)  # 输入是 X_rf_selected
            print(f"TruncatedSVD 完成。保留特征数: {self.svd.n_components}")
        else:
            print("步骤 4/4: 跳过降维步骤 (未启用 TruncatedSVD)。")

        # --- ---

        print(f"预处理完成。")
        return self

    def transform(self, X):
        """应用预处理步骤"""
        # 处理输入类型
        if isinstance(X, pd.DataFrame):
            X_values = X.values  # 转换为 numpy array 以便处理
            if self.force_sparse_output:
                X_values = sparse.csr_matrix(X_values)
        elif sparse.issparse(X):
            X_values = X
        else:  # numpy array
            X_values = X
            if self.force_sparse_output:
                X_values = sparse.csr_matrix(X_values)

        # 1. 缺失值处理
        if sparse.issparse(X_values):
            X_dense_for_impute = X_values.toarray()
            X_dense_for_impute[:, self.numerical_features] = self.imputer_num.transform(
                X_dense_for_impute[:, self.numerical_features])
            X_imputed = sparse.csr_matrix(X_dense_for_impute)
        else:  # numpy array or dense
            X_imputed = X_values.copy()
            X_imputed[:, self.numerical_features] = self.imputer_num.transform(X_values[:, self.numerical_features])

        # 2. 异常值处理
        if sparse.issparse(X_imputed):
            X_winsorized_dense = X_imputed.toarray()
            X_winsorized_dense[:, self.numerical_features] = self.winsorizer.transform(
                X_winsorized_dense[:, self.numerical_features])
            X_winsorized = sparse.csr_matrix(X_winsorized_dense)
        else:  # numpy array or dense
            X_winsorized = X_imputed.copy()
            X_winsorized[:, self.numerical_features] = self.winsorizer.transform(X_imputed[:, self.numerical_features])

        # 3. 标准化
        X_scaled = self.scaler.transform(X_winsorized)
        if not sparse.issparse(X_scaled) and (sparse.issparse(X_winsorized) or self.force_sparse_output):
            X_scaled = sparse.csr_matrix(X_scaled)

        # 3.1. 特征分箱 (如果已拟合)
        if self.binner is not None:
            # --- 关键修改点：确保输入到 transform 的是密集数组 ---
            if sparse.issparse(X_scaled):
                X_scaled_dense = X_scaled.toarray()
            else:
                X_scaled_dense = X_scaled
            X_binned_dense = self.binner.transform(X_scaled_dense)  # 输入是密集的 X_scaled_dense
            X_binned = sparse.csr_matrix(X_binned_dense)
        else:
            X_binned = X_scaled  # 输入是 X_scaled

        # 3.2. 特征交叉 (如果已拟合)
        if self.crosser is not None:
            X_crossed = self.crosser.transform(X_binned)  # 输入是 X_binned
            if not sparse.issparse(X_crossed):
                X_crossed = sparse.csr_matrix(X_crossed)
        else:
            X_crossed = X_binned  # 输入是 X_binned

        # 3.3. 随机森林特征选择 (如果已拟合)
        if self.rf_selector is not None:
            X_rf_selected = self.rf_selector.transform(X_crossed)  # 输入是 X_crossed
            if not sparse.issparse(X_rf_selected):
                X_rf_selected = sparse.csr_matrix(X_rf_selected)
        else:
            X_rf_selected = X_crossed  # 输入是 X_crossed

        # 4. TruncatedSVD 降维
        if self.use_truncated_svd and self.svd is not None:
            X_reduced = self.svd.transform(X_rf_selected)  # 输入是 X_rf_selected
            # SVD 输出通常是密集的，除非强制稀疏
            if self.force_sparse_output:
                X_reduced = sparse.csr_matrix(X_reduced)
        else:
            X_reduced = X_rf_selected  # 如果没有降维步骤

        # 最终输出格式控制
        if self.force_sparse_output and not sparse.issparse(X_reduced):
            X_reduced = sparse.csr_matrix(X_reduced)
        elif not self.force_sparse_output and sparse.issparse(X_reduced):
            X_reduced = X_reduced.toarray()

        return X_reduced

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)


# --- ---
# --- 特征选择器类 (修改为使用 BalancedRandomForestClassifier) ---
class GBDT_RFE_FeatureSelector:  # 类名保持不变，但功能已修改
    """基于平衡随机森林的RFE特征选择器 (支持稀疏矩阵)"""

    def __init__(self, n_features=None, step=0.1):
        self.n_features = n_features
        self.step = step
        self.selector = None
        self.feature_mask_ = None

    def fit(self, X, y):
        # 特征选择使用 BalancedRandomForestClassifier
        # --- 修改点: 使用 BalancedRandomForestClassifier ---
        brf = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            # learning_rate 参数不适用于 BalancedRandomForestClassifier，已移除
            random_state=SEED
        )

        if self.n_features is None:
            self.n_features = max(1, X.shape[1] // 2)  # 确保至少选择1个特征

        self.selector = RFE(
            estimator=brf,
            n_features_to_select=self.n_features,
            step=self.step,
            verbose=1
        )
        self.selector.fit(X, y)
        self.feature_mask_ = self.selector.support_
        return self

    def transform(self, X):
        return self.selector.transform(X)


# --- 阈值选择辅助函数 ---
def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 {filename}")


def find_best_f1_threshold(y_true, y_proba):
    """
    在验证集上寻找最佳阈值以优化 F1 分数。
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # 计算 F1 分数
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
    """
    在验证集上寻找能获得最高召回率且精确率不低于 min_precision 的阈值。
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # 寻找满足最小精确率要求的索引
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid_indices) == 0:
        print(f"警告: 没有阈值能满足精确率 >= {min_precision}。返回默认阈值 0.5。")
        return 0.5

    # 在满足条件的索引中找到最大召回率
    best_valid_idx = valid_indices[np.argmax(recalls[valid_indices])]
    best_threshold = thresholds[best_valid_idx]
    best_precision = precisions[best_valid_idx]
    best_recall = recalls[best_valid_idx]

    print(f"高召回率阈值 (Precision >= {min_precision}): {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    return best_threshold


# --- 新增：评估与可视化函数 ---
def evaluate_and_plot(y_true, y_proba, threshold_f1, threshold_hr, model_name="Model"):
    """
    评估模型并绘制 ROC, PR, Confusion Matrix
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. ROC Curve ---
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

    # --- 2. Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)  # AUC of PR curve
    axes[1].plot(recall, precision, color='b', lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")

    # --- 3. Confusion Matrix (使用 F1 优化阈值) ---
    y_pred_f1 = (y_proba >= threshold_f1).astype(int)
    cm = confusion_matrix(y_true, y_pred_f1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title(f'Confusion Matrix (Threshold={threshold_f1:.4f})')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # --- 打印详细报告 ---
    print(f"\n=== {model_name} 详细评估报告 ===")
    print(classification_report(y_true, y_pred_f1, target_names=['Negative', 'Positive']))
    print("-" * 40)


def train_and_evaluate_without_search(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor_params=None):
    """取消超参数搜索的训练评估流程 (XGBoost 版本, 支持稀疏矩阵)"""
    # --- 构建 Pipeline (注意: SMOTETomek 通常在Pipeline外处理) ---
    print("\n=== 构建 Scikit-learn Pipeline ===")
    # Pipeline 主要用于预处理和最终分类器
    # 注意：AdvancedPreprocessor 现在可以处理稀疏矩阵
    preprocessor = AdvancedPreprocessor(**preprocessor_params)
    classifier = XGBClassifier(random_state=SEED, n_jobs=-1)  # XGBoost 原生支持稀疏输入

    # --- 1. 预处理训练数据 (仅拟合一次!) ---
    print("\n=== 拟合预处理器 (基于原始训练数据) ===")
    preprocessor.fit(X_train, y_train)
    X_train_preprocessed = preprocessor.transform(X_train)

    # --- 2. 应用 SMOTETomek 重采样 (在预处理后的数据上) ---
    print("\n=== 应用 SMOTETomek 重采样 ===")
    # 应用 SMOTETomek (需要密集矩阵)
    X_train_resampled, y_train_resampled = moo_smotetomek_func(X_train_preprocessed, y_train)

    # --- 3. 模型训练 (在重采样后的数据上) ---
    print("\n=== 开始模型训练 ===")
    start_time = time.time()

    # 直接使用重采样后的数据训练分类器
    # 不需要重新拟合 preprocessor，因为它已经基于原始训练数据拟合好了
    classifier.fit(X_train_resampled, y_train_resampled)

    end_time = time.time()
    print(f"\n模型训练耗时: {end_time - start_time:.2f} 秒")

    # --- 保存训练好的模型 ---
    # 保存整个Pipeline或组件
    trained_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    save_model(trained_pipeline, 'trained_model_without_search_xgb_sparse.pkl')

    # --- 在验证集上寻找最佳阈值 ---
    print("\n=== 验证集阈值选择 ===")
    # 需要先对验证集进行预处理 (使用最初拟合的 preprocessor)
    X_val_processed = preprocessor.transform(X_val) # <-- 这里不会再报错，因为 preprocessor 是基于原始维度拟合的
    val_proba = classifier.predict_proba(X_val_processed)[:, 1]

    # 1. 寻找最佳 F1 阈值
    print("\n[阈值选择 1] 优化 F1 分数...")
    threshold_f1 = find_best_f1_threshold(y_val, val_proba)

    # 2. 寻找高召回率阈值 (例如，要求 Precision >= 0.4)
    print("\n[阈值选择 2] 优化召回率 (要求 Precision >= 0.4)...")
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # --- 测试集评估 ---
    print("\n=== 测试集评估 ===")
    # 需要先对测试集进行预处理 (使用最初拟合的 preprocessor)
    X_test_processed = preprocessor.transform(X_test) # <-- 这里也不会再报错
    test_proba = classifier.predict_proba(X_test_processed)[:, 1]

    # 使用 F1 优化的阈值评估
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

    # 使用高召回率阈值评估
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

    # --- 可视化评估结果 ---
    print("\n=== 绘制评估图表 ===")
    evaluate_and_plot(y_test, test_proba, threshold_f1, threshold_high_recall, model_name="XGBoost (No Search, Sparse)")

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
        'trained_model': trained_pipeline  # 返回训练好的Pipeline
    }



def main():
    # 检查CUDA可用性 (PyTorch)
    print("CUDA 可用性 (PyTorch):", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # --- 配置预处理器参数 ---
    # 建议: 使用稀疏友好的设置
    preprocessor_params = {
        'n_bins': 10,
        'cross_degree': 2,
        'rf_n_features_to_select': 1000,  # 可尝试更小值如 500
        'use_truncated_svd': True,  # 强烈推荐：使用 TruncatedSVD 替代 PCA
        'svd_n_components': 50,  # TruncatedSVD保留的成分
        'force_sparse_output': True  # 强制输出为稀疏矩阵
    }

    # 加载数据 (尝试加载为稀疏矩阵)
    print("\n=== 数据加载 ===")
    try:
        # 尝试以稀疏格式读取 (如果原始CSV是稀疏的，这可能有效)
        # 注意：pandas read_csv 的 sparse=True 可能不总是有效，取决于数据
        data = pd.read_csv("clean.csv", low_memory=False)  # 通常不直接支持 sparse=True 有效加载
        if "company_id" in data.columns:
            data = data.drop(columns=["company_id"])
        if "target" not in data.columns:
            raise ValueError("数据中必须包含'target'列")

        # 分离特征和目标
        y = data["target"].values.astype(int)
        feature_data = data.drop(columns=["target"])

        # 尝试转换为稀疏矩阵
        print("尝试将特征数据转换为稀疏矩阵...")
        X_dense = feature_data.values
        # 这里可以根据数据的稀疏性决定是否转换
        # 一个简单的启发式：如果0的比例超过某个阈值，则认为是稀疏的
        sparsity_ratio = 1.0 - np.count_nonzero(X_dense) / X_dense.size
        print(f"数据稀疏度 (0的比例): {sparsity_ratio:.4f}")
        if sparsity_ratio > 0.5:  # 例如，超过50%为0
            X = sparse.csr_matrix(X_dense)
            print("特征数据已转换为稀疏矩阵 (csr_matrix)。")
        else:
            X = X_dense
            print("特征数据保持为密集矩阵。")

        print(f"数据加载成功: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
        print(f"使用的预处理器参数: {preprocessor_params}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 移除y中的NaN
    if np.isnan(y).any():
        print("移除y中的NaN...")
        mask = ~np.isnan(y)
        if sparse.issparse(X):
            X = X[mask]  # 稀疏矩阵的布尔索引
        else:
            X = X[mask]
        y = y[mask]

    # 数据划分: 10% 测试集, 90% 临时集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    print(f"\n初步数据划分: 临时集={X_temp.shape[0]}, 测试集={X_test.shape[0]}")

    # 再次划分临时集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=SEED, stratify=y_temp
    )
    print(f"最终数据划分: 训练集={X_train.shape[0]}, 验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")

    # --- 调用新的训练评估函数 (取消搜索) ---
    print("\n=== 启动不带超参数搜索的训练评估流程 (XGBoost 版本, 稀疏矩阵) ===")
    results = train_and_evaluate_without_search(X_train, y_train, X_val, y_val, X_test, y_test,
                                                preprocessor_params=preprocessor_params)

    # 保存预测结果 (使用 F1 优化的预测)
    if 'y_proba' in results and 'y_pred' in results:
        pd.DataFrame({
            'true_label': y_test,
            'pred_prob': results['y_proba'],
            'pred_label_f1_optimized': results['y_pred']
        }).to_csv("predictions.csv", index=False)
        print("\n预测结果已保存")


if __name__ == "__main__":
    main()



