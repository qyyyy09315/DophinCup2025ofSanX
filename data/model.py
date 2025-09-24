# -*- coding: utf-8 -*-
"""取消超参数搜索示例"""

import os
import pickle
import random
import time
import warnings
import multiprocessing
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
# --- 导入随机森林和特征选择器 ---
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
# --- ---
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, ParameterSampler, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures
# --- 导入 TruncatedSVD 作为 PCA 的替代方案 (可选但推荐) ---
from sklearn.decomposition import PCA, TruncatedSVD
# --- 导入 Pipeline ---
from sklearn.pipeline import Pipeline
# --- 导入绘图库 ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- ---

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
    """
    print("  -> 应用固定参数的 SMOTETomek...")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTETomek.")
        return X, y

    try:
        # 使用固定的参数组合
        smotetomek = SMOTETomek(
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=SEED, n_jobs=-1),
            tomek=TomekLinks(sampling_strategy='majority', n_jobs=-1),
            random_state=SEED
        )

        X_resampled, y_resampled = smotetomek.fit_resample(X, y)
        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled.shape[0]}")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y

# --- 更新 AdvancedPreprocessor 类 (调整步骤顺序) ---
# --- 预处理器类 (增强版，调整特征处理顺序) ---
class AdvancedPreprocessor:
    """高级数据预处理管道，包含清洗、标准化、特征工程（分箱、交叉）、随机森林特征选择和降维"""

    def __init__(self, n_bins=None, cross_degree=None, rf_n_features_to_select=None, use_truncated_svd=False,
                 svd_n_components=None):
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
        """
        self.imputer_num = None
        self.scaler = None
        self.winsorizer = None
        self.binner = None
        self.crosser = None
        self.rf_selector = None  # 新增：随机森林选择器
        self.pca = None
        self.svd = None  # 新增：TruncatedSVD
        self.numerical_features = None
        self.n_bins = n_bins
        self.cross_degree = cross_degree
        self.rf_n_features_to_select = rf_n_features_to_select
        self.use_truncated_svd = use_truncated_svd
        self.svd_n_components = svd_n_components

    def fit(self, X, y=None):
        """拟合预处理器"""
        # 假设 X 是一个 DataFrame，或者需要提供列名列表 self.numerical_features
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        elif self.numerical_features is None:
            # 如果 X 是 numpy array 且未提供列名，则假设所有列都是数值型
            self.numerical_features = list(range(X.shape[1]))
            print("警告: 未提供列名，假设所有特征均为数值型。")

        print("步骤 1/4: 缺失值处理...")
        # 1. 缺失值处理 - 使用中位数填充数值型特征
        self.imputer_num = SimpleImputer(strategy='median')
        X_imputed = X.copy()
        if isinstance(X, np.ndarray):
            X_imputed[:, self.numerical_features] = self.imputer_num.fit_transform(X[:, self.numerical_features])
        else:  # DataFrame
            X_imputed.loc[:, self.numerical_features] = self.imputer_num.fit_transform(X[self.numerical_features])

        print("步骤 2/4: 异常值处理 (Winsorization)...")
        # 2. 异常值处理 - Winsorization
        self.winsorizer = QuantileTransformer(output_distribution='uniform', random_state=SEED,
                                              n_quantiles=min(1000, X.shape[0]))
        X_winsorized = X_imputed.copy()
        if isinstance(X_imputed, np.ndarray):
            X_winsorized[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[:, self.numerical_features])
        else:  # DataFrame
            X_winsorized.loc[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[self.numerical_features])

        print("步骤 3/4: 特征转换与标准化...")
        # 3. 特征转换与标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_winsorized)

        # --- 新增步骤: 特征分箱 ---
        if self.n_bins is not None and self.n_bins > 1:
            print(f"步骤 3.1: 特征分箱 (n_bins={self.n_bins})...")
            # 使用 'quantile' 策略以获得更平衡的箱
            self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='quantile')
            X_binned = self.binner.fit_transform(X_scaled)  # 注意输入是 X_scaled
            print(f"  -> 分箱后特征数: {X_binned.shape[1]}")
        else:
            X_binned = X_scaled  # 注意输入是 X_scaled
            print("  -> 跳过分箱步骤。")

        # --- 新增步骤: 特征交叉 ---
        if self.cross_degree is not None and self.cross_degree >= 2:
            print(f"步骤 3.2: 特征交叉 (degree={self.cross_degree})...")
            # 限制交互项数量，degree=2 通常足够
            self.crosser = PolynomialFeatures(degree=self.cross_degree, interaction_only=True, include_bias=False)
            X_crossed = self.crosser.fit_transform(X_binned)  # 注意输入是 X_binned
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
            temp_rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=SEED)
            # threshold=-np.inf 确保选出 max_features 个特征，即使重要性很低
            self.rf_selector = SelectFromModel(temp_rf, max_features=n_features_to_select, threshold=-np.inf)
            X_rf_selected = self.rf_selector.fit_transform(X_crossed, y)  # 输入是 X_crossed
            print(f"  -> 随机森林选择后特征数: {X_rf_selected.shape[1]}")
        else:
            X_rf_selected = X_crossed  # 输入是 X_crossed
            print("  -> 跳过随机森林特征选择步骤。")

        # --- 修改点: PCA / TruncatedSVD 降维 (输入是 X_rf_selected) ---
        if self.use_truncated_svd:
            print("步骤 4/4: TruncatedSVD 降维...")
            n_components_svd = self.svd_n_components if self.svd_n_components is not None else 100
            self.svd = TruncatedSVD(n_components=n_components_svd, random_state=SEED)
            self.svd.fit(X_rf_selected)  # 输入是 X_rf_selected
            print(f"TruncatedSVD 完成。保留特征数: {self.svd.n_components}")
        else:
            print("步骤 4/4: PCA 降维...")
            # 4. PCA 降维 (输入是 X_rf_selected)
            self.pca = PCA(n_components=0.95)  # 或者设置一个固定的较小整数，如 50
            try:
                self.pca.fit(X_rf_selected)  # 输入是 X_rf_selected
                print(f"PCA 完成。保留特征数: {self.pca.n_components_}")
            except ValueError as e:
                if "Indexing a matrix" in str(e) and "integer overflow" in str(e):
                    print(f"    -> PCA 失败: {e}")
                    print("    -> 数据量过大导致PCA整数溢出。建议:")
                    print("        1. 进一步减少特征数量 (减小 rf_n_features_to_select)。")
                    print("        2. 使用 TruncatedSVD (设置 use_truncated_svd=True)。")
                    print("        3. 减少特征交叉度数 (cross_degree)。")
                    raise e  # 重新抛出异常以便上层处理
                else:
                    raise e  # 重新抛出其他ValueError
        # --- ---

        print(f"预处理完成。")
        return self

    def transform(self, X):
        """应用预处理步骤"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values  # 转换为 numpy array 以便处理
        else:
            X_values = X

        # 1. 缺失值处理
        X_imputed = X_values.copy()
        X_imputed[:, self.numerical_features] = self.imputer_num.transform(X_values[:, self.numerical_features])

        # 2. 异常值处理
        X_winsorized = X_imputed.copy()
        X_winsorized[:, self.numerical_features] = self.winsorizer.transform(X_imputed[:, self.numerical_features])

        # 3. 标准化
        X_scaled = self.scaler.transform(X_winsorized)

        # 3.1. 特征分箱 (如果已拟合)
        if self.binner is not None:
            X_binned = self.binner.transform(X_scaled)  # 输入是 X_scaled
        else:
            X_binned = X_scaled  # 输入是 X_scaled

        # 3.2. 特征交叉 (如果已拟合)
        if self.crosser is not None:
            X_crossed = self.crosser.transform(X_binned)  # 输入是 X_binned
        else:
            X_crossed = X_binned  # 输入是 X_binned

        # 3.3. 随机森林特征选择 (如果已拟合)
        if self.rf_selector is not None:
            X_rf_selected = self.rf_selector.transform(X_crossed)  # 输入是 X_crossed
        else:
            X_rf_selected = X_crossed  # 输入是 X_crossed

        # 4. PCA / TruncatedSVD 降维
        if self.use_truncated_svd and self.svd is not None:
            X_reduced = self.svd.transform(X_rf_selected)  # 输入是 X_rf_selected
        elif self.pca is not None:
            X_reduced = self.pca.transform(X_rf_selected)  # 输入是 X_rf_selected
        else:
            X_reduced = X_rf_selected  # 如果没有降维步骤

        return X_reduced

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)

# --- ---
# --- 特征选择器类 ---
class GBDT_RFE_FeatureSelector:
    """基于GBDT的RFE特征选择器"""

    def __init__(self, n_features=None, step=0.1):
        self.n_features = n_features
        self.step = step
        self.selector = None
        self.feature_mask_ = None

    def fit(self, X, y):
        # 特征选择使用 GBDT
        gbdt = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.0001,
            random_state=SEED
        )

        if self.n_features is None:
            self.n_features = max(1, X.shape[1] // 2)  # 确保至少选择1个特征

        self.selector = RFE(
            estimator=gbdt,
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
    pr_auc = auc(recall, precision) # AUC of PR curve
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

# --- 修改后的训练评估流程 (取消超参数搜索) ---
def train_and_evaluate_without_search(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor_params=None):
    """取消超参数搜索的训练评估流程"""
    # --- 构建 Pipeline ---
    print("\n=== 构建 Scikit-learn Pipeline ===")
    pipeline = Pipeline([
        ('preprocessor', AdvancedPreprocessor(**preprocessor_params)),
        # ('smote_tomek', 'passthrough'), # SMOTETomek 通常在Pipeline外处理或用自定义转换器，这里我们手动处理
        ('classifier', GradientBoostingClassifier(random_state=SEED)) # 使用默认参数
    ])

    # --- 手动应用 SMOTETomek (如果需要) ---
    # 注意：Pipeline中的转换器通常假设输入是X，而SMOTETomek同时需要X和y。
    # 为了简化，我们在Pipeline外应用它。
    # 如果想在Pipeline内使用，需要自定义一个转换器。
    print("\n=== 应用 SMOTETomek 重采样 ===")
    X_train_resampled, y_train_resampled = moo_smotetomek_func(X_train, y_train)

    # --- 模型训练 ---
    print("\n=== 开始模型训练 ===")
    start_time = time.time()
    # pipeline.fit(X_train_resampled, y_train_resampled) # 如果在Pipeline内处理SMOTETomek
    # 由于我们手动处理了SMOTETomek，直接将预处理后的数据传给Pipeline (不包括分类器)
    pipeline.named_steps['preprocessor'].fit(X_train_resampled, y_train_resampled)
    X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train_resampled)
    pipeline.named_steps['classifier'].fit(X_train_processed, y_train_resampled) # 单独训练分类器

    end_time = time.time()
    print(f"\n模型训练耗时: {end_time - start_time:.2f} 秒")

    # --- 保存训练好的模型 ---
    save_model(pipeline, 'trained_model_without_search.pkl')

    # --- 在验证集上寻找最佳阈值 ---
    print("\n=== 验证集阈值选择 ===")
    # 需要先对验证集进行预处理
    X_val_processed = pipeline.named_steps['preprocessor'].transform(X_val)
    val_proba = pipeline.named_steps['classifier'].predict_proba(X_val_processed)[:, 1]

    # 1. 寻找最佳 F1 阈值
    print("\n[阈值选择 1] 优化 F1 分数...")
    threshold_f1 = find_best_f1_threshold(y_val, val_proba)

    # 2. 寻找高召回率阈值 (例如，要求 Precision >= 0.4)
    print("\n[阈值选择 2] 优化召回率 (要求 Precision >= 0.4)...")
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # --- 测试集评估 ---
    print("\n=== 测试集评估 ===")
    # 需要先对测试集进行预处理
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
    test_proba = pipeline.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]

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
    evaluate_and_plot(y_test, test_proba, threshold_f1, threshold_high_recall, model_name="GBDT (No Search)")

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
        'trained_model': pipeline # 返回训练好的Pipeline
    }

def main():
    # 检查CUDA可用性
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # --- 配置预处理器参数 ---
    # 建议1: 先简化预处理器，关闭分箱和交叉，看基础效果
    # preprocessor_params = {
    #     'n_bins': None, # 关闭分箱
    #     'cross_degree': None, # 关闭交叉
    #     'rf_n_features_to_select': 0.5, # 选择50%特征
    #     # 'use_truncated_svd': True,
    #     # 'svd_n_components': 50
    # }
    # 建议2: 使用原始复杂预处理器
    preprocessor_params = {
        'n_bins': 10,
        'cross_degree': 2,
        'rf_n_features_to_select': 1000, # 可尝试更小值如 500
        # 'use_truncated_svd': True,        # 可选：如果PCA仍有问题，启用TruncatedSVD
        # 'svd_n_components': 50           # 可选：TruncatedSVD保留的成分
    }

    # 加载数据
    print("\n=== 数据加载 ===")
    try:
        data = pd.read_csv("clean.csv")  # 确保您的数据文件路径正确
        if "company_id" in data.columns:
            data = data.drop(columns=["company_id"])
        if "target" not in data.columns:
            raise ValueError("数据中必须包含'target'列")

        # 分离特征和目标
        y = data["target"].values.astype(int)
        X = data.drop(columns=["target"]).values

        print(f"数据加载成功: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
        print(f"使用的预处理器参数: {preprocessor_params}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 移除y中的NaN
    if np.isnan(y).any():
        print("移除y中的NaN...")
        mask = ~np.isnan(y)
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
    print("\n=== 启动不带超参数搜索的训练评估流程 ===")
    results = train_and_evaluate_without_search(X_train, y_train, X_val, y_val, X_test, y_test,
                                 preprocessor_params=preprocessor_params)

    # 保存预测结果 (使用 F1 优化的预测)
    if 'y_proba' in results and 'y_pred' in results:
        pd.DataFrame({
            'true_label': y_test,
            'pred_prob': results['y_proba'],
            'pred_label_f1_optimized': results['y_pred']
        }).to_csv("predictions_gbdt.csv", index=False)
        print("\n预测结果已保存")

if __name__ == "__main__":
    main()



