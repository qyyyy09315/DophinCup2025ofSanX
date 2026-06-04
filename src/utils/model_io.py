import os
import pickle
import random
import time
import warnings
import multiprocessing

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
    precision_recall_curve
from sklearn.model_selection import train_test_split, ParameterSampler, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures
# --- 导入 TruncatedSVD 作为 PCA 的替代方案 (可选但推荐) ---
from sklearn.decomposition import PCA, TruncatedSVD

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


# --- 最终集成模型类 ---
class FinalEnsemble:
    """最终集成模型（GBDT + GBDT）"""

    def __init__(self, preprocessor_params=None):
        """
        初始化最终集成模型。
        :param preprocessor_params: dict, 传递给 AdvancedPreprocessor 的参数。
        """
        self.gbdt_model_auc = None  # 优化AUC的模型
        self.gbdt_model_recall = None  # 优化召回率的模型
        self.preprocessor = None  # 使用高级预处理器
        self.preprocessor_params = preprocessor_params if preprocessor_params is not None else {}

    def fit(self, X, y):
        # 第一阶段：高级数据预处理 (包含清洗, 标准化, 分箱, 交叉, PCA)
        print("\n[Stage 1] 高级数据预处理...")
        self.preprocessor = AdvancedPreprocessor(**self.preprocessor_params)  # 使用传入的参数
        X_processed = self.preprocessor.fit_transform(X, y)

        # 处理类别不平衡 - 使用固定参数的 SMOTETomek
        print("\n[Stage 1.5] 处理类别不平衡 (固定参数 SMOTETomek)...")
        try:
            X_resampled, y_resampled = moo_smotetomek_func(X_processed, y)
            print(f"  SMOTETomek应用成功。新样本数: {X_resampled.shape[0]}")
        except Exception as e:
            print(f"  SMOTETomek失败: {e}。回退到原始数据。")
            X_resampled, y_resampled = X_processed, y

        # --- 为 GBDT 训练准备带验证集的数据 ---
        # 注意：这部分代码现在主要用于最终模型训练，因为超参数搜索已移除
        # 我们仍然分割，以保持代码结构清晰，但最终模型是在全部重采样数据上训练的
        X_train_gbdt, X_val_gbdt, y_train_gbdt, y_val_gbdt = train_test_split(
            X_resampled, y_resampled, test_size=0.15, random_state=SEED, stratify=y_resampled
        )
        # 为了保持变量名一致性和语义清晰，我们重新命名最终训练用的数据
        X_full_train = X_resampled
        y_full_train = y_resampled

        # 第二阶段：训练第一个 GBDT 模型（优化AUC）- 使用固定参数
        print("\n[Stage 2] 训练优化AUC的GBDT模型...")
        print("    -> 使用固定参数: {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1}")

        fixed_gbdt_params_auc = {
            'subsample': 0.8,
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.1
        }

        self.gbdt_model_auc = GradientBoostingClassifier(
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=SEED,
            **fixed_gbdt_params_auc
        )
        # 使用全部处理后的数据训练最终模型
        self.gbdt_model_auc.fit(X_full_train, y_full_train)
        print("    -> AUC优化GBDT模型训练完成。")

        # 第三阶段：训练第二个 GBDT 模型（优化F1/Recall）- 使用固定参数
        print("\n[Stage 3] 训练优化F1/Recall的GBDT模型...")
        print("    -> 使用固定参数: {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05}")

        fixed_gbdt_params_recall = {
            'subsample': 0.8,
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05
        }

        self.gbdt_model_recall = GradientBoostingClassifier(
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=SEED,
            **fixed_gbdt_params_recall
        )
        # 使用全部处理后的数据训练最终模型
        self.gbdt_model_recall.fit(X_full_train, y_full_train)
        print("    -> F1/Recall优化GBDT模型训练完成。")

    def predict_proba(self, X):
        """预测概率（返回两个模型的平均概率）"""
        X_processed = self.preprocessor.transform(X)

        # 获取两个模型的预测概率 (取正类概率)
        gbdt_auc_proba = self.gbdt_model_auc.predict_proba(X_processed)[:, 1]
        gbdt_recall_proba = self.gbdt_model_recall.predict_proba(X_processed)[:, 1]

        # 返回平均概率
        return (gbdt_auc_proba + gbdt_recall_proba) / 2


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


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor_params=None):
    """训练评估流程，包含验证集和多阈值选择"""
    # 训练集成模型
    print("\n=== 训练集成模型 ===")
    start_time = time.time()
    # 传递预处理器参数
    ensemble = FinalEnsemble(preprocessor_params=preprocessor_params)
    ensemble.fit(X_train, y_train)
    end_time = time.time()
    print(f"\n训练耗时: {end_time - start_time:.2f} 秒")

    # 保存训练好的模型
    save_model(ensemble, 'best_cascaded_model_gbdt.pkl')

    # --- 在验证集上寻找最佳阈值 ---
    print("\n=== 验证集阈值选择 ===")
    val_proba = ensemble.predict_proba(X_val)

    # 1. 寻找最佳 F1 阈值
    print("\n[阈值选择 1] 优化 F1 分数...")
    threshold_f1 = find_best_f1_threshold(y_val, val_proba)

    # 2. 寻找高召回率阈值 (例如，要求 Precision >= 0.5)
    print("\n[阈值选择 2] 优化召回率 (要求 Precision >= 0.5)...")
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # --- 测试集评估 ---
    print("\n=== 测试集评估 ===")
    test_proba = ensemble.predict_proba(X_test)

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
        }
    }


def main():
    # 检查CUDA可用性
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # --- 配置预处理器参数 ---
    # 例如：启用分箱 (10个箱), 2度交叉, 并使用随机森林选择特征
    # 如果交叉后特征数少于1000，则随机森林会选择所有特征。
    preprocessor_params = {
        'n_bins': 10,
        'cross_degree': 2,
        'rf_n_features_to_select': 1000,  # 在交叉之后选择最多1000个特征
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

    # 训练和评估 (传入验证集和预处理器参数)
    results = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                                 preprocessor_params=preprocessor_params)

    # 保存预测结果 (使用 F1 优化的预测)
    pd.DataFrame({
        'true_label': y_test,
        'pred_prob': results['y_proba'],
        'pred_label_f1_optimized': results['y_pred']
    }).to_csv("predictions_gbdt.csv", index=False)


if __name__ == "__main__":
    main()



