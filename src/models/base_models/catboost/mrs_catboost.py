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
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, ParameterSampler, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
# --- CatBoost 导入 ---
from catboost import CatBoostClassifier

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
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=SEED),
            tomek=TomekLinks(sampling_strategy='majority'),
            random_state=SEED
        )

        X_resampled, y_resampled = smotetomek.fit_resample(X, y)
        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled.shape[0]}")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y


# --- 预处理器类 ---
class AdvancedPreprocessor:
    """高级数据预处理管道，包含清洗、标准化、特征工程"""

    def __init__(self):
        self.imputer_num = None
        self.scaler = None
        self.winsorizer = None
        self.pca = None
        self.numerical_features = None

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

        print("步骤 4/4: PCA 降维...")
        # 4. PCA 降维
        self.pca = PCA(n_components=0.95)
        self.pca.fit(X_scaled)

        print(f"预处理完成。PCA后特征数: {self.pca.n_components_}")
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

        # 4. PCA 降维
        X_pca = self.pca.transform(X_scaled)

        return X_pca

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)


# --- 特征选择器类 (保持不变，CatBoost 本身有很好的特征重要性评估) ---
class GBDT_RFE_FeatureSelector:
    """基于CatBoost的RFE特征选择器 (使用CatBoostClassifier)"""

    def __init__(self, n_features=None, step=0.1):
        self.n_features = n_features
        self.step = step
        self.selector = None
        self.feature_mask_ = None

    def fit(self, X, y):
        # 特征选择使用 CatBoost
        cb = CatBoostClassifier(
            iterations=500,
            depth=3,
            learning_rate=0.0001,
            verbose=0, # 关闭CatBoost训练日志
            random_seed=SEED,
            thread_count=NUM_CORES # 使用所有核心
        )

        if self.n_features is None:
            self.n_features = max(1, X.shape[1] // 2)  # 确保至少选择1个特征

        self.selector = RFE(
            estimator=cb,
            n_features_to_select=self.n_features,
            step=self.step,
            verbose=1
        )
        self.selector.fit(X, y)
        self.feature_mask_ = self.selector.support_
        return self

    def transform(self, X):
        return self.selector.transform(X)


# --- 最终集成模型类 (GBDT 改为 CatBoost) ---
class FinalEnsemble:
    """最终集成模型（CatBoost + CatBoost）"""

    def __init__(self):
        self.cb_model_auc = None  # 优化AUC的模型
        self.cb_model_recall = None  # 优化召回率的模型
        self.preprocessor = None  # 使用高级预处理器

    def fit(self, X, y):
        # 第一阶段：高级数据预处理 (包含清洗, 标准化, PCA)
        print("\n[Stage 1] 高级数据预处理...")
        self.preprocessor = AdvancedPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y)

        # 处理类别不平衡 - 使用固定参数的 SMOTETomek
        print("\n[Stage 1.5] 处理类别不平衡 (固定参数 SMOTETomek)...")
        try:
            X_resampled, y_resampled = moo_smotetomek_func(X_processed, y)
            print(f"  SMOTETomek应用成功。新样本数: {X_resampled.shape[0]}")
        except Exception as e:
            print(f"  SMOTETomek失败: {e}。回退到原始数据。")
            X_resampled, y_resampled = X_processed, y

        # --- 为 CatBoost 训练准备带验证集的数据 ---
        X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(
            X_resampled, y_resampled, test_size=0.15, random_state=SEED, stratify=y_resampled
        )

        # 第二阶段：训练第一个 CatBoost 模型（优化AUC）
        print("\n[Stage 2] 训练优化AUC的CatBoost模型...")
        # 参数搜索空间
        cb_param_dist_auc = {
            'iterations': [100, 300, 500],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'subsample': [0.8, 1.0],
        }
        cb_sampler_auc = ParameterSampler(cb_param_dist_auc, n_iter=20, random_state=SEED)

        best_score_auc = -np.inf
        best_cb_params_auc = None

        for i, params in enumerate(cb_sampler_auc):
            print(f"    -> 尝试 AUC 优化 CatBoost 参数组合 {i + 1}/20: {params}")
            try:
                model = CatBoostClassifier(
                    verbose=0, # 关闭训练日志
                    random_seed=SEED,
                    thread_count=NUM_CORES, # 使用所有核心
                    eval_set=[(X_val_cb, y_val_cb)],
                    early_stopping_rounds=20,
                    use_best_model=True,
                    **params
                )
                model.fit(X_train_cb, y_train_cb, silent=True) # silent=True 也用于抑制输出
                val_proba = model.predict_proba(X_val_cb)[:, 1]
                val_auc = roc_auc_score(y_val_cb, val_proba)
                print(f"      -> 验证集 AUC: {val_auc:.4f}")

                if val_auc > best_score_auc:
                    best_score_auc = val_auc
                    best_cb_params_auc = params
                    print(f"      -> 发现更优 AUC 参数: {best_cb_params_auc}, AUC: {best_score_auc:.4f}")
            except Exception as e:
                print(f"      -> 参数组合 {params} 评估失败: {e}")

        if best_cb_params_auc:
            print(f"\n    -> 最佳 AUC 优化 CatBoost 参数: {best_cb_params_auc}")
            self.cb_model_auc = CatBoostClassifier(
                verbose=0,
                random_seed=SEED,
                thread_count=NUM_CORES,
                eval_set=[(X_val_cb, y_val_cb)],
                early_stopping_rounds=20,
                use_best_model=True,
                **best_cb_params_auc
            )
            self.cb_model_auc.fit(X_resampled, y_resampled, silent=True)
        else:
            print("    -> 未能找到有效 AUC 优化参数。使用默认参数。")
            self.cb_model_auc = CatBoostClassifier(
                iterations=100,
                depth=3,
                learning_rate=0.1,
                verbose=0,
                random_seed=SEED,
                thread_count=NUM_CORES
            )
            self.cb_model_auc.fit(X_resampled, y_resampled, silent=True)

        # 第三阶段：训练第二个 CatBoost 模型（优化F1/Recall）
        print("\n[Stage 3] 训练优化F1/Recall的CatBoost模型...")
        # 参数搜索空间，可以偏向于更深的树来提高 recall
        cb_param_dist_recall = {
            'iterations': [100, 300, 500],
            'depth': [5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'subsample': [0.8, 1.0],
        }
        cb_sampler_recall = ParameterSampler(cb_param_dist_recall, n_iter=20, random_state=SEED)

        best_score_f1 = -np.inf
        best_cb_params_recall = None

        for i, params in enumerate(cb_sampler_recall):
            print(f"    -> 尝试 Recall/F1 优化 CatBoost 参数组合 {i + 1}/20: {params}")
            try:
                model = CatBoostClassifier(
                    verbose=0,
                    random_seed=SEED,
                    thread_count=NUM_CORES,
                    eval_set=[(X_val_cb, y_val_cb)],
                    early_stopping_rounds=20,
                    use_best_model=True,
                    **params
                )
                model.fit(X_train_cb, y_train_cb, silent=True)
                val_pred = model.predict(X_val_cb)
                val_f1 = f1_score(y_val_cb, val_pred)
                print(f"      -> 验证集 F1: {val_f1:.4f}")

                if val_f1 > best_score_f1:
                    best_score_f1 = val_f1
                    best_cb_params_recall = params
                    print(f"      -> 发现更优 F1 参数: {best_cb_params_recall}, F1: {best_score_f1:.4f}")
            except Exception as e:
                print(f"      -> 参数组合 {params} 评估失败: {e}")

        if best_cb_params_recall:
            print(f"\n    -> 最佳 Recall/F1 优化 CatBoost 参数: {best_cb_params_recall}")
            self.cb_model_recall = CatBoostClassifier(
                verbose=0,
                random_seed=SEED,
                thread_count=NUM_CORES,
                eval_set=[(X_val_cb, y_val_cb)],
                early_stopping_rounds=20,
                use_best_model=True,
                **best_cb_params_recall
            )
            self.cb_model_recall.fit(X_resampled, y_resampled, silent=True)
        else:
            print("    -> 未能找到有效 Recall/F1 优化参数。使用默认参数。")
            self.cb_model_recall = CatBoostClassifier(
                iterations=100,
                depth=7,
                learning_rate=0.1,
                verbose=0,
                random_seed=SEED,
                thread_count=NUM_CORES
            )
            self.cb_model_recall.fit(X_resampled, y_resampled, silent=True)

    def predict_proba(self, X):
        """预测概率（返回两个模型的平均概率）"""
        X_processed = self.preprocessor.transform(X)

        # 获取两个模型的预测概率 (取正类概率)
        cb_auc_proba = self.cb_model_auc.predict_proba(X_processed)[:, 1]
        cb_recall_proba = self.cb_model_recall.predict_proba(X_processed)[:, 1]

        # 返回平均概率
        return (cb_auc_proba + cb_recall_proba) / 2

    # --- 新增方法：加权投票预测 ---
    def predict_with_weighted_voting(self, X, weight_auc=0.3, weight_recall=0.7):
        """
        使用加权投票进行预测。
        :param X: 输入特征
        :param weight_auc: 优化AUC模型的权重
        :param weight_recall: 优化召回率模型的权重
        :return: 预测的类别 (0 或 1)
        """
        if not np.isclose(weight_auc + weight_recall, 1.0):
            warnings.warn("权重之和不为1，将进行归一化。")
            total_weight = weight_auc + weight_recall
            weight_auc /= total_weight
            weight_recall /= total_weight

        X_processed = self.preprocessor.transform(X)

        # 获取两个模型的预测概率 (取正类概率)
        cb_auc_proba = self.cb_model_auc.predict_proba(X_processed)[:, 1]
        cb_recall_proba = self.cb_model_recall.predict_proba(X_processed)[:, 1]

        # 计算加权平均概率
        weighted_proba = weight_auc * cb_auc_proba + weight_recall * cb_recall_proba

        # 使用 0.5 阈值进行分类
        # 注意：如果需要使用验证集上找到的最佳阈值，需要额外传递或存储它
        # 这里为了简化，直接使用 0.5
        return (weighted_proba >= 0.5).astype(int)


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


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """训练评估流程，包含验证集和多阈值选择"""
    # 训练集成模型
    print("\n=== 训练集成模型 ===")
    start_time = time.time()
    ensemble = FinalEnsemble()
    ensemble.fit(X_train, y_train)
    end_time = time.time()
    print(f"\n训练耗时: {end_time - start_time:.2f} 秒")

    # 保存训练好的模型
    save_model(ensemble, 'best_cascaded_model_catboost.pkl') # 更新模型文件名

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

    # --- 使用加权投票方法评估 ---
    print("\n--- 使用召回率优先加权投票评估 (权重 AUC: 0.3, Recall: 0.7) ---")
    # 使用 0.5 阈值进行加权投票预测
    y_pred_weighted_vote = ensemble.predict_with_weighted_voting(X_test, weight_auc=0.3, weight_recall=0.7)
    recall_wv = recall_score(y_test, y_pred_weighted_vote)
    auc_wv = roc_auc_score(y_test, test_proba) # AUC 仍使用概率
    precision_wv = precision_score(y_test, y_pred_weighted_vote)
    f1_wv = f1_score(y_test, y_pred_weighted_vote)
    accuracy_wv = accuracy_score(y_test, y_pred_weighted_vote)
    print(f"Weighted Vote Recall: {recall_wv:.4f}")
    print(f"Weighted Vote AUC: {auc_wv:.4f}")
    print(f"Weighted Vote Precision: {precision_wv:.4f}")
    print(f"Weighted Vote F1 Score: {f1_wv:.4f}")
    print(f"Weighted Vote Accuracy: {accuracy_wv:.4f}")
    print(f"Weighted Vote TestScore (20*P+50*AUC+30*R): {20 * precision_wv + 50 * auc_wv + 30 * recall_wv:.4f}")

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
        'weighted_vote_results': {
            'recall': recall_wv,
            'precision': precision_wv,
            'f1': f1_wv,
            'accuracy': accuracy_wv
        }
    }


def main():
    # 检查CUDA可用性 (注意: CatBoost 的 GPU 支持与 PyTorch 的 CUDA 检查不完全相同)
    print("CUDA 可用性 (PyTorch):", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # 加载数据
    print("\n=== 数据加载 ===")
    try:
        data = pd.read_csv("data/clean.csv")
        if "company_id" in data.columns:
            data = data.drop(columns=["company_id"])
        if "target" not in data.columns:
            raise ValueError("数据中必须包含'target'列")

        # 分离特征和目标
        y = data["target"].values.astype(int)
        X = data.drop(columns=["target"]).values

        print(f"数据加载成功: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
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

    # 训练和评估 (传入验证集)
    results = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    # 保存预测结果 (使用 F1 优化的预测)
    pd.DataFrame({
        'true_label': y_test,
        'pred_prob': results['y_proba'],
        'pred_label_f1_optimized': results['y_pred'],
        'pred_label_weighted_vote': results['weighted_vote_results']['recall'] # 示例，实际应保存 y_pred_weighted_vote
    }).to_csv("predictions_catboost.csv", index=False) # 更新预测文件名


if __name__ == "__main__":
    main()
