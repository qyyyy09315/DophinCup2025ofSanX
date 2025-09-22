import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, ParameterSampler, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# 环境配置
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# --- 简化的多目标优化 SMOTETomek ---
def moo_smotetomek_func(X, y, scoring_weights=(0.4, 0.4, 0.2), n_trials=10, cv_folds=3):
    """
    使用简化多目标优化寻找最佳 SMOTETomek 参数。
    目标：最大化 AUC, Recall, Accuracy 的加权组合。
    """
    print("  -> 应用简化MOO-SMOTETomek...")
    print(f"    -> 优化目标: {scoring_weights[0]}*AUC + {scoring_weights[1]}*Recall + {scoring_weights[2]}*Accuracy")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTETomek.")
        return X, y

    # 定义 SMOTETomek 参数搜索空间
    # SMOTETomek 结合了 SMOTE (over-sampling) 和 Tomek Links (under-sampling)
    # 我们主要调整 SMOTE 的参数
    param_dist = {
        'smote_k_neighbors': [3, 5, 7],  # SMOTE 的 k_neighbors
        'smote_sampling_strategy': ['auto', 0.8, 1.0, 1.2], # SMOTE 的 sampling_strategy
        # TomekLinks 通常没有太多参数需要调优
    }

    best_score = -np.inf
    best_params = None
    best_X_resampled, best_y_resampled = X, y

    # 简单的随机搜索
    sampler = ParameterSampler(param_dist, n_iter=n_trials, random_state=SEED)

    for i, params in enumerate(sampler):
        print(f"    -> 尝试参数组合 {i + 1}/{n_trials}: {params}")
        try:
            # 调整 k_neighbors 以适应当前数据
            n_minority = np.sum(y == 1)
            k = min(params['smote_k_neighbors'], n_minority - 1) if n_minority > 1 else 1
            k = max(1, k)

            sampling_strat = params['smote_sampling_strategy']

            # 创建 SMOTETomek 实例
            # 注意：imblearn 0.8.0+ 中，SMOTETomek 的参数传递方式可能有变化
            # 我们直接传递 SMOTE 实例和 TomekLinks 实例
            smote_sampler = SMOTE(random_state=SEED, k_neighbors=k, sampling_strategy=sampling_strat)
            tomek_sampler = TomekLinks(sampling_strategy='majority') # 只对多数类应用 Tomek Links
            smotetomek = SMOTETomek(smote=smote_sampler, tomek=tomek_sampler, random_state=SEED)

            X_resampled, y_resampled = smotetomek.fit_resample(X, y)

            # 使用简单的 GBDT 评估性能
            eval_model = GradientBoostingClassifier(
                n_estimators=100, # 减少评估模型的资源消耗
                max_depth=3,
                learning_rate=0.1,
                random_state=SEED
            )

            # 使用交叉验证评估
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            auc_scores = cross_val_score(eval_model, X_resampled, y_resampled, cv=cv, scoring='roc_auc', n_jobs=1)
            recall_scores = cross_val_score(eval_model, X_resampled, y_resampled, cv=cv, scoring='recall', n_jobs=1)
            acc_scores = cross_val_score(eval_model, X_resampled, y_resampled, cv=cv, scoring='accuracy', n_jobs=1)

            mean_auc = np.mean(auc_scores)
            mean_recall = np.mean(recall_scores)
            mean_acc = np.mean(acc_scores)

            # 计算综合得分
            score = (scoring_weights[0] * mean_auc +
                     scoring_weights[1] * mean_recall +
                     scoring_weights[2] * mean_acc)

            print(f"      -> AUC: {mean_auc:.4f}, Recall: {mean_recall:.4f}, Acc: {mean_acc:.4f}, Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params
                print(f"      -> 发现更优参数: {best_params}, Score: {best_score:.4f}")

        except ValueError as e:
            print(f"      -> 参数组合 {params} 无效: {e}")
        except Exception as e:
            print(f"      -> 评估过程中出错: {e}")

    if best_params:
        print(f"    -> 最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
        # 使用最佳参数对完整数据进行最终采样
        try:
            n_minority = np.sum(y == 1)
            k_final = min(best_params['smote_k_neighbors'], n_minority - 1) if n_minority > 1 else 1
            k_final = max(1, k_final)

            smote_sampler_final = SMOTE(random_state=SEED, k_neighbors=k_final,
                                sampling_strategy=best_params['smote_sampling_strategy'])
            tomek_sampler_final = TomekLinks(sampling_strategy='majority')
            smotetomek_final = SMOTETomek(smote=smote_sampler_final, tomek=tomek_sampler_final, random_state=SEED)

            best_X_resampled, best_y_resampled = smotetomek_final.fit_resample(X, y)
            print(f"    -> 应用最佳参数后，样本数: {best_X_resampled.shape[0]}")
        except Exception as e:
            print(f"    -> 应用最佳参数失败: {e}。回退到原始数据。")
    else:
        print("    -> 未能找到有效参数组合。回退到原始数据。")

    return best_X_resampled, best_y_resampled


# --- 预处理器类 (保持不变) ---
class AdvancedPreprocessor:
    """高级数据预处理管道，包含清洗、标准化、特征工程"""

    def __init__(self):
        self.imputer_num = None
        self.scaler = None
        self.winsorizer = None
        self.pca = None
        self.numerical_features = None  # 需要外部传入或在fit时推断

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
        self.winsorizer = QuantileTransformer(output_distribution='uniform', random_state=SEED)
        X_winsorized = X_imputed.copy()
        if isinstance(X_imputed, np.ndarray):
            X_winsorized[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[:, self.numerical_features])
        else:  # DataFrame
            X_winsorized.loc[:, self.numerical_features] = self.winsorizer.fit_transform(
                X_imputed[self.numerical_features])

        print("步骤 3/4: 特征转换与标准化...")
        # 3. 特征转换与标准化
        # Z-score 标准化
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


# --- 特征选择器类 (保持不变，但使用 GBDT) ---
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


# --- 最终集成模型类 (修改采样部分和模型为 GBDT) ---
class FinalEnsemble:
    """最终集成模型（GBDT + GBDT）"""

    def __init__(self):
        self.gbdt_model_auc = None  # 优化AUC的模型
        self.gbdt_model_recall = None # 优化召回率的模型
        self.preprocessor = None  # 使用高级预处理器

    def fit(self, X, y):
        # 第一阶段：高级数据预处理 (包含清洗, 标准化, PCA)
        print("\n[Stage 1] 高级数据预处理...")
        self.preprocessor = AdvancedPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y)

        # 处理类别不平衡 - 使用简化的MOO-SMOTETomek
        print("\n[Stage 1.5] 处理类别不平衡 (简化MOO-SMOTETomek)...")
        # 定义优化目标权重 (AUC, Recall, Accuracy)
        # 示例：更关注AUC和Recall，适度关注Accuracy (可以调整以优化Precision)
        scoring_weights = (0.3, 0.4, 0.3)  # 可调
        n_trials = 30  # 可调 (减少试验次数)
        try:
            X_resampled, y_resampled = moo_smotetomek_func(X_processed, y, scoring_weights=scoring_weights,
                                                      n_trials=n_trials)
            print(f"  MOO-SMOTETomek应用成功。新样本数: {X_resampled.shape[0]}")
        except Exception as e:
            print(f"  MOO-SMOTETomek失败: {e}。回退到原始数据。")
            X_resampled, y_resampled = X_processed, y

        # --- 为 GBDT 训练准备带验证集的数据 ---
        # Scikit-learn 的 GBDT 本身不直接支持 early stopping，但我们可以在训练后选择最佳迭代轮次
        # 或者使用 validation_fraction 和 n_iter_no_change 参数
        X_train_gbdt, X_val_gbdt, y_train_gbdt, y_val_gbdt = train_test_split(
            X_resampled, y_resampled, test_size=0.15, random_state=SEED, stratify=y_resampled
        )


        # 第二阶段：训练第一个 GBDT 模型（优化AUC）
        print("\n[Stage 2] 训练优化AUC的GBDT模型...")
        # 参数搜索空间
        gbdt_param_dist_auc = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            # 'max_features': ['sqrt', 'log2', None] # 可选
        }
        gbdt_sampler_auc = ParameterSampler(gbdt_param_dist_auc, n_iter=20, random_state=SEED)

        best_score_auc = -np.inf
        best_gbdt_params_auc = None

        for i, params in enumerate(gbdt_sampler_auc):
            print(f"    -> 尝试 AUC 优化 GBDT 参数组合 {i + 1}/20: {params}")
            try:
                model = GradientBoostingClassifier(
                    validation_fraction=0.1, # 使用一部分训练数据作为内部验证集
                    n_iter_no_change=10, # 10轮内没有 improvement 则停止
                    tol=1e-4, # improvement 的阈值
                    random_state=SEED,
                    **params
                )
                # 注意：GradientBoostingClassifier 的 best_estimator_ 是在训练后根据验证集选择的迭代次数
                model.fit(X_train_gbdt, y_train_gbdt)
                # 使用验证集评估
                val_proba = model.predict_proba(X_val_gbdt)[:, 1]
                val_auc = roc_auc_score(y_val_gbdt, val_proba)
                print(f"      -> 验证集 AUC: {val_auc:.4f}")

                if val_auc > best_score_auc:
                    best_score_auc = val_auc
                    best_gbdt_params_auc = params
                    print(f"      -> 发现更优 AUC 参数: {best_gbdt_params_auc}, AUC: {best_score_auc:.4f}")
            except Exception as e:
                print(f"      -> 参数组合 {params} 评估失败: {e}")

        if best_gbdt_params_auc:
            print(f"\n    -> 最佳 AUC 优化 GBDT 参数: {best_gbdt_params_auc}")
            self.gbdt_model_auc = GradientBoostingClassifier(
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
                random_state=SEED,
                **best_gbdt_params_auc
            )
            self.gbdt_model_auc.fit(X_resampled, y_resampled) # 使用全部重采样数据训练最终模型
        else:
            print("    -> 未能找到有效 AUC 优化参数。使用默认参数。")
            self.gbdt_model_auc = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=SEED
            )
            self.gbdt_model_auc.fit(X_resampled, y_resampled)


        # 第三阶段：训练第二个 GBDT 模型（优化F1/Recall）
        print("\n[Stage 3] 训练优化F1/Recall的GBDT模型...")
        # 参数搜索空间，可以偏向于更深的树来提高 recall
        gbdt_param_dist_recall = {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 7, 10], # 更深的树
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
        gbdt_sampler_recall = ParameterSampler(gbdt_param_dist_recall, n_iter=20, random_state=SEED)

        best_score_f1 = -np.inf
        best_gbdt_params_recall = None

        for i, params in enumerate(gbdt_sampler_recall):
            print(f"    -> 尝试 Recall/F1 优化 GBDT 参数组合 {i + 1}/20: {params}")
            try:
                model = GradientBoostingClassifier(
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    tol=1e-4,
                    random_state=SEED,
                    **params
                )
                model.fit(X_train_gbdt, y_train_gbdt)
                # 使用验证集评估 F1
                val_pred = model.predict(X_val_gbdt)
                val_f1 = f1_score(y_val_gbdt, val_pred)
                print(f"      -> 验证集 F1: {val_f1:.4f}")

                if val_f1 > best_score_f1:
                    best_score_f1 = val_f1
                    best_gbdt_params_recall = params
                    print(f"      -> 发现更优 F1 参数: {best_gbdt_params_recall}, F1: {best_score_f1:.4f}")
            except Exception as e:
                print(f"      -> 参数组合 {params} 评估失败: {e}")

        if best_gbdt_params_recall:
            print(f"\n    -> 最佳 Recall/F1 优化 GBDT 参数: {best_gbdt_params_recall}")
            self.gbdt_model_recall = GradientBoostingClassifier(
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
                random_state=SEED,
                **best_gbdt_params_recall
            )
            self.gbdt_model_recall.fit(X_resampled, y_resampled) # 使用全部重采样数据训练最终模型
        else:
            print("    -> 未能找到有效 Recall/F1 优化参数。使用默认参数。")
            self.gbdt_model_recall = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=7, # 稍深一点
                learning_rate=0.1,
                random_state=SEED
            )
            self.gbdt_model_recall.fit(X_resampled, y_resampled)


    def predict_proba(self, X):
        """预测概率（返回两个模型的平均概率）"""
        X_processed = self.preprocessor.transform(X)

        # 获取两个模型的预测概率 (取正类概率)
        gbdt_auc_proba = self.gbdt_model_auc.predict_proba(X_processed)[:, 1]
        gbdt_recall_proba = self.gbdt_model_recall.predict_proba(X_processed)[:, 1]

        # 返回平均概率
        return (gbdt_auc_proba + gbdt_recall_proba) / 2


# --- 阈值选择辅助函数 (保持不变)---
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
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 防止除零

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
    valid_indices = np.where(precisions[:-1] >= min_precision)[0] # precisions[:-1] 对应 thresholds

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
    save_model(ensemble, 'best_cascaded_model_gbdt.pkl') # 修改保存文件名

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
    auc_f1 = roc_auc_score(y_test, test_proba) # AUC 不依赖阈值
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
    auc_hr = roc_auc_score(y_test, test_proba) # AUC 不依赖阈值
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

    # 可以选择一个默认阈值返回，或者返回两个结果
    # 这里我们返回 F1 优化的结果作为主要结果
    return {
        'recall': recall_f1,
        'auc': auc_f1,
        'precision': precision_f1,
        'f1': f1_f1,
        'accuracy': accuracy_f1,
        'y_proba': test_proba,
        'y_pred': y_pred_f1, # 返回 F1 优化的预测
        'threshold': threshold_f1, # 返回 F1 优化的阈值
        # 也可以将高召回率的结果作为附加信息返回
        'high_recall_results': {
            'recall': recall_hr,
            'precision': precision_hr,
            'f1': f1_hr,
            'threshold': threshold_high_recall
        }
    }


def main():
    # 检查CUDA可用性 (虽然GBDT不使用GPU，但保留检查)
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))

    # 加载数据
    print("\n=== 数据加载 ===")
    try:
        data = pd.read_csv("clean.csv")
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

    # 再次划分临时集: 10% 验证集, 80% 最终训练集 (相对于原始数据是 81%)
    # 0.1 / 0.9 ≈ 0.1111
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
        'pred_label_f1_optimized': results['y_pred']
    }).to_csv("predictions_gbdt.csv", index=False) # 修改输出文件名

    # 如果需要，也可以保存高召回率的预测结果
    # pd.DataFrame({
    #     'true_label': y_test,
    #     'pred_prob': results['y_proba'],
    #     'pred_label_f1_optimized': results['y_pred'],
    #     'pred_label_high_recall': (results['y_proba'] >= results['high_recall_results']['threshold']).astype(int)
    # }).to_csv("predictions_detailed_gbdt.csv", index=False) # 修改输出文件名


if __name__ == "__main__":
    main()




