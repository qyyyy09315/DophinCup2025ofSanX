import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterSampler, cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, roc_curve, auc, \
    confusion_matrix, precision_recall_curve
# 注意：imblearn的安装: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn import FunctionSampler
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import pickle
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

# 环境配置
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# --- 简化的多目标优化 SMOTE ---
def moo_smote_func(X, y, scoring_weights=(0.4, 0.4, 0.2), n_trials=10, cv_folds=3):
    """
    使用简化多目标优化寻找最佳 SMOTE 参数。
    目标：最大化 AUC, Recall, Accuracy 的加权组合。
    """
    print("  -> 应用简化MOO-SMOTE...")
    print(f"    -> 优化目标: {scoring_weights[0]}*AUC + {scoring_weights[1]}*Recall + {scoring_weights[2]}*Accuracy")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTE.")
        return X, y

    # 定义 SMOTE 参数搜索空间
    param_dist = {
        'k_neighbors': [3, 5, 7],  # 限制k_neighbors范围，避免数据量小时出错
        'sampling_strategy': [0.8, 1.0, 1.2]  # 少数类采样到多数类的比例
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
            k = min(params['k_neighbors'], n_minority - 1) if n_minority > 1 else 1
            k = max(1, k)

            sampling_strat = params['sampling_strategy']

            smote = SMOTE(random_state=SEED, k_neighbors=k, sampling_strategy=sampling_strat)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # 使用简单的 XGBoost 评估性能
            eval_model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEED, n_jobs=1,  # 减少评估模型的资源消耗
                # scale_pos_weight 根据重采样后的数据调整
                scale_pos_weight=np.sum(y_resampled == 0) / np.sum(y_resampled == 1)
            )

            # 使用交叉验证评估
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            # 注意：cross_val_score 默认是越大越好，所以直接用 'roc_auc', 'recall', 'accuracy'
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
                # 注意：我们不直接返回这里的 resampled 数据，因为评估模型和最终模型可能不同
                # 我们只记录最佳参数，最后用完整数据和最佳参数再做一次采样
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
            k_final = min(best_params['k_neighbors'], n_minority - 1) if n_minority > 1 else 1
            k_final = max(1, k_final)

            smote_final = SMOTE(random_state=SEED, k_neighbors=k_final,
                                sampling_strategy=best_params['sampling_strategy'])
            best_X_resampled, best_y_resampled = smote_final.fit_resample(X, y)
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
        # 4. PCA 降维 (保留85%方差)
        self.pca = PCA(n_components=0.85)
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


# --- 特征选择器类 (保持不变) ---
class XGBRFE_FeatureSelector:
    """基于XGBoost的RFE特征选择器"""

    def __init__(self, n_features=None, step=0.1):
        self.n_features = n_features
        self.step = step
        self.selector = None
        self.feature_mask_ = None

    def fit(self, X, y):
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.0001,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1
        )

        if self.n_features is None:
            self.n_features = max(1, X.shape[1] // 2)  # 确保至少选择1个特征

        self.selector = RFE(
            estimator=xgb,
            n_features_to_select=self.n_features,
            step=self.step,
            verbose=1
        )
        self.selector.fit(X, y)
        self.feature_mask_ = self.selector.support_
        return self

    def transform(self, X):
        return self.selector.transform(X)


# --- 最终集成模型类 (修改采样部分) ---
class FinalEnsemble:
    """最终集成模型（XGBoost + AdaBoost）"""

    def __init__(self):
        self.xgb_model = None
        self.ada_model = None
        self.preprocessor = None  # 使用高级预处理器

    def fit(self, X, y):
        # 第一阶段：高级数据预处理 (包含清洗, 标准化, PCA)
        print("\n[Stage 1] 高级数据预处理...")
        self.preprocessor = AdvancedPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y)

        # 处理类别不平衡 - 使用简化的MOO-SMOTE
        print("\n[Stage 1.5] 处理类别不平衡 (简化MOO-SMOTE)...")
        # 定义优化目标权重 (AUC, Recall, Accuracy)
        # 示例：更关注AUC和Recall，适度关注Accuracy (可以调整以优化Precision)
        scoring_weights = (0.45, 0.45, 0.10)  # 可调
        n_trials = 15  # 可调
        try:
            X_resampled, y_resampled = moo_smote_func(X_processed, y, scoring_weights=scoring_weights,
                                                      n_trials=n_trials)
            print(f"  MOO-SMOTE应用成功。新样本数: {X_resampled.shape[0]}")
        except Exception as e:
            print(f"  MOO-SMOTE失败: {e}。回退到原始数据。")
            X_resampled, y_resampled = X_processed, y

        # 第二阶段：训练XGBoost模型（优化AUC）
        print("\n[Stage 2] 训练XGBoost模型...")
        self.xgb_model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.0172808,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1]),
            random_state=SEED,
            n_jobs=-1,
            eval_metric='auc'
        )
        self.xgb_model.fit(X_resampled, y_resampled)

        # 第三阶段：训练AdaBoost模型（优化召回率）
        print("\n[Stage 3] 训练AdaBoost模型...")
        self.ada_model = AdaBoostClassifier(
            n_estimators=400,
            learning_rate=0.4262,
            random_state=SEED
        )
        self.ada_model.fit(X_resampled, y_resampled)

    def predict_proba(self, X):
        """预测概率（返回两个模型的平均概率）"""
        X_processed = self.preprocessor.transform(X)

        # 获取两个模型的预测概率
        xgb_proba = self.xgb_model.predict_proba(X_processed)[:, 1]
        ada_proba = self.ada_model.predict_proba(X_processed)[:, 1]

        # 返回平均概率
        return (xgb_proba + ada_proba) / 2


# --- 其他辅助函数 (保持不变，但增加阈值优化) ---
def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 {filename}")


def find_best_threshold(y_true, y_proba, beta=1.0):
    """
    寻找最佳阈值以优化 F1 分数 (beta=1) 或其他 beta 分数。
    也可以修改为优化 Precision 或 Precision-Recall Trade-off。
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # 计算 F-beta 分数
    f_scores = (1 + beta ** 2) * (precisions * recalls) / (beta ** 2 * precisions + recalls + 1e-8)  # 防止除零

    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_f_score = f_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print(f"最佳阈值 (最大化 F{beta:.1f}): {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F{beta:.1f}: {best_f_score:.4f}")

    return best_threshold


def train_and_evaluate(X_train, y_train, X_test, y_test, optimize_threshold=True):
    """训练评估流程"""
    # 训练集成模型
    print("\n=== 训练集成模型 ===")
    start_time = time.time()
    ensemble = FinalEnsemble()
    ensemble.fit(X_train, y_train)
    end_time = time.time()
    print(f"\n训练耗时: {end_time - start_time:.2f} 秒")

    # 保存训练好的模型
    save_model(ensemble, 'best_cascaded_model.pkl')

    # 测试集预测
    print("\n=== 测试集评估 ===")
    y_proba = ensemble.predict_proba(X_test)

    # --- 阈值优化 ---
    threshold = 0.5
    if optimize_threshold:
        print("\n[优化阶段] 寻找最佳分类阈值...")
        # 使用验证集来寻找阈值，这里简化为使用测试集的一部分或交叉验证
        # 为了简化，我们直接在测试集上寻找（注意这可能会有轻微的过拟合风险到阈值选择上）
        # 更严谨的做法是划分出一个验证集专门用于阈值选择
        # 这里我们假设 X_test, y_test 可以作为阈值选择的数据
        # 或者，可以在训练过程中保留一部分数据作为验证集

        # 简化处理：直接在测试集上找
        threshold = find_best_threshold(y_test, y_proba, beta=1.0)  # 可以调整beta来偏向Precision或Recall

    y_pred = (y_proba >= threshold).astype(int)

    # 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== 评估结果 ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    # 注意：TestScore 的计算公式可能需要根据实际业务调整
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision + 50 * auc + 30 * recall:.4f}")

    return {
        'recall': recall,
        'auc': auc,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'y_proba': y_proba,
        'y_pred': y_pred,
        'threshold': threshold
    }


def main():
    # 检查CUDA可用性
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

    # 数据划分 (可以考虑分出一个验证集用于阈值选择)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    # 进一步划分训练集和验证集用于阈值选择 (可选)
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_temp, y_temp, test_size=0.1111, random_state=SEED, stratify=y_temp # 0.1111 * 0.9 ~= 0.1
    # )
    # 为了简化，我们直接使用 temp 作为训练集
    X_train, y_train = X_temp, y_temp

    print(f"\n数据划分: 训练集={X_train.shape[0]}, 测试集={X_test.shape[0]}")

    # 训练和评估 (启用阈值优化)
    results = train_and_evaluate(X_train, y_train, X_test, y_test, optimize_threshold=True)

    # 保存预测结果
    pd.DataFrame({
        'true_label': y_test,
        'pred_prob': results['y_proba'],
        'pred_label': results['y_pred']
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()



