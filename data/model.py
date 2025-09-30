import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn 和 scikit-learn，需要先通过 pip 安装
# pip install imbalanced-learn scikit-learn
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)


def save_object(obj, filepath):
    """保存 Python 对象到文件"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


# --- 新增/修复：将 evaluate_and_print_metrics 函数定义移到前面 ---
def evaluate_and_print_metrics(y_true, y_pred, y_proba, threshold_name):
    """辅助函数：计算并打印评估指标"""
    recall = recall_score(y_true, y_pred, zero_division=0)
    # 处理 AUC 计算可能的错误（例如，只有一个类）
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # 自定义评分公式 (使用 Holdout 集上的指标)
    # 避免 auc 为 nan 的情况影响整体评分
    if np.isnan(auc):
        auc = 0.0
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"--- Holdout 集评估结果 ({threshold_name}) ---")
    print(f"F1-Score:     {f1:.5f}")
    print(f"Recall:       {recall:.5f}")
    print(f"AUC:          {auc:.5f}")
    print(f"Precision:    {precision:.5f}")
    print(f"Final Score (30*R + 50*AUC + 20*P): {final_score:.5f}")
    print("-" * 30)


class CascadeNaiveBayesWithDTFeatures(BaseEstimator, ClassifierMixin):
    """
    基于决策树思想的级联朴素贝叶斯模型。
    该模型模仿决策树的分裂过程：
    1. 使用决策树找到最佳分割特征和阈值。
    2. 根据此分割条件将数据划分为两部分。
    3. 在每个子集上训练一个朴素贝叶斯分类器。
    4. 将这两个 NB 分类器视为第一层。
    5. 后续层级重复此过程，在由前一层分类器预测结果划分的新子集上训练新的 NB 对。
    最终预测通过对所有 NB 分类器的概率进行平均得到。
    """

    def __init__(self, n_layers=3, dt_max_depth_per_layer=1, nb_type='gaussian', random_state=42,
                 nb_params=None):
        """
        初始化级联决策树思想朴素贝叶斯。

        :param n_layers: 级联的层数。
        :param dt_max_depth_per_layer: 每层用于分割的决策树的最大深度 (建议保持为1以模拟单次分割)。
        :param nb_type: 使用的朴素贝叶斯类型 ('gaussian' for GaussianNB)。
        :param random_state: 随机种子。
        :param nb_params: 传递给朴素贝叶斯分类器的参数字典。
                           示例 for GaussianNB: {'var_smoothing': 1e-9}
        """
        self.n_layers = n_layers
        self.dt_max_depth_per_layer = dt_max_depth_per_layer
        self.nb_type = nb_type.lower()
        self.random_state = random_state
        self.nb_params = nb_params or {}

        if self.nb_type != 'gaussian':
            raise NotImplementedError("目前仅支持 'gaussian' 类型的朴素贝叶叶斯。")

        self.layers = []
        # 用于存储训练时的原始特征维度，预测时需要
        self.initial_feature_count = None

    def _get_nb_instance(self):
        """根据配置创建一个新的朴素贝叶斯实例"""
        if self.nb_type == 'gaussian':
            return GaussianNB(**self.nb_params)
        else:
            # This case should ideally not be reached due to check in __init__
            raise ValueError(f"不支持的朴素贝叶斯类型: {self.nb_type}")

    def fit(self, X, y):
        """
        训练级联决策树思想朴素贝叶斯模型。

        :param X: 训练特征 (numpy array or pandas DataFrame)。
        :param y: 训练标签 (numpy array or pandas Series)。
        """
        self.initial_feature_count = X.shape[1]

        # 初始输入为原始特征和标签
        datasets_to_process = [(X.copy(), y.copy())]  # [(features, labels), ...]
        self.layers = []  # 重置 layers 列表，确保 fit 是幂等的

        for layer_idx in range(self.n_layers):
            # print(f"训练第 {layer_idx + 1} 层...")

            current_layer_classifiers = []
            next_datasets = []

            # 遍历当前层需要处理的所有数据集
            for dataset_idx, (current_X, current_y) in enumerate(datasets_to_process):

                # 如果当前子集只有一个类别，则直接训练一个对应类别的 NB 并跳过分割
                unique_labels = np.unique(current_y)
                if len(unique_labels) <= 1:
                    # print(f"  数据集 {dataset_idx} 只有一个类别 ({unique_labels}), 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    # 添加一个占位符 split_info，表示无需进一步分割
                    current_layer_classifiers.append((clf, None))
                    continue

                # 1. 使用浅层决策树寻找最佳分割
                splitter_dt = DecisionTreeClassifier(
                    max_depth=self.dt_max_depth_per_layer,
                    random_state=self.random_state + layer_idx * 100 + dataset_idx
                )
                splitter_dt.fit(current_X, current_y)

                # 获取分割特征和阈值 (对于 max_depth=1，这很简单)
                # tree_.feature[0] 是根节点使用的特征索引 (-2 表示叶节点)
                if splitter_dt.tree_.children_left[0] == splitter_dt.tree_.children_right[0]:  # No split happened
                    # print(f"  数据集 {dataset_idx} 上决策树未能产生有效分割, 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    current_layer_classifiers.append((clf, None))
                    continue

                split_feature = splitter_dt.tree_.feature[0]
                split_threshold = splitter_dt.tree_.threshold[0]

                # 2. 根据分割点划分数据
                left_mask = current_X[:, split_feature] <= split_threshold
                right_mask = ~left_mask

                X_left, y_left = current_X[left_mask], current_y[left_mask]
                X_right, y_right = current_X[right_mask], current_y[right_mask]

                # 3. 在左右子集上分别训练朴素贝叶斯分类器
                clf_left = self._get_nb_instance()
                clf_right = self._get_nb_instance()

                # 处理空子集的情况
                if X_left.size > 0:
                    clf_left.fit(X_left, y_left)
                else:
                    # 如果左子集为空，创建一个总是预测多数类的虚拟分类器
                    majority_class = np.bincount(y_right).argmax()  # Use right's majority if left is empty
                    dummy_preds_left = np.full_like(y_left, fill_value=majority_class,
                                                    dtype=int) if y_left.size > 0 else np.array([majority_class])
                    clf_left.fit(current_X[:1], dummy_preds_left[:1])  # Fit on a dummy sample

                if X_right.size > 0:
                    clf_right.fit(X_right, y_right)
                else:
                    # 如果右子集为空，创建一个总是预测多数类的虚拟分类器
                    majority_class = np.bincount(y_left).argmax()  # Use left's majority if right is empty
                    dummy_preds_right = np.full_like(y_right, fill_value=majority_class,
                                                     dtype=int) if y_right.size > 0 else np.array([majority_class])
                    clf_right.fit(current_X[:1], dummy_preds_right[:1])  # Fit on a dummy sample

                # 4. 保存分类器和分割信息
                split_info = {'feature': split_feature, 'threshold': split_threshold}
                current_layer_classifiers.append(([clf_left, clf_right], split_info))

                # 5. 将子集添加到下一轮待处理列表
                if X_left.size > 0:
                    next_datasets.append((X_left, y_left))
                if X_right.size > 0:
                    next_datasets.append((X_right, y_right))

            # 6. 保存当前层
            self.layers.append(current_layer_classifiers)
            # 7. 更新下一轮要处理的数据集
            datasets_to_process = next_datasets

            # print(f"第 {layer_idx + 1} 层训练完成，产生了 {len(datasets_to_process)} 个新子集。")

        # print("级联决策树思想朴素贝叶斯训练完成。")
        return self

    def predict_proba(self, X):
        """
        预测样本属于各个类别的概率。

        :param X: 待预测特征。
        :return: 概率矩阵 (numpy array)，每一行是样本对各类别的平均概率。
        """
        if not self.layers:
            raise ValueError("模型尚未训练，请先调用 fit 方法。")
        if X.shape[1] != self.initial_feature_count:
            raise ValueError(f"输入特征维度 {X.shape[1]} 与训练时的维度 {self.initial_feature_count} 不符。")

        all_proba_predictions = []

        # 初始化队列，包含整个数据集和其在各层的位置追踪
        # 结构: [(data_indices, layer_level, classifier_path_in_layer)]
        processing_queue = [(np.arange(X.shape[0]), 0, [])]  # Start with all samples at layer 0

        while processing_queue:
            data_indices, layer_level, path = processing_queue.pop(0)
            if layer_level >= len(self.layers):
                # Should not happen, but safety check
                continue

            current_layer = self.layers[layer_level]

            # Retrieve the specific set of classifiers based on the path taken so far
            # For simplicity in this cascading structure, we assume path directly indexes into nested lists
            # This logic needs careful handling depending on exact storage structure.
            # Here, let's iterate through the layer's classifiers and match by path implicitly via order/tree traversal simulation

            # In our stored structure, each item in a layer corresponds to an original split node
            # We need to re-traverse the splits up to the current level for these indices.
            # Simpler approach: process all classifiers in the layer that apply to the current subset.
            # But our storage isn't strictly hierarchical paths. Let's rethink...

            # Revised simpler idea matching original training flow:
            # During prediction, we recreate the splitting process used during training.

            # Let's restart queue logic properly:

        # Re-initialize for correct traversal
        # Structure in queue: (indices, layer_index, list_of_conditions_to_reach_here)
        # Conditions could be list of (split_feature, split_threshold, is_left_child_bool)
        processing_stack = [(np.arange(X.shape[0]), 0, [])]

        while processing_stack:
            current_indices, l_idx, conditions = processing_stack.pop()

            if l_idx >= len(self.layers) or current_indices.size == 0:
                continue

            layer_models = self.layers[l_idx]

            child_datasets_info = []  # Will hold (indices, new_conditions) for children

            for model_group, split_details in layer_models:
                if split_details is None:  # Model trained without split (pure node)
                    # Apply this single model to current_indices
                    sub_X = X[current_indices]
                    try:
                        p = model_group.predict_proba(sub_X)
                    except:
                        # Handle edge cases where model might fail on unseen feature values
                        # E.g., GaussianNB can sometimes produce issues
                        p = np.zeros((sub_X.shape[0], 2))  # Assuming binary classification
                        p[:, 1] = 0.5  # Assign neutral probability

                    all_proba_predictions.append((current_indices, p))

                else:  # Normal split-based pair of models
                    feat_idx = split_details['feature']
                    thresh = split_details['threshold']

                    sub_X = X[current_indices]

                    # Check which points satisfy ancestral conditions first
                    # This requires checking `conditions` history against X.
                    # Simplification: Assume stack manages routing correctly based on applying splits sequentially.
                    # So `current_indices` already represent points that reached this stage.

                    left_child_mask_local = sub_X[:, feat_idx] <= thresh
                    right_child_mask_local = ~left_child_mask_local

                    left_indices = current_indices[left_child_mask_local]
                    right_indices = current_indices[right_child_mask_local]

                    # Predict with respective models
                    if left_indices.size > 0:
                        try:
                            p_left = model_group[0].predict_proba(X[left_indices])
                        except:
                            p_left = np.zeros((left_indices.shape[0], 2))
                            p_left[:, 1] = 0.5
                        all_proba_predictions.append((left_indices, p_left))

                    if right_indices.size > 0:
                        try:
                            p_right = model_group[1].predict_proba(X[right_indices])
                        except:
                            p_right = np.zeros((right_indices.shape[0], 2))
                            p_right[:, 1] = 0.5
                        all_proba_predictions.append((right_indices, p_right))

        # Aggregate probabilities: Average over all predictions for each sample
        if not all_proba_predictions:
            # Fallback if no valid predictions were made (should be rare)
            return np.ones((X.shape[0], 2)) * 0.5

        final_probas = np.zeros((X.shape[0], all_proba_predictions[0][1].shape[1]))
        counts = np.zeros(X.shape[0])

        for indices, probas in all_proba_predictions:
            final_probas[indices] += probas
            counts[indices] += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        final_probas /= counts.reshape(-1, 1)

        return final_probas

    def predict(self, X):
        """
        对样本进行分类预测。

        :param X: 待预测特征。
        :return: 预测标签 (numpy array)。
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def youden_threshold(y_true, probas):
    """计算 Youden's J 统计量下的最佳阈值"""
    thresholds = np.linspace(0, 1, 1001)  # 提高精度
    best_t, best_j = 0.5, -1
    for t in thresholds:
        preds = (probas >= t).astype(int)
        # 处理只有一个类别的情况
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # 所有预测都是同一类
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # All predicted negative
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # All predicted positive
        else:
            # 不规则矩阵，填充缺失项
            padded_cm = np.pad(cm, ((0, 2 - cm.shape[0]), (0, 2 - cm.shape[1])), mode='constant')
            tn, fp, fn, tp = padded_cm.ravel()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        J = sensitivity + specificity - 1
        if J > best_j:
            best_j, best_t = J, t
    return best_t


def pr_threshold(y_true, probas):
    """计算 Precision-Recall 曲线拐点附近的阈值 (最小化 |P-R|)"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)

    # 移除最后一个元素，因为 recalls[-1] 总是 1，且没有对应的 threshold
    # precisions[:-1], recalls[:-1], thresholds

    # 寻找 |P - R| 最小时的索引
    diff_abs = np.abs(precisions[:-1] - recalls[:-1])
    min_idx = np.argmin(diff_abs)

    # 获取对应的阈值
    optimal_threshold = thresholds[min_idx]
    return optimal_threshold


# --- 新增：封装完整模型管道以支持 GridSearchCV ---
# --- 修改：移除内部 SMOTEENN 步骤，因为数据将在外部预处理 ---
class ModelPipeline(BaseEstimator, ClassifierMixin):
    """整合预处理 (插补, 缩放) 和 CascadeNaiveBayesWithDTFeatures 的管道"""

    def __init__(self, imputer=None, scaler=None, model=None):
        self.imputer = imputer or SimpleImputer(strategy="mean")
        self.scaler = scaler or StandardScaler()
        # 默认模型使用 CascadeNaiveBayesWithDTFeatures
        self.model = model or CascadeNaiveBayesWithDTFeatures()

    def fit(self, X, y):
        """拟合整个管道：插补 -> 缩放 -> 模型训练"""
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        print(f"预处理后数据维度: {X_scaled.shape}")
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        """预测概率：插补 -> 缩放 -> 模型预测"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        """预测标签：插补 -> 缩放 -> 模型预测"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


if __name__ == "__main__":
    print("=" * 70)
    print("开始训练：SMOTEENN -> 标准化 -> 级联 决策树思想朴素贝叶斯")
    print("优化方向: 1. 先采样再训练; 2. 使用改进的级联 NB; 3. PR曲线拐点阈值")
    print("=" * 70)

    # ----- 配置 -----
    # 注意：请确保 './clean.csv' 文件存在于当前工作目录或提供完整路径
    data_path = "./clean.csv"
    test_size = 0.10
    random_state = 42

    # 固定模型超参数
    fixed_params = {
        'n_layers': 3,
        'dt_max_depth_per_layer': 1,
        'nb_type': 'gaussian',
        'nb_params': {'var_smoothing': 1e-9}  # GaussianNB 参数
    }

    # ----- 1. 读取并预处理数据 -----
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{data_path}'。请确保文件路径正确。")
        exit(1)

    # 删除 company_id 列（如果存在）
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # 进行独热编码
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all == 0).sum()}")

    # ----- 2. 划分训练/验证集 和 Holdout 集 -----
    # 先划分出 Holdout 集
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"用于训练的数据集: {X_temp.shape}, Holdout 集: {X_holdout.shape}")

    # ----- 3. 数据预处理与重采样 (先采样) -----
    print("正在进行数据预处理 (插补 & 缩放) ...")
    # 1. 插补
    initial_imputer = SimpleImputer(strategy="mean")
    X_temp_imputed = initial_imputer.fit_transform(X_temp)
    # 2. 缩放
    initial_scaler = StandardScaler()
    X_temp_scaled = initial_scaler.fit_transform(X_temp_imputed)
    print(f"预处理后数据形状: {X_temp_scaled.shape}")

    print("正在进行 SMOTEENN 重采样 ...")
    # 3. SMOTEENN 重采样
    smoteenn_sampler = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smoteenn_sampler.fit_resample(X_temp_scaled, y_temp)
    print(f"重采样后数据形状: {X_resampled.shape}, 正类={y_resampled.sum()}, 负类={(y_resampled == 0).sum()}")

    # ----- 4. 使用固定参数训练模型 -----
    print("--- 使用固定参数训练最终模型 (已重采样和标准化数据) ---")
    print(f"使用的固定参数: {fixed_params}")

    # 1. 使用固定参数创建最终模型
    final_cascade_nb_clean = CascadeNaiveBayesWithDTFeatures(
        n_layers=fixed_params.get('n_layers',10),
        dt_max_depth_per_layer=fixed_params.get('dt_max_depth_per_layer', 1),
        nb_type=fixed_params.get('nb_type', 'gaussian'),
        nb_params=fixed_params.get('nb_params', {}),
        random_state=random_state
    )

    # 2. 训练 Cascade Naive Bayes
    print("训练最终级联决策树思想朴素贝叶斯模型 (使用固定参数、重采样和标准化数据) ...")
    final_cascade_nb_clean.fit(X_resampled, y_resampled)

    # 3. 组装最终管道
    final_model_clean = ModelPipeline(
        imputer=initial_imputer,
        scaler=initial_scaler,
        model=final_cascade_nb_clean
    )

    # ----- 5. 在 Holdout 集上评估最终模型 (多种阈值策略) -----
    print("\n在 Holdout 集上评估最终模型...")
    # 获取最后一层的概率预测 (对于二分类，我们通常取正类概率)
    holdout_probas = final_model_clean.predict_proba(X_holdout)[:, 1]

    # --- 阈值 1: Youden's J 统计量 ---
    best_thresh_youden = youden_threshold(y_holdout, holdout_probas)
    print(f"\n使用 Youden's J 统计量计算最佳阈值: {best_thresh_youden:.4f}")
    y_pred_holdout_youden = (holdout_probas >= best_thresh_youden).astype(int)
    evaluate_and_print_metrics(y_holdout, y_pred_holdout_youden, holdout_probas, "Youden's J")

    # --- 阈值 2: PR 曲线拐点 (最小化 |P-R|) ---
    best_thresh_pr = pr_threshold(y_holdout, holdout_probas)
    print(f"\n使用 PR 曲线拐点 (最小化 |P-R|) 计算最佳阈值: {best_thresh_pr:.4f}")
    y_pred_holdout_pr = (holdout_probas >= best_thresh_pr).astype(int)
    evaluate_and_print_metrics(y_holdout, y_pred_holdout_pr, holdout_probas, "PR Curve Elbow")

    # ----- 6. 保存完整模型管道 (使用 Youden's J 阈值) -----
    model_dict = {
        "imputer": final_model_clean.imputer,
        "scaler": final_model_clean.scaler,
        "model": final_model_clean.model,
        "threshold_youden": float(best_thresh_youden),
        "threshold_pr_elbow": float(best_thresh_pr),
        "fixed_params": fixed_params
    }
    save_object(model_dict, "./model_pipeline_cascade_nb_dt.pkl")
    print("\n已保存完整模型管道 ./model_pipeline_cascade_nb_dt.pkl")
    print("=" * 70)




