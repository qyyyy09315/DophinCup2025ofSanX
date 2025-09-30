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


class ImprovedCascadeNaiveBayesWithDTFeatures(BaseEstimator, ClassifierMixin):
    """
    改进版：基于决策树思想的级联朴素贝叶斯模型。

    改进点：
    1. **模型结构优化**:
       - 允许配置 `dt_max_depth_per_layer` (建议 2-3) 以进行更复杂的分割。
       - 添加 `min_samples_split` 参数，防止过度碎片化。
       - 使用加权平均聚合概率，按子集样本量加权。
    2. **处理数据碎片化**:
       - 引入 `min_samples_split` 参数，在子集样本量低于此值时停止分裂。
       - 移除了空子集的虚拟分类器策略，因为 `min_samples_split` 应该能避免。
    3. **特征选择优化**:
       - 在每次分割前，使用互信息筛选特征 (`feature_selection_method='mutual_info'`)。
       - 保留了修改决策树 `criterion` 的可能性。
    """

    def __init__(self, n_layers=2, dt_max_depth_per_layer=2, nb_type='gaussian', random_state=42,
                 nb_params=None, min_samples_split=500, feature_selection_method='mutual_info',
                 feature_selection_threshold=0.01, dt_criterion='gini'):
        """
        初始化改进版级联决策树思想朴素贝叶斯。

        :param n_layers: 级联的层数 (建议 1-2)。
        :param dt_max_depth_per_layer: 每层用于分割的决策树的最大深度 (建议 2-3)。
        :param nb_type: 使用的朴素贝叶斯类型 ('gaussian' for GaussianNB)。
        :param random_state: 随机种子。
        :param nb_params: 传递给朴素贝叶斯分类器的参数字典。
                           示例 for GaussianNB: {'var_smoothing': 1e-9}
        :param min_samples_split: 分裂所需的最小样本数。如果子集样本数小于此值，则不再分裂。
        :param feature_selection_method: 分割前的特征选择方法 ('mutual_info', 'none')。
        :param feature_selection_threshold: 特征选择的阈值 (用于 'mutual_info')。
        :param dt_criterion: 决策树分割标准 ('gini', 'entropy', 'log_loss')。
        """
        self.n_layers = n_layers
        self.dt_max_depth_per_layer = dt_max_depth_per_layer
        self.nb_type = nb_type.lower()
        self.random_state = random_state
        self.nb_params = nb_params or {}
        self.min_samples_split = min_samples_split
        self.feature_selection_method = feature_selection_method
        self.feature_selection_threshold = feature_selection_threshold
        self.dt_criterion = dt_criterion

        if self.nb_type != 'gaussian':
            raise NotImplementedError("目前仅支持 'gaussian' 类型的朴素贝叶叶斯。")
        if self.feature_selection_method not in ['mutual_info', 'none']:
            raise ValueError("feature_selection_method 必须是 'mutual_info' 或 'none'。")

        self.layers = []
        # 用于存储训练时的原始特征维度，预测时需要
        self.initial_feature_count = None
        # 用于存储训练时的特征名称，用于特征选择
        self.feature_names = None

    def _get_nb_instance(self):
        """根据配置创建一个新的朴素贝叶斯实例"""
        if self.nb_type == 'gaussian':
            return GaussianNB(**self.nb_params)
        else:
            # This case should ideally not be reached due to check in __init__
            raise ValueError(f"不支持的朴素贝叶斯类型: {self.nb_type}")

    def _select_features(self, X_subset, y_subset):
        """
        根据配置选择特征。
        :param X_subset: 当前子集的特征。
        :param y_subset: 当前子集的标签。
        :return: (selected_indices, selected_X)
        """
        if self.feature_selection_method == 'none' or self.feature_names is None:
            return np.arange(X_subset.shape[1]), X_subset

        if self.feature_selection_method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            try:
                mi_scores = mutual_info_classif(X_subset, y_subset, random_state=self.random_state)
                selected_indices = np.where(mi_scores > self.feature_selection_threshold)[0]
                if len(selected_indices) == 0:  # Fallback if no features selected
                    selected_indices = np.argsort(mi_scores)[-10:]  # Select top 10
                return selected_indices, X_subset[:, selected_indices]
            except Exception as e:
                print(f"特征选择失败: {e}, 使用所有特征。")
                return np.arange(X_subset.shape[1]), X_subset
        return np.arange(X_subset.shape[1]), X_subset

    def fit(self, X, y):
        """
        训练改进版级联决策树思想朴素贝叶斯模型。

        :param X: 训练特征 (numpy array or pandas DataFrame)。
        :param y: 训练标签 (numpy array or pandas Series)。
        """
        self.initial_feature_count = X.shape[1]
        # Store feature names if available for feature selection
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 初始输入为原始特征和标签
        # 结构: [(features, labels, original_indices, sample_weight)]
        datasets_to_process = [(X.copy(), y.copy(), np.arange(len(y)), len(y))]
        self.layers = []  # 重置 layers 列表，确保 fit 是幂等的

        for layer_idx in range(self.n_layers):
            # print(f"训练第 {layer_idx + 1} 层...")

            current_layer_classifiers = []
            next_datasets = []

            # 遍历当前层需要处理的所有数据集
            for dataset_idx, (current_X, current_y, current_indices, current_weight) in enumerate(datasets_to_process):

                # 检查是否满足最小样本量要求
                if len(current_y) < self.min_samples_split:
                    # print(f"  数据集 {dataset_idx} 样本量 ({len(current_y)}) 小于阈值 ({self.min_samples_split}), 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    # 添加一个占位符 split_info，表示无需进一步分割
                    # 存储样本量用于加权
                    current_layer_classifiers.append((clf, None, current_weight))
                    continue

                # 如果当前子集只有一个类别，则直接训练一个对应类别的 NB 并跳过分割
                unique_labels = np.unique(current_y)
                if len(unique_labels) <= 1:
                    # print(f"  数据集 {dataset_idx} 只有一个类别 ({unique_labels}), 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    current_layer_classifiers.append((clf, None, current_weight))
                    continue

                # --- 特征选择 ---
                selected_feat_indices, X_selected = self._select_features(current_X, current_y)
                if X_selected.shape[1] == 0:
                    # print(f"  数据集 {dataset_idx} 特征选择后无特征, 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    current_layer_classifiers.append((clf, None, current_weight))
                    continue

                # 1. 使用指定深度和标准的决策树寻找最佳分割
                splitter_dt = DecisionTreeClassifier(
                    max_depth=self.dt_max_depth_per_layer,
                    criterion=self.dt_criterion,
                    random_state=self.random_state + layer_idx * 1000 + dataset_idx * 10
                )
                splitter_dt.fit(X_selected, current_y)

                # 检查是否实际发生了分割
                if splitter_dt.tree_.children_left[0] == splitter_dt.tree_.children_right[0]:
                    # print(f"  数据集 {dataset_idx} 上决策树未能产生有效分割, 直接训练 NB.")
                    clf = self._get_nb_instance()
                    clf.fit(current_X, current_y)
                    current_layer_classifiers.append((clf, None, current_weight))
                    continue

                # 获取根节点的分割信息
                split_feature_local_idx = splitter_dt.tree_.feature[0]
                # 映射回原始特征索引
                split_feature_global_idx = selected_feat_indices[split_feature_local_idx]
                split_threshold = splitter_dt.tree_.threshold[0]

                # 2. 根据分割点划分数据 (使用原始特征)
                left_mask = current_X[:, split_feature_global_idx] <= split_threshold
                right_mask = ~left_mask

                X_left, y_left, indices_left = current_X[left_mask], current_y[left_mask], current_indices[left_mask]
                X_right, y_right, indices_right = current_X[right_mask], current_y[right_mask], current_indices[
                    right_mask]

                # 3. 在左右子集上分别训练朴素贝叶斯分类器
                clf_left = self._get_nb_instance()
                clf_right = self._get_nb_instance()

                # 处理空子集的情况 (理论上 min_samples_split 应该能避免，但作为安全检查)
                if X_left.size > 0:
                    clf_left.fit(X_left, y_left)
                else:
                    # print(f"警告: 第 {layer_idx + 1} 层, 数据集 {dataset_idx} 左子集为空。")
                    #  fallback to parent data
                    clf_left.fit(current_X, current_y)

                if X_right.size > 0:
                    clf_right.fit(X_right, y_right)
                else:
                    # print(f"警告: 第 {layer_idx + 1} 层, 数据集 {dataset_idx} 右子集为空。")
                    #  fallback to parent data
                    clf_right.fit(current_X, current_y)

                # 4. 保存分类器和分割信息及样本量
                split_info = {
                    'feature': split_feature_global_idx,
                    'threshold': split_threshold,
                    'selected_features': selected_feat_indices.tolist()  # 可选存储，预测时可能用到
                }
                # 存储子集样本量用于加权
                weight_left = len(y_left)
                weight_right = len(y_right)
                current_layer_classifiers.append(([clf_left, clf_right], split_info, current_weight))

                # 5. 将子集添加到下一轮待处理列表
                if X_left.size > 0 and len(y_left) >= self.min_samples_split:
                    next_datasets.append((X_left, y_left, indices_left, weight_left))
                elif X_left.size > 0:  # 存在但不满足分裂条件，直接存为终端节点
                    clf_terminal_left = self._get_nb_instance()
                    clf_terminal_left.fit(X_left, y_left)
                    # 在预测时需要知道它是一个终端节点，这里用 None 标记 split_info
                    current_layer_classifiers.append((clf_terminal_left, None, weight_left))

                if X_right.size > 0 and len(y_right) >= self.min_samples_split:
                    next_datasets.append((X_right, y_right, indices_right, weight_right))
                elif X_right.size > 0:  # 存在但不满足分裂条件，直接存为终端节点
                    clf_terminal_right = self._get_nb_instance()
                    clf_terminal_right.fit(X_right, y_right)
                    current_layer_classifiers.append((clf_terminal_right, None, weight_right))

            # 6. 保存当前层
            self.layers.append(current_layer_classifiers)
            # 7. 更新下一轮要处理的数据集
            datasets_to_process = next_datasets

            # print(f"第 {layer_idx + 1} 层训练完成，产生了 {len(datasets_to_process)} 个新子集。")

        # print("改进版级联决策树思想朴素贝叶斯训练完成。")
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
        # 结构: (indices, layer_index, accumulated_weight)
        processing_stack = [(np.arange(X.shape[0]), 0, 1.0)]

        while processing_stack:
            current_indices, l_idx, accumulated_weight = processing_stack.pop()

            if l_idx >= len(self.layers) or current_indices.size == 0:
                continue

            layer_models = self.layers[l_idx]

            for model_group, split_details, model_weight in layer_models:
                effective_weight = accumulated_weight * model_weight

                if split_details is None:  # Terminal model (no split or min_samples not met)
                    sub_X = X[current_indices]
                    try:
                        p = model_group.predict_proba(sub_X)
                    except Exception:
                        p = np.zeros((sub_X.shape[0], 2))  # Assuming binary
                        p[:, 1] = 0.5
                        # Weight the probabilities
                    weighted_p = p * effective_weight
                    all_proba_predictions.append((current_indices, weighted_p, effective_weight))

                else:  # Normal split-based pair of models
                    feat_idx = split_details['feature']
                    thresh = split_details['threshold']

                    sub_X = X[current_indices]

                    left_child_mask_local = sub_X[:, feat_idx] <= thresh
                    right_child_mask_local = ~left_child_mask_local

                    left_indices = current_indices[left_child_mask_local]
                    right_indices = current_indices[right_child_mask_local]

                    if left_indices.size > 0:
                        try:
                            p_left = model_group[0].predict_proba(X[left_indices])
                        except Exception:
                            p_left = np.zeros((left_indices.shape[0], 2))
                            p_left[:, 1] = 0.5
                        weighted_p_left = p_left * effective_weight
                        # Push left child to stack for next layer
                        processing_stack.append((left_indices, l_idx + 1, effective_weight))

                    if right_indices.size > 0:
                        try:
                            p_right = model_group[1].predict_proba(X[right_indices])
                        except Exception:
                            p_right = np.zeros((right_indices.shape[0], 2))
                            p_right[:, 1] = 0.5
                        weighted_p_right = p_right * effective_weight
                        # Push right child to stack for next layer
                        processing_stack.append((right_indices, l_idx + 1, effective_weight))

        # Aggregate probabilities: Sum weighted predictions and weights for each sample
        if not all_proba_predictions:
            return np.ones((X.shape[0], 2)) * 0.5

        final_probas = np.zeros((X.shape[0], all_proba_predictions[0][1].shape[1]))
        total_weights = np.zeros(X.shape[0])

        for indices, weighted_probas, weight in all_proba_predictions:
            final_probas[indices] += weighted_probas
            total_weights[indices] += weight

        # Avoid division by zero
        total_weights[total_weights == 0] = 1
        final_probas /= total_weights.reshape(-1, 1)

        # 确保概率和为1
        row_sums = final_probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero again
        final_probas /= row_sums

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
    """整合预处理 (插补, 缩放) 和 ImprovedCascadeNaiveBayesWithDTFeatures 的管道"""

    def __init__(self, imputer=None, scaler=None, model=None):
        self.imputer = imputer or SimpleImputer(strategy="mean")
        self.scaler = scaler or StandardScaler()
        # 默认模型使用 改进版 CascadeNaiveBayesWithDTFeatures
        self.model = model or ImprovedCascadeNaiveBayesWithDTFeatures()

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
    print("开始训练：SMOTEENN -> 标准化 -> 改进版 级联 决策树思想朴素贝叶斯")
    print("优化方向: 1. 先采样再训练; 2. 使用改进的级联 NB; 3. PR曲线拐点阈值")
    print("=" * 70)

    # ----- 配置 -----
    # 注意：请确保 './clean.csv' 文件存在于当前工作目录或提供完整路径
    data_path = "./clean.csv"
    test_size = 0.10
    random_state = 42

    # --- 改进后的模型超参数 ---
    improved_params = {
        'n_layers': 2,  # 减少层数
        'dt_max_depth_per_layer': 2,  # 增加决策树深度
        'nb_type': 'gaussian',
        'nb_params': {'var_smoothing': 1e-9},
        'min_samples_split': 500,  # 添加最小样本量约束
        'feature_selection_method': 'mutual_info',  # 启用特征选择
        'feature_selection_threshold': 0.01,
        'dt_criterion': 'entropy'  # 修改决策树标准
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

    # ----- 4. 使用改进参数训练模型 -----
    print("--- 使用改进参数训练最终模型 (已重采样和标准化数据) ---")
    print(f"使用的改进参数: {improved_params}")

    # 1. 使用改进参数创建最终模型
    final_improved_cascade_nb = ImprovedCascadeNaiveBayesWithDTFeatures(
        n_layers=improved_params.get('n_layers', 2),
        dt_max_depth_per_layer=improved_params.get('dt_max_depth_per_layer', 2),
        nb_type=improved_params.get('nb_type', 'gaussian'),
        nb_params=improved_params.get('nb_params', {}),
        min_samples_split=improved_params.get('min_samples_split', 500),
        feature_selection_method=improved_params.get('feature_selection_method', 'none'),
        feature_selection_threshold=improved_params.get('feature_selection_threshold', 0.0),
        dt_criterion=improved_params.get('dt_criterion', 'gini'),
        random_state=random_state
    )

    # 2. 训练 改进版 Cascade Naive Bayes
    print("训练最终改进版级联决策树思想朴素贝叶斯模型 (使用改进参数、重采样和标准化数据) ...")
    final_improved_cascade_nb.fit(X_resampled, y_resampled)

    # 3. 组装最终管道
    final_model_improved = ModelPipeline(
        imputer=initial_imputer,
        scaler=initial_scaler,
        model=final_improved_cascade_nb
    )

    # ----- 5. 在 Holdout 集上评估最终模型 (多种阈值策略) -----
    print("\n在 Holdout 集上评估最终改进模型...")
    # 获取最后一层的概率预测 (对于二分类，我们通常取正类概率)
    holdout_probas = final_model_improved.predict_proba(X_holdout)[:, 1]

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
        "imputer": final_model_improved.imputer,
        "scaler": final_model_improved.scaler,
        "model": final_model_improved.model,
        "threshold_youden": float(best_thresh_youden),
        "threshold_pr_elbow": float(best_thresh_pr),
        "improved_params": improved_params
    }
    save_object(model_dict, "./model_pipeline_improved_cascade_nb_dt.pkl")
    print("\n已保存包含改进版级联决策树思想朴素贝叶斯的完整模型管道 ./model_pipeline_improved_cascade_nb_dt.pkl")
    print("=" * 70)




