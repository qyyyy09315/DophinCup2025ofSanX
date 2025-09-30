import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # 新增 PCA
from sklearn.base import BaseEstimator, ClassifierMixin

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


class CascadeRandomForest(BaseEstimator, ClassifierMixin):
    """
    级联随机森林模型。
    通过逐层训练随机森林，并将预测概率作为新特征传递给下一层，
    实现层次化的特征学习。
    """

    def __init__(self, n_layers=3, n_estimators=100, max_depth=None, random_state=42, class_weight_option='balanced'):
        """
        初始化级联随机森林。

        :param n_layers: 级联的层数。
        :param n_estimators: 每层中随机森林的树的数量。
        :param max_depth: 每棵树的最大深度。
        :param random_state: 随机种子。
        :param class_weight_option: 'balanced', None, 或字典形式的类别权重 (e.g., {0:1, 1:5})
                                    用于传递给底层 RandomForestClassifier 以实现代价敏感学习。
        """
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight_option = class_weight_option  # 新增参数
        self.layers = []
        # 用于存储训练时的原始特征维度，预测时需要
        self.initial_feature_count = None

    def fit(self, X, y):
        """
        训练级联随机森林模型。

        :param X: 训练特征 (numpy array or pandas DataFrame)。
        :param y: 训练标签 (numpy array or pandas Series)。
        """
        self.initial_feature_count = X.shape[1]
        X_current = X.copy()  # 避免修改原始输入
        self.layers = []  # 重置 layers 列表，确保 fit 是幂等的
        for i in range(self.n_layers):
            # 为了增加多样性，可以对每层使用不同的随机种子
            # 或者调整其他超参数
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,  # 改变随机种子
                n_jobs=-1,
                class_weight=self.class_weight_option  # 应用代价敏感学习
            )
            # print(f"训练第 {i + 1} 层随机森林...") # 为简洁可关闭此打印
            rf.fit(X_current, y)
            self.layers.append(rf)

            # 将原始特征与当前层的概率预测拼接，作为下一层的输入
            probas = rf.predict_proba(X_current)
            X_current = np.hstack([X, probas])  # 始终与原始特征拼接

        # print("级联随机森林训练完成。")
        return self  # 符合 sklearn 接口

    def predict_proba(self, X):
        """
        预测样本属于各个类别的概率。

        :param X: 待预测特征。
        :return: 概率矩阵 (numpy array)。
        """
        if not self.layers:
            raise ValueError("模型尚未训练，请先调用 fit 方法。")
        if X.shape[1] != self.initial_feature_count:
            raise ValueError(f"输入特征维度 {X.shape[1]} 与训练时的维度 {self.initial_feature_count} 不符。")

        X_current = X.copy()
        for i, rf in enumerate(self.layers):
            # 将原始特征与当前层的概率预测拼接
            probas = rf.predict_proba(X_current)
            if i < len(self.layers) - 1:  # 最后一层不需要拼接，直接输出
                X_current = np.hstack([X, probas])
        # 返回最后一层的概率预测
        return probas

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
# --- 修改：加入 PCA 步骤 ---
class ModelPipeline(BaseEstimator, ClassifierMixin):
    """整合预处理 (插补, 缩放, PCA) 和 CascadeRandomForest 的管道 (移除了内部 SMOTEENN)"""

    def __init__(self, imputer=None, scaler=None, pca=None, model=None): # 新增 pca 参数
        self.imputer = imputer or SimpleImputer(strategy="mean")
        self.scaler = scaler or StandardScaler()
        self.pca = pca or PCA(n_components=0.95) # 默认保留 95% 方差 # 新增 PCA
        # 默认模型使用 class_weight=None，网格搜索会覆盖
        self.model = model or CascadeRandomForest(class_weight_option=None)

    def fit(self, X, y):
        """拟合整个管道：插补 -> 缩放 -> PCA -> 模型训练"""
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        X_pca = self.pca.fit_transform(X_scaled) # 新增 PCA 拟合和变换
        print(f"PCA 后数据维度: {X_pca.shape}")
        self.model.fit(X_pca, y) # 在 PCA 变换后的数据上训练模型
        return self

    def predict_proba(self, X):
        """预测概率：插补 -> 缩放 -> PCA -> 模型预测"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled) # 新增 PCA 变换
        return self.model.predict_proba(X_pca) # 在 PCA 变换后的数据上预测

    def predict(self, X):
        """预测标签：插补 -> 缩放 -> PCA -> 模型预测"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


if __name__ == "__main__":
    print("=" * 70)
    print("开始训练：SMOTEENN -> PCA -> 级联随机森林 (Cascade RF) (无网格搜索)")
    print("优化方向: 1. 先采样再训练; 2. 加入 PCA 降维; 3. PR曲线拐点阈值")
    print("=" * 70)

    # ----- 配置 -----
    # 注意：请确保 './clean.csv' 文件存在于当前工作目录或提供完整路径
    data_path = "./clean.csv"
    test_size = 0.10
    random_state = 42

    # 固定模型超参数 (取消网格搜索)
    fixed_params = {
        'n_layers': 3,
        'n_estimators': 100,
        'max_depth': None,
        'class_weight_option': 'balanced'
    }
    pca_n_components = 0.95 # PCA 保留的方差比例

    # ----- 1. 读取并预处理数据 -----
    # 注意：此部分需要一个名为 'clean.csv' 的有效文件
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{data_path}'。请确保文件路径正确。")
        # 如果没有文件，可以创建一个示例数据集用于演示

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

    # ----- 4. PCA 降维 (在重采样后的数据上) -----
    print(f"正在进行 PCA 降维 (保留 {pca_n_components*100:.1f}% 方差) ...")
    pca_transformer = PCA(n_components=pca_n_components)
    X_resampled_pca = pca_transformer.fit_transform(X_resampled)
    print(f"PCA 后数据形状: {X_resampled_pca.shape}")
    n_components_retained = pca_transformer.n_components_
    print(f"保留的主成分数量: {n_components_retained}")

    # ----- 5. 使用固定参数训练模型 -----
    print("--- 使用固定参数训练最终模型 (已重采样和PCA数据) ---")
    print(f"使用的固定参数: {fixed_params}")

    # 1. 使用固定参数创建最终模型
    final_cascade_rf_clean = CascadeRandomForest(
        n_layers=fixed_params.get('n_layers', 5),
        n_estimators=fixed_params.get('n_estimators', 200),
        max_depth=fixed_params.get('max_depth', None),
        random_state=random_state,
        class_weight_option=fixed_params.get('class_weight_option', 'balanced')
    )

    # 2. 训练 CascadeRF (在重采样和PCA后的数据上)
    print("训练最终级联随机森林模型 (使用固定参数、重采样和PCA数据) ...")
    final_cascade_rf_clean.fit(X_resampled_pca, y_resampled)

    # 3. 组装最终管道 (包含原始的 imputer, scaler, pca)
    final_model_clean = ModelPipeline(
        imputer=initial_imputer,   # 使用之前训练好的 imputer
        scaler=initial_scaler,     # 使用之前训练好的 scaler
        pca=pca_transformer,       # 使用训练好的 PCA 变换器
        model=final_cascade_rf_clean # 使用训练好的模型
    )

    # ----- 6. 在 Holdout 集上评估最终模型 (多种阈值策略) -----
    print("\n在 Holdout 集上评估最终模型...")
    # 获取最后一层的概率预测 (对于二分类，我们通常取正类概率)
    holdout_probas = final_model_clean.predict_proba(X_holdout)[:, 1]

    # --- 阈值 1: Youden's J 统计量 ---
    best_thresh_youden = youden_threshold(y_holdout, holdout_probas)
    print(f"\n使用 Youden's J 统计量计算最佳阈值: {best_thresh_youden:.4f}")
    y_pred_holdout_youden = (holdout_probas >= best_thresh_youden).astype(int)
    evaluate_and_print_metrics(y_holdout, y_pred_holdout_youden, holdout_probas, "Youden's J") # 调用已修复的函数

    # --- 阈值 2: PR 曲线拐点 (最小化 |P-R|) ---
    best_thresh_pr = pr_threshold(y_holdout, holdout_probas)
    print(f"\n使用 PR 曲线拐点 (最小化 |P-R|) 计算最佳阈值: {best_thresh_pr:.4f}")
    y_pred_holdout_pr = (holdout_probas >= best_thresh_pr).astype(int)
    evaluate_and_print_metrics(y_holdout, y_pred_holdout_pr, holdout_probas, "PR Curve Elbow") # 调用已修复的函数

    # ----- 7. 保存完整模型管道 (使用 Youden's J 阈值) -----
    # (可根据业务需求选择保存哪个阈值的结果)
    model_dict = {
        "imputer": final_model_clean.imputer,
        "scaler": final_model_clean.scaler,
        "pca": final_model_clean.pca, # 保存 PCA 变换器
        "model": final_model_clean.model,  # 包含了级联结构和所有层
        "threshold_youden": float(best_thresh_youden),
        "threshold_pr_elbow": float(best_thresh_pr),
        "fixed_params": fixed_params  # 保存固定参数
    }
    save_object(model_dict, "./model_pipeline_cascade_rf_pca.pkl")
    print("\n已保存包含PCA的完整模型管道 ./model_pipeline_cascade_rf_pca.pkl")
    print("=" * 70)
