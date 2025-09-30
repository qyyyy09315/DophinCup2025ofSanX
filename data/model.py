import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, make_scorer, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
class ModelPipeline(BaseEstimator, ClassifierMixin):
    """整合预处理和 CascadeRandomForest 的管道"""

    def __init__(self, imputer=None, scaler=None, model=None):
        self.imputer = imputer or SimpleImputer(strategy="mean")
        self.scaler = scaler or StandardScaler()
        self.model = model or CascadeRandomForest()

    def fit(self, X, y):
        """拟合整个管道：插补 -> 缩放 -> 模型训练"""
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
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


# --- 修复：修改自定义评分函数以接受任意额外关键字参数 ---
def custom_score_function(y_true, y_proba, **kwargs):
    """
    计算自定义评分：30 * Recall + 50 * AUC + 20 * Precision
    注意：GridSearchCV 传入的是概率，需要处理阈值或直接使用概率计算 AUC。
    这里我们直接使用 y_proba 计算 AUC，并使用默认 0.5 阈值计算其他指标。
    修复：添加 **kwargs 以捕获可能由 make_scorer 传递的意外参数 (如 needs_proba)。
    """
    # GridSearchCV 传给 make_scorer 的是 predict_proba 的结果，对于二分类是 (n_samples, 2)
    # 我们通常取正类的概率
    if y_proba.ndim > 1:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba  # 如果已经是正类概率

    y_pred = (y_proba_pos >= 0.5).astype(int)

    recall = recall_score(y_true, y_pred, zero_division=0)
    # AUC 需要概率
    try:
        auc = roc_auc_score(y_true, y_proba_pos)
    except ValueError:
        # 如果 y_true 只有一个类，AUC 无法计算，返回 0 或一个默认值
        auc = 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)

    # 避免 auc 为 nan 的情况影响整体评分
    if np.isnan(auc):
        auc = 0.0

    final_score = 30 * recall + 50 * auc + 20 * precision
    return final_score


# 包装成 sklearn 可用的 scorer
custom_scorer = make_scorer(custom_score_function, needs_proba=True)

if __name__ == "__main__":
    print("=" * 70)
    print("开始训练：SMOTEENN -> 级联随机森林 (Cascade RF) 带网格搜索调优 & 优化")
    print("优化方向: 1. 代价敏感学习(class_weight); 2. PR曲线拐点阈值")
    print("=" * 70)

    # ----- 配置 -----
    # 注意：请确保 './clean.csv' 文件存在于当前工作目录或提供完整路径
    data_path = "./clean.csv"
    test_size = 0.10
    random_state = 42

    # 网格搜索配置
    param_grid = {
        # 注意：参数名需要与 ModelPipeline.model 的属性对应
        'model__n_layers': [3, 5], # 示例：减少搜索空间以加快速度
        'model__n_estimators': [100, 200], # 示例：减少搜索空间以加快速度
        'model__max_depth': [4, None]  # None 表示不限制深度 # 示例：减少搜索空间以加快速度
        # 'model__class_weight_option': ['balanced'] # 可以也将其加入网格搜索
    }
    cv_folds = 3  # 交叉验证折数
    n_jobs = -1  # 并行运行作业数

    # ----- 1. 读取并预处理数据 -----
    # 注意：此部分需要一个名为 'clean.csv' 的有效文件
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{data_path}'。请确保文件路径正确。")
        # 如果没有文件，可以创建一个示例数据集用于演示
        print("创建示例数据集用于演示...")
        n_samples = 1000
        n_features = 20
        np.random.seed(42)
        X_demo = np.random.randn(n_samples, n_features)
        # 创建一个稍微不平衡的目标变量
        y_demo = np.random.binomial(1, 0.1, n_samples) # 10% 正类
        # 添加一些与目标相关的特征
        X_demo[y_demo==1, 0] += 1
        X_demo[y_demo==1, 1] += 1
        data = pd.DataFrame(X_demo, columns=[f"feature_{i}" for i in range(n_features)])
        data['target'] = y_demo
        # 添加一个分类特征并进行独热编码
        data['category'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
        # 确保至少有一个 'company_id' 列用于演示删除
        data['company_id'] = range(n_samples)
        # 重新生成 target 列以确保它在最后
        target_col = data.pop('target')
        data['target'] = target_col

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
    # 再将剩余部分划分为用于网格搜索的训练集和验证集 (这里简化处理，实际可以再细分)
    # 或者直接使用 GridSearchCV 的 cv 功能，它会在 X_temp 上进行 cv 划分
    print(f"用于网格搜索的数据集: {X_temp.shape}, Holdout 集: {X_holdout.shape}")

    # ----- 3. 网格搜索超参数调优 -----
    print("开始网格搜索超参数调优 ...")
    print(f"搜索空间: {param_grid}")

    # 创建管道实例 (内部模型使用默认参数，会被 GridSearchCV 覆盖)
    pipeline = ModelPipeline()

    # 创建 GridSearchCV 对象
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=custom_scorer,  # 使用修复后的自定义评分函数
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=2  # 显示进度
    )

    # 执行网格搜索 (注意：GridSearchCV 内部会处理 fit/predict_proba)
    grid_search.fit(X_temp, y_temp)

    print("网格搜索完成。")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.5f}")

    # ----- 4. 使用最佳参数重新训练完整模型 -----

    # --- 采用严谨方法重新训练 ---
    print("--- 严谨地重新训练最佳模型 ---")
    # 1. 提取最佳参数
    best_params = grid_search.best_params_
    print(f"使用的最佳参数: {best_params}")
    # 重建模型 (强制加入 class_weight='balanced')
    final_cascade_rf_clean = CascadeRandomForest(
        n_layers=best_params.get('model__n_layers', 3),  # 默认值以防万一
        n_estimators=best_params.get('model__n_estimators', 100),
        max_depth=best_params.get('model__max_depth', None),
        random_state=random_state,
        class_weight_option='balanced'  # 强制应用代价敏感学习
    )
    # 2. 用 X_temp 重新训练预处理步骤
    final_imputer = SimpleImputer(strategy="mean")
    final_scaler = StandardScaler()
    X_temp_imputed_clean = final_imputer.fit_transform(X_temp)
    X_temp_scaled_clean = final_scaler.fit_transform(X_temp_imputed_clean)

    # 3. 对预处理后的数据进行 SMOTEENN
    print("正在进行 SMOTEENN 重采样 (严谨流程)...")
    final_smoteenn = SMOTEENN(random_state=random_state)
    X_final_resampled, y_final_resampled = final_smoteenn.fit_resample(X_temp_scaled_clean, y_temp)
    print(
        f"重采样后: X={X_final_resampled.shape}, 正类={y_final_resampled.sum()}, 负类={(y_final_resampled == 0).sum()}")

    # 4. 用重采样后的数据训练 CascadeRF
    print("训练最终级联随机森林模型 (带有 class_weight='balanced') ...")
    final_cascade_rf_clean.fit(X_final_resampled, y_final_resampled)

    # 5. 组装最终管道
    final_model_clean = ModelPipeline(
        imputer=final_imputer,
        scaler=final_scaler,
        model=final_cascade_rf_clean
    )

    # ----- 5. 在 Holdout 集上评估最终模型 (多种阈值策略) -----
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

    # ----- 6. 保存完整模型管道 (使用 Youden's J 阈值) -----
    # (可根据业务需求选择保存哪个阈值的结果)
    model_dict = {
        "imputer": final_model_clean.imputer,
        "scaler": final_model_clean.scaler,
        "model": final_model_clean.model,  # 包含了级联结构和所有层
        "threshold_youden": float(best_thresh_youden),
        "threshold_pr_elbow": float(best_thresh_pr),
        "best_params": best_params  # 保存最佳参数
    }
    save_object(model_dict, "./model_pipeline_cascade_rf_optimized.pkl")
    print("\n已保存优化后的完整模型管道 ./model_pipeline_cascade_rf_optimized.pkl")
    print("=" * 70)




