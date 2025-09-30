import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


class CascadeRandomForest:
    """
    级联随机森林模型。
    通过逐层训练随机森林，并将预测概率作为新特征传递给下一层，
    实现层次化的特征学习。
    """

    def __init__(self, n_layers=3, n_estimators=100, max_depth=None, random_state=42):
        """
        初始化级联随机森林。

        :param n_layers: 级联的层数。
        :param n_estimators: 每层中随机森林的树的数量。
        :param max_depth: 每棵树的最大深度。
        :param random_state: 随机种子。
        """
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
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
        for i in range(self.n_layers):
            # 为了增加多样性，可以对每层使用不同的随机种子
            # 或者调整其他超参数
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,  # 改变随机种子
                n_jobs=-1
            )
            print(f"训练第 {i + 1} 层随机森林...")
            rf.fit(X_current, y)
            self.layers.append(rf)

            # 将原始特征与当前层的概率预测拼接，作为下一层的输入
            probas = rf.predict_proba(X_current)
            X_current = np.hstack([X, probas])  # 始终与原始特征拼接

        print("级联随机森林训练完成。")

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
    thresholds = np.linspace(0, 1, 101)
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


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练：SMOTEENN -> 级联随机森林 (Cascade Random Forest)")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"

    test_size = 0.10
    random_state = 42

    # 级联随机森林超参
    cascade_params = {
        'n_layers': 3,
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': random_state
    }

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path)
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

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # ----- 2. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_scaled, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集 (Holdout): {X_holdout.shape}")

    # ----- 3. SMOTEENN 重采样 -----
    print("正在进行 SMOTEENN 重采样 ...")
    smoteenn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train_full, y_train_full)
    print(f"重采样后: X={X_resampled.shape}, 正类={y_resampled.sum()}, 负类={(y_resampled == 0).sum()}")

    # ----- 4. 训练级联随机森林模型 -----
    print("训练级联随机森林模型 ...")
    cascade_rf_model = CascadeRandomForest(**cascade_params)
    cascade_rf_model.fit(X_resampled, y_resampled)

    save_object(cascade_rf_model, "./cascade_random_forest_model.pkl")
    print("已训练并保存级联随机森林模型 ./cascade_random_forest_model.pkl")

    # ----- 5. 模型评估与阈值选择 -----
    print("在 Holdout 集上评估模型...")
    # 获取最后一层的概率预测 (对于二分类，我们通常取正类概率)
    holdout_probas = cascade_rf_model.predict_proba(X_holdout)[:, 1]

    # 使用 Youden's J 统计量选择最佳阈值
    best_thresh = youden_threshold(y_holdout, holdout_probas)
    print(f"使用 Youden's J 统计量计算最佳阈值: {best_thresh:.4f}")

    # 根据最佳阈值进行预测
    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)

    # 计算评估指标
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # 自定义评分公式
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"--- Holdout 集评估结果 (阈值={best_thresh:.4f}) ---")
    print(f"F1-Score: {f1:.5f}")
    print(f"Recall:   {recall:.5f}")
    print(f"AUC:      {auc:.5f}")
    print(f"Precision:{precision:.5f}")
    print(f"Final Score (30*R + 50*AUC + 20*P): {final_score:.5f}")
    print("-" * 40)

    # ----- 6. 保存完整模型管道 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "model": cascade_rf_model,  # 包含了级联结构和所有层
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model_pipeline_cascade_rf.pkl")
    print("已保存完整模型管道 ./model_pipeline_cascade_rf.pkl")
    print("=" * 60)
