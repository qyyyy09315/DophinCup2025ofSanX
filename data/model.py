import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 环境配置
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class XGBRFE_FeatureSelector:
    """基于XGBoost的RFE特征选择器"""

    def __init__(self, n_features=None, step=0.1):
        self.n_features = n_features
        self.step = step
        self.selector = None
        self.feature_mask_ = None

    def fit(self, X, y):
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1
        )

        if self.n_features is None:
            self.n_features = X.shape[1] // 2

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


class TwoStageEnsemble:
    """两阶段集成模型（XGB+AdaBoost -> CatBoost）"""

    def __init__(self):
        self.xgb_model = None
        self.ada_model = None
        self.cat_model = None
        self.feature_selector = None
        self.scaler = None

    def fit(self, X, y):
        # 第一阶段：特征选择
        print("\n[Stage 1] 特征选择...")
        self.feature_selector = XGBRFE_FeatureSelector()
        X_selected = self.feature_selector.fit(X, y).transform(X)

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        # 处理类别不平衡
        smote_tomek = SMOTETomek(random_state=SEED)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_scaled, y)

        # 第二阶段：训练初级模型
        print("\n[Stage 2] 训练初级模型...")
        # XGBoost模型（优化AUC）
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=9,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1]),
            random_state=SEED,
            n_jobs=-1,
            eval_metric='auc'
        )
        self.xgb_model.fit(X_resampled, y_resampled)

        # AdaBoost模型（优化召回率）
        self.ada_model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=SEED
        )
        self.ada_model.fit(X_resampled, y_resampled)

        # 生成初级模型预测概率作为新特征
        print("\n[Stage 3] 生成中级特征...")
        xgb_proba = self.xgb_model.predict_proba(X_resampled)[:, 1].reshape(-1, 1)
        ada_proba = self.ada_model.predict_proba(X_resampled)[:, 1].reshape(-1, 1)
        X_meta = np.hstack([X_resampled, xgb_proba, ada_proba])

        # 第三阶段：训练CatBoost最终模型
        print("\n[Stage 4] 训练CatBoost最终模型...")
        self.cat_model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=SEED,
            verbose=100
        )
        self.cat_model.fit(X_meta, y_resampled)

    def predict_proba(self, X):
        """预测概率（只返回正类概率）"""
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)

        # 生成初级模型预测
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
        ada_proba = self.ada_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
        X_meta = np.hstack([X_scaled, xgb_proba, ada_proba])

        # 最终预测
        return self.cat_model.predict_proba(X_meta)[:, 1]


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练评估流程"""
    # 数据预处理管道
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 训练两阶段集成模型
    print("\n=== 训练两阶段集成模型 ===")
    ensemble = TwoStageEnsemble()
    ensemble.fit(X_train_processed, y_train)

    # 测试集预测
    print("\n=== 测试集评估 ===")
    y_proba = ensemble.predict_proba(X_test_processed)

    # 动态选择最佳阈值（基于验证集）
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    # 在验证集上寻找最佳阈值
    val_proba = ensemble.predict_proba(X_val)
    thresholds = np.linspace(0.1, 0.5, 50)
    best_auc = -1
    best_thresh = 0.2

    for thresh in thresholds:
        y_pred = (val_proba >= thresh).astype(int)
        auc = roc_auc_score(y_val, val_proba)
        recall = recall_score(y_val, y_pred)
        # 优先优化AUC和Recall的组合指标
        score = 0.7 * auc + 0.3 * recall

        if score > best_auc:
            best_auc = score
            best_thresh = thresh

    # 应用最佳阈值
    y_pred = (y_proba >= best_thresh).astype(int)

    # 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== 评估结果 ===")
    print(f"最佳阈值: {best_thresh:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {
        'recall': recall,
        'auc': auc,
        'precision': precision,
        'f1': f1,
        'y_proba': y_proba,
        'best_threshold': best_thresh
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

        X = data.drop(columns=["target"]).values
        y = data["target"].values.astype(int)
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

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    print(f"\n数据划分: 训练集={X_train.shape[0]}, 测试集={X_test.shape[0]}")

    # 训练和评估
    results = train_and_evaluate(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
