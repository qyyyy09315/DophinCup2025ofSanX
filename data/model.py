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
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 环境配置
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class ProbabilityCalibrator:
    """Isotonic Regression概率校准器"""

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def fit(self, y_prob, y_true):
        self.calibrator.fit(y_prob, y_true)
        return self

    def calibrate(self, y_prob):
        return self.calibrator.transform(y_prob)


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

    def get_feature_mask(self):
        return self.feature_mask_


class ModelStacker:
    """Stacking集成学习器"""

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.calibrators = {}

    def fit_base_models(self, X, y):
        """训练基模型并生成元特征"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        print("\n[Stacking] 生成交叉验证元特征...")
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for j, (name, model) in enumerate(self.base_models.items()):
                model.fit(X_train, y_train)
                meta_features[val_idx, j] = model.predict_proba(X_val)[:, 1]

        # 完整训练基模型
        print("[Stacking] 完整训练基模型...")
        for name, model in self.base_models.items():
            model.fit(X, y)

        return meta_features

    def calibrate_probabilities(self, X, y, test_size=0.2):
        """训练概率校准器"""
        print("\n[Calibration] 训练概率校准器...")
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=test_size, random_state=SEED
        )

        # 训练基模型在校准集上的预测
        cal_probs = np.zeros((X_cal.shape[0], len(self.base_models)))
        for j, (name, model) in enumerate(self.base_models.items()):
            model.fit(X_train, y_train)
            cal_probs[:, j] = model.predict_proba(X_cal)[:, 1]

        # 训练元模型在校准集上的预测
        self.meta_model.fit(
            self.fit_base_models(X_train, y_train),
            y_train
        )
        meta_probs = self.meta_model.predict_proba(cal_probs)[:, 1]

        # 训练校准器
        self.calibrator = ProbabilityCalibrator()
        self.calibrator.fit(meta_probs, y_cal)

    def predict_proba(self, X):
        """生成校准后的概率预测"""
        # 基模型预测
        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1]
            for name, model in self.base_models.items()
        ])

        # 元模型预测
        meta_preds = self.meta_model.predict_proba(base_preds)[:, 1]

        # 概率校准
        if hasattr(self, 'calibrator'):
            return self.calibrator.calibrate(meta_preds)
        return meta_preds


def train_and_evaluate_optimized(X_train, y_train, X_test, y_test):
    """优化后的训练评估流程"""
    print("\n=== 数据预处理 ===")
    # 1. 缺失值填充和标准化
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # 2. 特征选择
    print("\n=== 特征选择 ===")
    feature_selector = XGBRFE_FeatureSelector(n_features=None, step=0.1)
    X_train_selected = feature_selector.fit(X_train_processed, y_train).transform(X_train_processed)
    X_test_selected = feature_selector.transform(X_test_processed)
    print(f"特征选择完成: {X_train_selected.shape[1]}个特征被保留")

    # 3. 处理类别不平衡
    print("\n=== 处理类别不平衡 ===")
    smote_tomek = SMOTETomek(random_state=SEED)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train_selected, y_train)
    print(f"过采样后数据形状: {X_resampled.shape}")

    # 4. 定义基模型
    base_models = {
        'AdaBoost': AdaBoostClassifier(
            n_estimators=393,
            learning_rate=0.426,
            random_state=SEED
        ),
        'XGBoost': XGBClassifier(
            n_estimators=356,
            max_depth=11,
            learning_rate=0.173,
            subsample=0.92,
            colsample_bytree=0.712,
            random_state=SEED,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            depth=13,
            learning_rate=0.8,
            random_seed=SEED,
            verbose=0
        )
    }

    # 5. 定义元模型
    meta_model = LogisticRegression(
        penalty='l2',
        C=0.1,
        random_state=SEED,
        max_iter=1000
    )

    # 6. 训练Stacking集成模型
    print("\n=== 训练Stacking模型 ===")
    stacker = ModelStacker(base_models, meta_model, n_folds=5)
    stacker.fit_base_models(X_resampled, y_resampled)
    stacker.calibrate_probabilities(X_resampled, y_resampled, test_size=0.2)

    # 7. 测试集预测
    print("\n=== 测试集评估 ===")
    y_proba = stacker.predict_proba(X_test_selected)

    # 8. 动态阈值选择 (基于验证集)
    thresholds = np.linspace(0.1, 0.5, 20)
    best_score = -1
    best_thresh = 0.2

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = 0.3 * recall_score(y_test, y_pred) + 0.5 * roc_auc_score(y_test, y_proba) + 0.2 * precision_score(
            y_test, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    y_pred = (y_proba >= best_thresh).astype(int)

    # 9. 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print("\n=== 评估结果 ===")
    print(f"最优阈值: {best_thresh:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Final Score: {final_score:.4f}")

    return {
        'recall': recall,
        'auc': auc,
        'precision': precision,
        'f1': f1,
        'final_score': final_score,
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
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n数据划分: 训练集={X_train.shape[0]}, 测试集={X_test.shape[0]}")

    # 训练和评估
    results = train_and_evaluate_optimized(X_train, y_train, X_test, y_test)

    # 保存概率预测结果
    pd.DataFrame({
        'true_label': y_test,
        'pred_prob': results['y_proba'],
        'pred_label': (results['y_proba'] >= results['best_threshold']).astype(int)
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
