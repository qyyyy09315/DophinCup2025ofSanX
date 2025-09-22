import pandas as pd
import joblib
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
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle  # 新增导入pickle模块
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
            n_estimators=500,
            max_depth=15,
            learning_rate=0.001,
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


class FinalEnsemble:
    """最终集成模型（XGBoost + AdaBoost）"""

    def __init__(self):
        self.xgb_model = None
        self.ada_model = None
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

        # 第二阶段：训练XGBoost模型（优化AUC）
        print("\n[Stage 2] 训练XGBoost模型...")
        self.xgb_model = XGBClassifier(
            n_estimators=356,
            max_depth=11,
            learning_rate=0.172808,
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
            n_estimators=393,
            learning_rate=0.4262,
            random_state=SEED
        )
        self.ada_model.fit(X_resampled, y_resampled)

    def predict_proba(self, X):
        """预测概率（返回两个模型的平均概率）"""
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)

        # 获取两个模型的预测概率
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        ada_proba = self.ada_model.predict_proba(X_scaled)[:, 1]

        # 返回平均概率
        return (xgb_proba + ada_proba) / 2


def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 {filename}")


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练评估流程"""
    # 数据预处理管道
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 训练集成模型
    print("\n=== 训练集成模型 ===")
    ensemble = FinalEnsemble()
    ensemble.fit(X_train_processed, y_train)

    # 保存训练好的模型
    save_model(ensemble, 'best_cascaded_model.pkl')

    # 测试集预测
    print("\n=== 测试集评估 ===")
    y_proba = ensemble.predict_proba(X_test_processed)

    # 固定阈值0.2进行分类
    y_pred = (y_proba >= 0.2).astype(int)

    # 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== 评估结果 ===")
    print(f"固定阈值: 0.2")
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
        'y_pred': y_pred
    }





# 以上填写对象
# 1. 加载测试数据
test_data = pd.read_csv(r'testClean.csv')

# 2. 加载完整的级联模型
cascaded_model = joblib.load('best_cascaded_model.pkl')

# 3. 应用预处理和模型预测
X_test = test_data.drop(columns=['company_id'], errors='ignore')

# 获取预测概率（模型返回的是两个模型的平均概率）
y_proba = cascaded_model.predict_proba(X_test)

# 应用固定阈值0.2进行分类
threshold = 0.2  # 这里明确定义分类阈值
y_pred = (y_proba >= threshold).astype(int)

# 4. 创建结果数据框
results_df = pd.DataFrame({
    'uuid': test_data['company_id'],
    'proba': y_proba,
    'prediction': y_pred
})

# 5. 保存结果
results_df.to_csv(r'C:\Users\YKSHb\Desktop\submit_template.csv', index=False)
print(f"预测完成，使用阈值 {threshold} 进行分类，结果已保存到 桌面/submit_template.csv")
