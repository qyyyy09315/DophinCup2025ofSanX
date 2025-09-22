import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 环境配置
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class AdvancedPreprocessor:
    """高级数据预处理管道，包含清洗、标准化、特征工程"""

    def __init__(self):
        self.imputer_num = None
        self.scaler = None
        self.winsorizer = None
        self.pca = None
        self.feature_selector = None
        self.numerical_features = None # 需要外部传入或在fit时推断

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
        X_imputed[:, self.numerical_features] = self.imputer_num.fit_transform(X[:, self.numerical_features] if isinstance(X, np.ndarray) else X[self.numerical_features])

        print("步骤 2/4: 异常值处理 (Winsorization)...")
        # 2. 异常值处理 - Winsorization
        self.winsorizer = QuantileTransformer(output_distribution='uniform', random_state=SEED)
        X_winsorized = X_imputed.copy()
        X_winsorized[:, self.numerical_features] = self.winsorizer.fit_transform(X_imputed[:, self.numerical_features] if isinstance(X_imputed, np.ndarray) else X_imputed[self.numerical_features])

        print("步骤 3/4: 特征转换与标准化...")
        # 3. 特征转换与标准化
        # Z-score 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_winsorized)

        print("步骤 4/4: PCA 降维...")
        # 4. PCA 降维 (保留85%方差)
        self.pca = PCA(n_components=0.85)
        self.pca.fit(X_scaled)

        # 注意：此处未包含Target Encoding和Box-Cox变换，因为需要目标变量y或对特征分布有特定假设
        # 如果需要，可以在fit方法中添加，并在transform中应用

        print(f"预处理完成。PCA后特征数: {self.pca.n_components_}")
        return self

    def transform(self, X):
        """应用预处理步骤"""
        if isinstance(X, pd.DataFrame):
             X = X.values # 转换为 numpy array 以便处理

        # 1. 缺失值处理
        X_imputed = X.copy()
        X_imputed[:, self.numerical_features] = self.imputer_num.transform(X[:, self.numerical_features])

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
            self.n_features = max(1, X.shape[1] // 2) # 确保至少选择1个特征

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
        # self.feature_selector = None # 不再使用自定义的 RFE，因为AdvancedPreprocessor已经做了降维
        self.preprocessor = None # 使用高级预处理器

    def fit(self, X, y):
        # 第一阶段：高级数据预处理 (包含清洗, 标准化, PCA)
        print("\n[Stage 1] 高级数据预处理...")
        self.preprocessor = AdvancedPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y)

        # 处理类别不平衡 (注意: SMOTETomek 通常用于数值型数据，这里假设 X_processed 是数值的)
        print("\n[Stage 1.5] 处理类别不平衡 (SMOTETomek)...")
        smote_tomek = SMOTETomek(random_state=SEED)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_processed, y)

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


def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 {filename}")


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练评估流程"""
    # 训练集成模型
    print("\n=== 训练集成模型 ===")
    ensemble = FinalEnsemble()
    ensemble.fit(X_train, y_train)

    # 保存训练好的模型
    save_model(ensemble, 'best_cascaded_model.pkl')

    # 测试集预测
    print("\n=== 测试集评估 ===")
    y_proba = ensemble.predict_proba(X_test)

    # 固定阈值0.5进行分类 (原文注释是0.2，但代码是0.5，这里保持代码一致)
    y_pred = (y_proba >= 0.5).astype(int)

    # 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== 评估结果 ===")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # 注意：TestScore 的计算公式可能需要根据实际业务调整
    print(f"TestScore (示例公式 20*P + 50*AUC + 30*R): {20*precision+50*auc+30*recall:.4f}")

    return {
        'recall': recall,
        'auc': auc,
        'precision': precision,
        'f1': f1,
        'y_proba': y_proba,
        'y_pred': y_pred
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

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    print(f"\n数据划分: 训练集={X_train.shape[0]}, 测试集={X_test.shape[0]}")

    # 训练和评估
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 保存预测结果
    pd.DataFrame({
        'true_label': y_test,
        'pred_prob': results['y_proba'],
        'pred_label': results['y_pred']
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()



