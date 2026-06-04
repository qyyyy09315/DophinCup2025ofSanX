import numpy as np
import pandas as pd
import xgboost as xgb
from deepforest import CascadeForestClassifier
from imblearn.combine import SMOTEENN
from joblib import dump, parallel_backend
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import os

# 设置并行处理的超时时间
os.environ['JOBLIB_TIMEOUT'] = '1800'  # 以秒为单位，设置为30分钟

# 自定义包装器，确保classes_被正确初始化
class CascadeForestWrapper(ClassifierMixin):
    def __init__(self, cascade_forest, **kwargs):
        self.cascade_forest = cascade_forest
        self.classes_ = None

    def fit(self, X, y):
        self.cascade_forest.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.cascade_forest.predict(X)

    def predict_proba(self, X):
        return self.cascade_forest.predict_proba(X)

    def get_params(self, deep=True):
        return {"cascade_forest": self.cascade_forest}

    def set_params(self, **params):
        if 'cascade_forest' in params:
            self.cascade_forest = params['cascade_forest']
        return self

# 读取训练数据并忽略 'company_id' 列
data = pd.read_csv('../clean.csv').drop(columns=['company_id'])

# 对字符串列进行独热编码
data = pd.get_dummies(data, drop_first=True)

# 提取特征和标签
X = data.drop(columns=['target'])
y = data['target']

# 检查缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 特征提取：先进行SMOTEENN欠采样处理，再应用CEMMDAN方法
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# CEMMDAN步骤：假设你有CEMMDAN的相关实现
# X_resampled = apply_cemmdan(X_resampled)

# 特征选择 (以XGBoost为例)
selector = SelectFromModel(xgb.XGBClassifier(n_estimators=100, random_state=42))
X_resampled = selector.fit_transform(X_resampled, y_resampled)

# 保存特征选择器
dump(selector, '../feature_selector_xgboost_cemmdan.joblib')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# 定义基础模型，使用网格搜索获得的最佳参数
cascade_forest = CascadeForestWrapper(CascadeForestClassifier(
    random_state=42,
    n_estimators=100,  # 最佳参数
    max_depth=5  # 最佳参数
))
adaboost = AdaBoostClassifier(
    algorithm='SAMME',
    random_state=42,
    n_estimators=200,  # 最佳参数
    learning_rate=1.0  # 最佳参数
)

# 定义堆叠模型，保持并行处理
stacking_model = StackingClassifier(
    estimators=[
        ('cascade_forest', cascade_forest),
        ('adaboost', adaboost)
    ],
    final_estimator=GaussianNB(),
    n_jobs=-1  # 使用所有可用核
)

# 在并行后端中设置超时时间
with parallel_backend('loky', inner_max_num_threads=2):
    # 训练堆叠模型
    stacking_model.fit(X_train, y_train)

# 保存模型到本地
dump(stacking_model, 'model_pipeline_xgboost_cemmdan_bestparams.joblib')

# 打印最终评估结果
y_pred_test = stacking_model.predict(X_test)
y_proba_test = stacking_model.predict_proba(X_test)[:, 1]

recall = recall_score(y_test, y_pred_test)
auc = roc_auc_score(y_test, y_proba_test)
precision = precision_score(y_test, y_pred_test)

final_score = 0.4 * recall + 0.4 * auc + 0.2 * precision
print(f"Recall: {recall:.6f}")
print(f"AUC: {auc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Final Score: {final_score:.6f}")
