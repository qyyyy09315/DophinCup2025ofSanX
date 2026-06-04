import numpy as np
import pandas as pd
from deepforest import CascadeForestClassifier
from imblearn.combine import SMOTEENN
from joblib import dump, load
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

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

# 记录训练数据的特征列
feature_columns = X.columns

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

# 保存特征选择器和其它转换器
dump(selector, '../feature_selector_xgboost_cemmdan.joblib')
dump(imputer, '../imputer.joblib')
dump(scaler, '../scaler.joblib')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# 定义基础模型，并用包装器包装CascadeForest
cascade_forest = CascadeForestWrapper(CascadeForestClassifier(random_state=42))
adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=42)

# 定义堆叠模型
stacking_model = StackingClassifier(
    estimators=[
        ('cascade_forest', cascade_forest),
        ('adaboost', adaboost)
    ],
    final_estimator=GaussianNB(),
    n_jobs=-1  # 最大化CPU利用率
)

# 训练堆叠模型
stacking_model.fit(X_train, y_train)

# 保存模型到本地
dump(stacking_model, '../model_pipeline_xgboost_cemmdan.joblib')

# 打印评估结果
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

# 应用部分：读取test.csv
test_data = pd.read_csv('../test.csv').drop(columns=['company_id'])

# 对字符串列进行独热编码，保持与训练数据一致
test_data = pd.get_dummies(test_data, drop_first=True)

# 确保测试数据的特征列与训练数据一致
test_data = test_data.reindex(columns=feature_columns, fill_value=0)

# 读取已保存的转换器
imputer = load('../imputer.joblib')
scaler = load('../scaler.joblib')
selector = load('../feature_selector_xgboost_cemmdan.joblib')

# 对测试数据进行缺失值处理、特征缩放和特征选择
test_data = imputer.transform(test_data)
test_data = scaler.transform(test_data)
test_data = selector.transform(test_data)

# 读取保存的模型
stacking_model = load('../model_pipeline_xgboost_cemmdan.joblib')

# 进行预测
y_pred_test = stacking_model.predict(test_data)
y_proba_test = stacking_model.predict_proba(test_data)[:, 1]

# 假设测试数据中有 'uuid' 列用于识别
uuid_column = pd.read_csv('../test.csv')['company_id']

# 创建结果数据框并保存
results_df = pd.DataFrame({
    'uuid': uuid_column,
    'proba': y_proba_test,
    'prediction': y_pred_test,
})

# 保存结果到CSV
results_df.to_csv('submit_template.csv', index=False)

# 本地98.4434