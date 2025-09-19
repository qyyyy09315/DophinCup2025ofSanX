import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap

# 自定义自适应注意力机制
class AdaptiveAttention(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# 读取CSV文件
data = pd.read_csv('../clean.csv')

# 提取特征和标签
X = data.drop(columns=['target'])  # 假设 'target' 列是标签
y = data['target']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义基础模型
rf = RandomForestClassifier(n_estimators=100)
ada = AdaBoostClassifier(n_estimators=100, algorithm='SAMME')

base_estimators = [
    ('rf', rf),
    ('ada', ada)
]

# 定义Stacking分类器
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=GaussianNB()  # 使用朴素贝叶斯作为最终分类器
)

# 定义流水线，包括特征缩放和堆叠模型
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('attention', AdaptiveAttention()),  # 添加自适应注意力机制
    ('stacking', stacking_model)
])

# 训练模型
pipeline.fit(X_train, y_train)

# 从管道中提取训练后的随机森林模型
rf_model = pipeline.named_steps['stacking'].named_estimators_['rf']

# 现在 rf_model 是已被拟合的模型，可以提取特征重要性
importances = rf_model.feature_importances_

# 输出特征重要性
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print("Feature Importances from RandomForestClassifier:")
print(feature_importance_df.to_string(index=False))  # 输出所有特征重要性条目

# 计算基于排列的特征重要性
result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

print("\nPermutation Feature Importance:")
for i in sorted_idx:
    print(f"{X.columns[i]}: {result.importances_mean[i]:.4f}")

# 使用SHAP解释整个模型（以随机森林为例）
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# 输出SHAP值
print("\nSHAP Values Summary:")
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
print(shap_values_df.describe().transpose())  # 输出SHAP值摘要
