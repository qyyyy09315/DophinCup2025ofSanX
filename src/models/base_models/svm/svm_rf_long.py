import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# 读取CSV文件
data = pd.read_csv('../clean.csv')

# 假设目标列为'target'，特征列除去'target'列
X = data.drop(columns=['target'])
y = data['target']

# 数据预处理
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 定义评分函数
def weighted_score(y_true, y_pred_proba):
    y_pred = np.round(y_pred_proba)  # 将概率值转换为预测标签
    auc = roc_auc_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return 0.4 * auc + 0.4 * recall + 0.2 * precision

# 自定义评分函数
scorer = make_scorer(weighted_score, needs_proba=True)

# 定义融合模型
voting_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='soft'
)

# 定义贝叶斯优化参数空间
param_space = {
    'rf__n_estimators': Integer(50, 200),
    'rf__max_depth': Integer(5, 50),
    'svc__C': Real(0.01, 100, prior='log-uniform'),
    'svc__gamma': Real(0.0001, 1, prior='log-uniform')
}

# 定义贝叶斯优化
opt = BayesSearchCV(
    voting_model,
    param_space,
    n_iter=10,
    cv=2,
    scoring=scorer,
    n_jobs=-1
)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
opt.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found:")
print(opt.best_params_)
print("Best weighted score found:")
print(opt.best_score_)

# 在测试集上评估模型
best_model = opt.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# 计算测试集上的加权分数
test_weighted_score = weighted_score(y_test, y_pred_proba)
print("Test set weighted score:")
print(test_weighted_score)
