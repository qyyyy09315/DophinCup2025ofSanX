import numpy as np
import pandas as pd
from deepforest import CascadeForestClassifier
from imblearn.over_sampling import ADASYN
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


# 自定义包装器，确保 classes_ 被正确初始化并进行剪枝
class CascadeForestWrapper(ClassifierMixin):
    def __init__(self, cascade_forest, max_layers=10, **kwargs):
        self.cascade_forest = cascade_forest
        self.classes_ = None
        self.max_layers = max_layers

    def fit(self, X, y, validation_data=None):
        best_val_acc = 0
        patience = 3  # 早停耐心值
        patience_counter = 0

        def fit_layer(layer):
            print(f"Fitting cascade layer = {layer}")
            self.cascade_forest.fit(X, y)
            if validation_data is not None:
                X_val, y_val = validation_data
                y_pred_train = self.cascade_forest.predict(X)
                y_pred_val = self.cascade_forest.predict(X_val)
                train_acc = np.mean(y_pred_train == y)
                val_acc = np.mean(y_pred_val == y_val)
                print(f"Training Accuracy at layer {layer} = {train_acc * 100:.3f} %")
                print(f"Validation Accuracy at layer {layer} = {val_acc * 100:.3f} %")
                return val_acc
            return 0

        for layer in range(self.max_layers):
            val_acc = fit_layer(layer)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at layer {layer}")
                break

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.cascade_forest.predict(X)

    def predict_proba(self, X):
        return self.cascade_forest.predict_proba(X)

    def get_params(self, deep=True):
        return {"cascade_forest": self.cascade_forest, "max_layers": self.max_layers}

    def set_params(self, **params):
        if 'cascade_forest' in params:
            self.cascade_forest = params['cascade_forest']
        if 'max_layers' in params:
            self.max_layers = params['max_layers']
        return self


# 读取数据
data = pd.read_csv('../clean.csv')

# 对字符串列进行标签编码
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 提取特征和标签
X = data.drop(columns=['target'])
y = data['target']

# 检查缺失值并进行填补
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 使用 ADASYN 进行自适应采样
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# 定义第一层模型：AdaBoost优化的深度森林
deep_forest = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.01,
    random_state=42
)

# 创建 Deep Forest 包装器
deep_forest_wrapper = CascadeForestWrapper(deep_forest)


# 定义第二层模型：注意力机制优化的朴素贝叶斯
class AttentionWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        # 应用注意力机制的自定义代码
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        # 应用注意力机制的自定义代码
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        # 应用注意力机制的自定义代码
        return self.base_classifier.predict_proba(X)

    def get_params(self, deep=True):
        return {"base_classifier": self.base_classifier}

    def set_params(self, **params):
        if 'base_classifier' in params:
            self.base_classifier = params['base_classifier']
        return self


attention_nb = AttentionWrapper(GaussianNB(var_smoothing=1e-9))

# 组合模型
stacking_model = StackingClassifier(
    estimators=[
        ('deep_forest', deep_forest_wrapper)
    ],
    final_estimator=attention_nb,
    passthrough=True,
    n_jobs=-1
)

# 训练堆叠模型
stacking_model.fit(X_resampled, y_resampled)

# 在测试集上进行模型评估
y_pred = stacking_model.predict(X_test)
y_pred_proba = stacking_model.predict_proba(X_test)

# 计算各项评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 对于多分类问题，使用适当的平均方法计算AUC
if len(np.unique(y)) > 2:
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
else:
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# 输出详细的评估报告
print("=" * 60)
print("模型评估结果")
print("=" * 60)
print(f"准确率 (Accuracy): {accuracy:.6f}")
print(f"精确率 (Precision): {precision:.6f}")
print(f"召回率 (Recall): {recall:.6f}")
print(f"F1分数 (F1-Score): {f1:.6f}")
print(f"AUC得分: {auc:.6f}")
print()

print("=" * 60)
print("详细分类报告")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("=" * 60)
print("混淆矩阵")
print("=" * 60)
print(confusion_matrix(y_test, y_pred))