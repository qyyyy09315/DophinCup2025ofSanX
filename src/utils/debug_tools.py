import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import joblib

# 数据加载与预处理
data = pd.read_csv('clean.csv')

# 删除 'company_id' 列
print(f"原始数据维度: {data.shape}")
data = data.drop(columns=['company_id'])
print(f"删除公司ID后的数据维度: {data.shape}")

# 分离特征和目标
X = data.drop(columns=['target'])
y = data['target']
print(f"特征数据 X 维度: {X.shape}")
print(f"目标数据 y 维度: {y.shape}")

# 数据填补
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
print(f"填补后 X 数据维度: {X_imputed.shape}")

# 去除常量特征
std_devs = np.std(X_imputed, axis=0)
non_constant_features = std_devs > 0
print(f"非零标准差特征数量: {np.sum(non_constant_features)}")
X_non_constant = X_imputed[:, non_constant_features]
print(f"非零标准差后的 X 维度: {X_non_constant.shape}")

# 使用正确的列名
columns_after_dropping = data.drop(columns=['target']).columns  # 删除目标列后的列名

# 确保布尔索引的维度一致
non_constant_feature_names = columns_after_dropping[non_constant_features]  # 得到非常量特征名
print(f"非零标准差特征名的数量: {len(non_constant_feature_names)}")

# 通过 VarianceThreshold 进一步特征选择
selector = VarianceThreshold()
X_selected = selector.fit_transform(X_non_constant)
print(f"VarianceThreshold 选择后的特征数量: {X_selected.shape[1]}")

# 获取选择后的特征名
selected_feature_names = non_constant_feature_names[selector.get_support()]
print(f"选择后的特征名数量: {len(selected_feature_names)}")

# 检查是否匹配
assert X_selected.shape[1] == len(selected_feature_names), "特征选择后的维度和特征名数量不匹配！"
print("特征选择过程成功完成")

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)

# 下采样
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# 保存预处理步骤
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(undersampler, 'undersampler.pkl')
joblib.dump(selector, 'variance_threshold_selector.pkl')
joblib.dump(non_constant_features, 'non_constant_features.pkl')
joblib.dump(selected_feature_names, 'selected_feature_names.pkl')

print("预处理器和特征选择器已保存。")
