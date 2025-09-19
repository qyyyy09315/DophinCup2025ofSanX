import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
data = pd.read_csv('test.csv')

# 检查是否存在缺失值
print("Checking for missing values...")
print(data.isna().sum())

# 处理缺失值：仅对数值列（排除 company_id）用均值填充
numeric_columns = data.select_dtypes(include=['number']).columns.difference(['company_id'])
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 对字符串特征进行编码：排除 company_id
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column == 'company_id':
        continue  # 跳过 company_id
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# 保存清洗后的数据
data.to_csv('testClean.csv', index=False)

print("Data cleaning complete. 'company_id' column preserved as original.")