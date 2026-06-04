import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
data = pd.read_csv('train.csv')

# 检查是否存在缺失值
print("Checking for missing values...")
print(data.isna().sum())

# 处理缺失值
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

for i in range(5):
    print(i)
    

# 对字符串特征进行编码
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# 保存清洗后的数据到clean.csv
data.to_csv('clean.csv', index=False)

print("Data cleaning complete. Cleaned data saved to 'clean.csv'.")