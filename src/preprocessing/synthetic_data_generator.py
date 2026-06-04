import pandas as pd
import numpy as np
from scipy import stats

# 读取CSV文件
data = pd.read_csv('clean.csv')

# 使用Z-score方法识别离群值
numeric_data = data.select_dtypes(include=['number'])
z_scores = stats.zscore(numeric_data)
abs_z_scores = np.abs(z_scores)
outliers_z = (abs_z_scores > 2.5).any(axis=1)  # 修改为any(axis=1)
print("Outliers based on Z-score:")
print(data[outliers_z][['company_id']])  # 只显示公司ID

# 使用IQR方法识别离群值
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
print("Outliers based on IQR:")
print(data[outliers_iqr][['company_id']])  # 只显示公司ID

# # 检查日期数据范围（假设有一个日期列'date'）
# if 'date' in data.columns:
#     data['date'] = pd.to_datetime(data['date'], errors='coerce')
#     print("Date column summary:")
#     print(data['date'].describe())
#
# # 检查分类数据的唯一值（假设有一个分类列'category'）
# if 'category' in data.columns:
#     print("Unique values in 'category' column:")
#     print(data['category'].unique())
#
# # 打印前几行数据
# print("First few rows of the dataset:")
# print(data.head())
