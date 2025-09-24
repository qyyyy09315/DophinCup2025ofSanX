import pandas as pd

# 读取文件
file_path = "train_bert_embedded.csv"
df = pd.read_csv(file_path)

# 检查是否存在 'target' 列
if 'target' in df.columns:
    print("✅ 文件包含 'target' 列")
else:
    print("❌ 文件不包含 'target' 列")

