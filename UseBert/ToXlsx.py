import pandas as pd
import os

# 1. 读取CSV文件
csv_path = '../data/train.csv'
try:
    df = pd.read_csv(csv_path)
    print("CSV文件读取成功！")
except Exception as e:
    print("读取CSV失败:", e)
    exit()

# 2. 定义输出路径（自动获取桌面路径）
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "train_data.xlsx")

# 3. 保存为Excel
try:
    df.to_excel(output_path, index=False, engine='openpyxl')  # 需要安装openpyxl库
    print(f"文件已保存至: {output_path}")
except Exception as e:
    print("保存Excel失败:", e)