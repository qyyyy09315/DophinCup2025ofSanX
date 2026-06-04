import pandas as pd
import re


def extract_chinese_columns(df):
    """提取包含中文内容的列（包括 company_id）"""
    # 1. 确保 company_id 列存在
    if "company_id" not in df.columns:
        raise ValueError("DataFrame 中没有 'company_id' 列")

    # 2. 筛选包含中文的列（非空且至少有一个中文字符）
    chinese_columns = []
    for col in df.columns:
        # 跳过 company_id（单独处理）
        if col == "company_id":
            continue
        # 检查列中是否有至少一个中文字符（或非空）
        has_chinese = df[col].astype(str).apply(
            lambda x: bool(re.search(r'[\u4e00-\u9fff]', x))
        ).any()
        if has_chinese:
            chinese_columns.append(col)

    # 3. 返回结果（包含 company_id + 所有含中文的列）
    selected_columns = ["company_id"] + chinese_columns
    return df[selected_columns]


# 读取训练集和测试集
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# 提取目标列
train_result = extract_chinese_columns(train_df)
test_result = extract_chinese_columns(test_df)

# 保存结果（可选）
train_result.to_csv("train_chinese.csv", index=False)
test_result.to_csv("test_chinese.csv", index=False)

# 打印结果示例
print("训练集提取的列:", train_result.columns.tolist())
print("测试集提取的列:", test_result.columns.tolist())