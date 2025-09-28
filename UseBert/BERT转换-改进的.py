import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os

# === 1. 加载模型（强制使用 GPU，如果可用） ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 确保本地模型路径存在
local_model_path = "./bert-base-chinese"
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"本地模型路径 {local_model_path} 不存在。请确保模型已下载至此目录。")

tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path).to(device)
model.eval()


# === 2. 简化版 BERT 嵌入函数（仅使用 [CLS] 向量）===
def get_bert_embedding(texts, batch_size=32):
    """
    使用 BERT 模型为文本列表生成嵌入向量。
    仅使用 [CLS] token 的最后一层隐藏状态作为句向量。
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        # padding=True, truncation=True 是关键参数
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 使用 [CLS] token 的向量 ([:, 0, :]) 作为整个句子的表示
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    # 将所有批次的嵌入向量拼接成一个大的 numpy 数组
    final_embeddings = np.concatenate(embeddings, axis=0)
    return final_embeddings


# === 3. 数据处理函数 ===
def process_data(df, df_name=""):
    """
    处理单个 DataFrame，为指定列生成 BERT 嵌入。
    """
    print(f"--- Processing {df_name} ---")
    # 需要生成嵌入的中文列名
    chinese_columns = [
        'province', 'city', 'industry_l1_name', 'industry_l2_name',
        'industry_l3_name', 'industry_l4_name', 'legal_person',
        'business_scope', 'company_type', 'tags', 'company_on_scale', 'honor_titles',
        'sci_tech_ent_tags', 'top100_tags'
    ]

    # 筛选出实际存在于 DataFrame 中的列
    valid_cols = [col for col in chinese_columns if col in df.columns]
    # 选出不需要生成嵌入的其他列
    non_chinese_cols = [col for col in df.columns if col not in chinese_columns]

    # 创建一个只包含非中文列的 DataFrame 作为基础
    result_df = df[non_chinese_cols].copy()

    # 为每个有效的中文列生成嵌入
    embed_dfs = []
    for col in valid_cols:
        print(f"  Processing column: {col}")
        # 处理缺失值，用空字符串填充
        texts = df[col].fillna("").tolist()
        # 获取嵌入向量
        embeddings = get_bert_embedding(texts, batch_size=32)

        # 将嵌入向量转换为 DataFrame，列名加上前缀以便区分
        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embed_dfs.append(embed_df)

    # 如果生成了嵌入，则将它们与基础 DataFrame 拼接起来
    if embed_dfs:
        all_embeddings = pd.concat(embed_dfs, axis=1)
        result_df = pd.concat([result_df, all_embeddings], axis=1)

    return result_df


# === 4. 主执行流程 ===
if __name__ == "__main__":
    # --- 读取数据 ---
    print("Loading data...")
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # --- 处理训练集 ---
    train_processed = process_data(train_df, "Train Data")

    # --- 处理测试集 ---
    test_processed = process_data(test_df, "Test Data")

    # --- 保存为 Parquet 格式 ---
    print("Saving results to Parquet files...")
    train_output_path = "train_bert_embedded2.parquet"
    test_output_path = "test_bert_embedded2.parquet"

    train_processed.to_parquet(train_output_path, index=False)
    test_processed.to_parquet(test_output_path, index=False)

    print(f"处理完成！")
    print(f"训练集嵌入已保存至: {train_output_path} (Shape: {train_processed.shape})")
    print(f"测试集嵌入已保存至: {test_output_path} (Shape: {test_processed.shape})")




