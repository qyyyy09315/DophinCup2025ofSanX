import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# 1. 加载模型（直接强制使用 GPU，如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

local_model_path = "./bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path).to(device)
model.eval()

# 2. 读取数据 (修改为读取 Parquet 文件)
# 假设原始数据文件已转换为 parquet 格式，或者你可以先用 pd.read_csv(...).to_parquet() 转换一次
try:
    train_df = pd.read_parquet("../data/train.parquet")
    test_df = pd.read_parquet("../data/test.parquet")
    print("数据已从 Parquet 文件读取。")
except FileNotFoundError:
    print("未找到 Parquet 数据文件，尝试从 CSV 读取并转换...")
    # 如果 Parquet 文件不存在，则从 CSV 读取并保存为 Parquet 以备将来使用
    train_df_csv = pd.read_csv("../data/train.csv")
    test_df_csv = pd.read_csv("../data/test.csv")
    train_df_csv.to_parquet("../data/train.parquet", index=False)
    test_df_csv.to_parquet("../data/test.parquet", index=False)
    train_df = pd.read_parquet("../data/train.parquet")
    test_df = pd.read_parquet("../data/test.parquet")
    print("CSV 数据已转换为 Parquet 格式并读取。")


# 3. 需嵌入的中文列名
chinese_columns = [
    'province', 'city', 'industry_l1_name', 'industry_l2_name',
    'industry_l3_name', 'industry_l4_name', 'establish_year', 'legal_person',
    'business_scope', 'company_type', 'tags', 'company_on_scale', 'honor_titles',
    'sci_tech_ent_tags', 'top100_tags'
]


# 4. 改进的 BERT 嵌入函数（GPU 加速优化）
def get_bert_embedding(texts, batch_size=64):  # 增大 batch_size 以充分利用 GPU
    # 预处理文本（直接在 GPU 上处理，避免来回传输）
    texts = [str(text) if not pd.isna(text) else "" for text in texts]
    embeddings = []

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
        batch = texts[i:i + batch_size]

        # 关键优化：直接在 GPU 上处理，避免中间变量拷贝到 CPU
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)  # 直接将输入数据放到 GPU

            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

            # 显式释放 GPU 缓存（避免 OOM）
            del inputs, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


# 5. 优化后的数据处理函数（减少内存碎片）
def process_data(df):
    valid_cols = [col for col in chinese_columns if col in df.columns]
    non_chinese_cols = [col for col in df.columns if col not in chinese_columns]

    # 保留原始非中文列
    result_df = df[non_chinese_cols].copy()

    # 批量生成嵌入列
    embed_dfs = []
    for col in valid_cols:
        print(f"Processing column: {col}")
        texts = df[col].fillna("").tolist()
        embeddings = get_bert_embedding(texts, batch_size=64)  # 更大的 batch_size

        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embed_dfs.append(embed_df)

    # 一次性合并所有嵌入列
    if embed_dfs:
        all_embeddings = pd.concat(embed_dfs, axis=1)
        result_df = pd.concat([result_df, all_embeddings], axis=1)

    return result_df


# 6. 处理数据
train_processed = process_data(train_df)
test_processed = process_data(test_df)

# 7. 保存结果 (修改为保存为 Parquet 文件)
train_processed.to_parquet("train_bert_embedded.parquet", index=False)
test_processed.to_parquet("test_bert_embedded.parquet", index=False)

print("处理完成！嵌入后的数据已保存为 Parquet 格式。")
