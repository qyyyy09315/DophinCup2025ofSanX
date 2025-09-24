import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# 1. 加载模型
local_model_path = "./bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 读取数据
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# 3. 明确指定中文列名（需嵌入的列）
chinese_columns = [
    'province', 'city', 'industry_l1_name', 'industry_l2_name',
    'industry_l3_name', 'establish_year', 'legal_person',
    'business_scope', 'company_type', 'tags'
]


# 4. 改进的BERT嵌入函数（保持不变）
def get_bert_embedding(texts, batch_size=32):
    texts = [str(text) if not pd.isna(text) else "" for text in texts]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)


# 5. 优化后的数据处理函数（保留原始非中文列）
def process_data(df):
    # 分离需嵌入的中文列和需保留的原始列
    valid_cols = [col for col in chinese_columns if col in df.columns]
    non_chinese_cols = [col for col in df.columns if col not in chinese_columns]

    # 保留原始非中文列（包括 company_id 和其他特征）
    result_df = df[non_chinese_cols].copy()

    # 为每列中文文本生成嵌入
    for col in valid_cols:
        print(f"Processing column: {col}")
        texts = df[col].fillna("").tolist()
        embeddings = get_bert_embedding(texts)

        # 将嵌入列命名为 {原列名}_emb_{i}
        for i in range(embeddings.shape[1]):
            result_df[f"{col}_emb_{i}"] = embeddings[:, i]

    return result_df


# 6. 处理数据
train_processed = process_data(train_df)
test_processed = process_data(test_df)

# 7. 保存结果（保留原始列 + 新增嵌入列）
train_processed.to_csv("train_bert_embedded.csv", index=False)
test_processed.to_csv("test_bert_embedded.csv", index=False)

print("处理完成！")