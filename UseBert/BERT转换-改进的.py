import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# 1. 加载模型（强制使用 GPU，如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

local_model_path = "./bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path).to(device)
model.eval()


# 2. 定义注意力加权池化层
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, hidden_states, mask=None):
        weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(weights, dim=1)  # [batch_size, seq_len, 1]
        pooled = torch.sum(hidden_states * weights, dim=1)  # [batch_size, hidden_size]
        return pooled, weights.squeeze(-1)


# 初始化注意力池化层
attention_pooler = AttentionPooling(model.config.hidden_size).to(device)

# 3. 读取数据
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# 4. 需嵌入的中文列名
chinese_columns = [
    'province', 'city', 'industry_l1_name', 'industry_l2_name',
    'industry_l3_name', 'industry_l4_name', 'legal_person',
    'business_scope', 'company_type', 'tags', 'company_on_scale', 'honor_titles',
    'sci_tech_ent_tags', 'top100_tags'
]

# 行业层次结构定义
industry_hierarchy = {
    'industry_l1_name': ['industry_l2_name', 'industry_l3_name', 'industry_l4_name'],
    'industry_l2_name': ['industry_l3_name', 'industry_l4_name'],
    'industry_l3_name': ['industry_l4_name']
}


# 5. 改进的 BERT 嵌入函数（注意力权重取均值）
def get_bert_embedding(texts, batch_size=32, use_attention=False):
    embeddings = []
    attention_weights = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 默认使用 [CLS] 向量
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

        if use_attention:
            hidden_states = outputs.last_hidden_state
            mask = inputs['attention_mask']
            pooled_emb, attn_weights = attention_pooler(hidden_states, mask)

            embeddings[-1] = pooled_emb.cpu().detach().numpy()

            # === 核心修改：取平均权重，保证维度固定 ===
            attn_weights_mean = attn_weights.mean(dim=1).cpu().detach().numpy()  # [batch_size]
            attention_weights.append(attn_weights_mean[:, None])  # [batch_size, 1]

    emb = np.concatenate(embeddings, axis=0)
    if use_attention:
        weights = np.concatenate(attention_weights, axis=0)  # [num_samples, 1]
        return emb, weights
    else:
        return emb, None


# 6. 行业层次化嵌入处理
def process_industry_hierarchy(df):
    if 'industry_l1_name' not in df.columns:
        return None

    industry_cols = ['industry_l1_name', 'industry_l2_name', 'industry_l3_name', 'industry_l4_name']
    industry_cols = [col for col in industry_cols if col in df.columns]

    industry_embeddings = {}
    for col in industry_cols:
        texts = df[col].fillna("").tolist()
        non_empty_texts = [t for t in texts if t.strip() != ""]
        if len(non_empty_texts) == 0:
            emb = np.zeros((len(texts), model.config.hidden_size))
            weights = np.zeros((len(texts), 1))
        else:
            emb_full = np.zeros((len(texts), model.config.hidden_size))
            weights_full = np.zeros((len(texts), 1))

            non_empty_indices = [i for i, t in enumerate(texts) if t.strip() != ""]
            non_empty_results, non_empty_weights = get_bert_embedding(
                non_empty_texts, batch_size=32, use_attention=True
            )

            emb_full[non_empty_indices] = non_empty_results
            if non_empty_weights is not None:
                weights_full[non_empty_indices, 0] = non_empty_weights[:, 0]

            emb, weights = emb_full, weights_full

        industry_embeddings[col] = (emb, weights)

    hierarchical_embeddings = []
    for i in range(len(df)):
        hier_emb = np.zeros(model.config.hidden_size)
        has_valid_industry = False

        for level in ['industry_l1_name', 'industry_l2_name', 'industry_l3_name', 'industry_l4_name']:
            if level in industry_embeddings and pd.notna(df.iloc[i][level]) and df.iloc[i][level] != "":
                emb, weights = industry_embeddings[level]
                hier_emb = emb[i]
                has_valid_industry = True
                break

        if not has_valid_industry:
            hier_emb = np.zeros(model.config.hidden_size)

        hierarchical_embeddings.append(hier_emb)

    hierarchical_embeddings = np.stack(hierarchical_embeddings)
    return hierarchical_embeddings


# 7. 优化后的数据处理函数
def process_data(df):
    valid_cols = [col for col in chinese_columns if col in df.columns]
    non_chinese_cols = [col for col in df.columns if col not in chinese_columns]

    result_df = df[non_chinese_cols].copy()

    print("Processing industry hierarchy...")
    industry_hier_emb = process_industry_hierarchy(df)
    if industry_hier_emb is not None:
        industry_hier_df = pd.DataFrame(
            industry_hier_emb,
            columns=[f"industry_hier_emb_{i}" for i in range(industry_hier_emb.shape[1])]
        )
        result_df = pd.concat([result_df, industry_hier_df], axis=1)

    embed_dfs = []
    for col in valid_cols:
        if col.startswith('industry_l'):
            continue

        print(f"Processing column: {col}")
        texts = df[col].fillna("").tolist()
        embeddings, _ = get_bert_embedding(texts, batch_size=32, use_attention=False)

        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embed_dfs.append(embed_df)

    if embed_dfs:
        all_embeddings = pd.concat(embed_dfs, axis=1)
        result_df = pd.concat([result_df, all_embeddings], axis=1)

    return result_df


# 8. 处理数据
train_processed = process_data(train_df)
test_processed = process_data(test_df)

# 9. 保存结果
train_processed.to_csv("train_bert_embedded_hierarchical.csv", index=False)
test_processed.to_csv("test_bert_embedded_hierarchical.csv", index=False)

print("处理完成！")
