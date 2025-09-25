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
    'industry_l3_name', 'industry_l4_name', 'establish_year', 'legal_person',
    'business_scope', 'company_type', 'tags', 'company_on_scale', 'honor_titles',
    'sci_tech_ent_tags', 'top100_tags'
]

# 行业层次结构定义
industry_hierarchy = {
    'industry_l1_name': ['industry_l2_name', 'industry_l3_name', 'industry_l4_name'],
    'industry_l2_name': ['industry_l3_name', 'industry_l4_name'],
    'industry_l3_name': ['industry_l4_name']
}


# 5. 改进的 BERT 嵌入函数（加入注意力池化）
def get_bert_embedding(texts, batch_size=32, use_attention=False):
    embeddings = []
    attention_weights = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)  # 确保输入在正确设备上

        with torch.no_grad():
            outputs = model(**inputs)

        # 获取 [CLS] 嵌入
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [batch_size, hidden_size]
        embeddings.append(batch_embeddings)

        if use_attention:
            # 使用自定义注意力池化层
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            mask = inputs['attention_mask']  # [batch_size, seq_len]
            pooled_emb, attn_weights = attention_pooler(hidden_states, mask)
            embeddings.append(pooled_emb.cpu().detach().numpy())  # 替换为注意力加权嵌入
            attention_weights.append(attn_weights.cpu().detach().numpy())

    emb = np.concatenate(embeddings, axis=0)
    if use_attention:
        weights = np.concatenate(attention_weights, axis=0)
        return emb, weights
    else:
        return emb, None


# 6. 行业层次化嵌入处理
def process_industry_hierarchy(df):
    if 'industry_l1_name' not in df.columns:
        return None

    industry_cols = ['industry_l1_name', 'industry_l2_name', 'industry_l3_name', 'industry_l4_name']
    industry_cols = [col for col in industry_cols if col in df.columns]

    # 为每个行业层级生成嵌入
    industry_embeddings = {}
    for col in industry_cols:
        texts = df[col].fillna("").tolist()
        emb, weights = get_bert_embedding(texts, batch_size=32, use_attention=True)
        industry_embeddings[col] = (emb, weights)

    # 构建层次化嵌入
    hierarchical_embeddings = []
    for i in range(len(df)):
        # 初始化层次化嵌入（使用L1的嵌入作为基础）
        if 'industry_l1_name' in industry_embeddings and not pd.isna(df.loc[i, 'industry_l1_name']):
            hier_emb = industry_embeddings['industry_l1_name'][0][i].copy()
        else:
            hier_emb = np.zeros(model.config.hidden_size)

        # 逐层加权融合
        for col, children in industry_hierarchy.items():
            if col in industry_embeddings and not pd.isna(df.loc[i, col]):
                current_weight = np.mean(industry_embeddings[col][1][i])  # 当前层级的平均注意力权重
                for child in children:
                    if child in industry_embeddings and not pd.isna(df.loc[i, child]):
                        child_weight = np.mean(industry_embeddings[child][1][i])
                        child_emb = industry_embeddings[child][0][i]
                        # 加权融合（当前层级和子层级）
                        hier_emb = hier_emb * current_weight + child_emb * child_weight
                        hier_emb /= np.linalg.norm(hier_emb)  # 归一化

        hierarchical_embeddings.append(hier_emb)

    hierarchical_embeddings = np.stack(hierarchical_embeddings)
    return hierarchical_embeddings


# 7. 优化后的数据处理函数
def process_data(df):
    valid_cols = [col for col in chinese_columns if col in df.columns]
    non_chinese_cols = [col for col in df.columns if col not in chinese_columns]

    # 保留原始非中文列
    result_df = df[non_chinese_cols].copy()

    # 处理行业层次结构
    print("Processing industry hierarchy...")
    industry_hier_emb = process_industry_hierarchy(df)
    if industry_hier_emb is not None:
        industry_hier_df = pd.DataFrame(
            industry_hier_emb,
            columns=[f"industry_hier_emb_{i}" for i in range(industry_hier_emb.shape[1])]
        )
        result_df = pd.concat([result_df, industry_hier_df], axis=1)

    # 处理其他文本列
    embed_dfs = []
    for col in valid_cols:
        if col.startswith('industry_l'):  # 跳过已经处理的行业列
            continue

        print(f"Processing column: {col}")
        texts = df[col].fillna("").tolist()
        embeddings, _ = get_bert_embedding(texts, batch_size=32, use_attention=False)  # 不重复计算注意力权重

        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embed_dfs.append(embed_df)

    # 合并所有嵌入
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
