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


# 5. 改进的 BERT 嵌入函数（加入注意力池化）- 修复版本
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
            # 不替换embeddings，而是单独存储注意力池化结果
            # 这里修正了原代码的逻辑错误：pooled_emb是池化后的向量，不是原始hidden_states的加权和
            embeddings[-1] = pooled_emb.cpu().detach().numpy()  # 替换为注意力加权池化嵌入
            attention_weights.append(attn_weights.cpu().detach().numpy())

    emb = np.concatenate(embeddings, axis=0)
    if use_attention:
        weights = np.concatenate(attention_weights, axis=0)
        return emb, weights
    else:
        return emb, None


# 6. 行业层次化嵌入处理 - 修复版本
def process_industry_hierarchy(df):
    if 'industry_l1_name' not in df.columns:
        return None

    industry_cols = ['industry_l1_name', 'industry_l2_name', 'industry_l3_name', 'industry_l4_name']
    industry_cols = [col for col in industry_cols if col in df.columns]

    # 为每个行业层级生成嵌入
    industry_embeddings = {}
    for col in industry_cols:
        texts = df[col].fillna("").tolist()
        # 筛选出非空文本
        non_empty_texts = [t for t in texts if t.strip() != ""]
        if len(non_empty_texts) == 0:
            # 如果该列全是空值，生成零向量
            emb = np.zeros((len(texts), model.config.hidden_size))
            weights = np.zeros((len(texts), 1))  # 简化的权重，只用于维度匹配
        else:
            # 生成嵌入和权重
            emb_full = np.zeros((len(texts), model.config.hidden_size))
            weights_full = np.zeros((len(texts), 1))

            # 只对非空文本计算嵌入
            non_empty_indices = [i for i, t in enumerate(texts) if t.strip() != ""]
            non_empty_results, non_empty_weights = get_bert_embedding(non_empty_texts, batch_size=32,
                                                                      use_attention=True)

            # 将结果放回原位置
            emb_full[non_empty_indices] = non_empty_results
            if non_empty_weights is not None:
                # 注意力权重的形状可能不一致，取平均值或最大长度填充
                max_len = max(len(w) for w in non_empty_weights) if len(non_empty_weights) > 0 else 1
                padded_weights = []
                for w in non_empty_weights:
                    if len(w) < max_len:
                        padded_w = np.pad(w, (0, max_len - len(w)), mode='constant', constant_values=0)
                    else:
                        padded_w = w[:max_len]  # 截断到最大长度
                    padded_weights.append(np.mean(padded_w))  # 取平均作为该样本的权重

                weights_full[non_empty_indices, 0] = padded_weights

            emb, weights = emb_full, weights_full

        industry_embeddings[col] = (emb, weights)

    # 构建层次化嵌入
    hierarchical_embeddings = []
    for i in range(len(df)):
        # 初始化层次化嵌入（使用L1的嵌入作为基础，如果存在且非空）
        hier_emb = np.zeros(model.config.hidden_size)
        has_valid_industry = False

        # 从最顶层开始，逐层构建
        for level in ['industry_l1_name', 'industry_l2_name', 'industry_l3_name', 'industry_l4_name']:
            if level in industry_embeddings and pd.notna(df.iloc[i][level]) and df.iloc[i][level] != "":
                emb, weights = industry_embeddings[level]
                hier_emb = emb[i]  # 直接使用该层级的嵌入
                has_valid_industry = True
                break  # 使用找到的第一个有效层级

        if not has_valid_industry:
            # 如果没有任何行业信息，使用零向量
            hier_emb = np.zeros(model.config.hidden_size)

        # 如果需要融合多个层级（原逻辑），可以按以下方式实现
        # 但这里我们简化为只使用第一个非空的层级
        # 如果确实需要融合，需要处理不同长度的注意力权重问题

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
        # 对于非行业列，我们使用普通的[CLS]嵌入，不需要注意力权重
        embeddings, _ = get_bert_embedding(texts, batch_size=32, use_attention=False)

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



