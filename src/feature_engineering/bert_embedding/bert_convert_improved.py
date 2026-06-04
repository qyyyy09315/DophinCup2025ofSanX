import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # For softmax
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

HIDDEN_SIZE = model.config.hidden_size
# 假设增强嵌入是 [CLS], Weighted Avg, Max Pool 的拼接，所以是 3 倍
ENHANCED_EMBEDDING_SIZE = HIDDEN_SIZE * 3

# === 2. 改进的 BERT 嵌入函数（结合多种池化策略）===
def get_enhanced_bert_embedding(texts, batch_size=32):
    """
    使用 BERT 模型为文本列表生成增强的嵌入向量。
    结合 [CLS] 向量、加权平均池化和最大池化。
    """
    embeddings_cls = []
    embeddings_weighted_avg = []
    embeddings_max_pool = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Enhanced BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            attention_mask = inputs['attention_mask']        # [batch_size, seq_len]

        # --- 1. [CLS] Token 向量 ---
        batch_embeddings_cls = last_hidden_states[:, 0, :].cpu().numpy() # [batch_size, hidden_size]
        embeddings_cls.append(batch_embeddings_cls)

        # --- 2. 加权平均池化 (Attention-based) ---
        # 使用 [CLS] token 的向量作为查询 (query) 来计算注意力分数
        # query: [batch_size, 1, hidden_size]
        query = last_hidden_states[:, :1, :]
        # key: [batch_size, seq_len, hidden_size]
        key = last_hidden_states

        # 计算注意力分数 (点积) -> [batch_size, 1, seq_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # 应用 softmax 获取权重 -> [batch_size, 1, seq_len]
        # 需要将 attention_mask 扩展维度以适应 softmax
        extended_attention_mask = attention_mask.unsqueeze(1).float() # [batch_size, 1, seq_len]
        # 将 mask 为 0 的位置设置为一个极大的负数，使 softmax 后接近 0
        attention_scores = attention_scores.masked_fill(extended_attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1) # [batch_size, 1, seq_len]

        # 计算加权平均 -> [batch_size, 1, hidden_size] -> squeeze(1) -> [batch_size, hidden_size]
        batch_weighted_avg = torch.matmul(attention_weights, last_hidden_states).squeeze(1)
        embeddings_weighted_avg.append(batch_weighted_avg.cpu().numpy())

        # --- 3. 最大池化 (修复广播问题) ---
        # 将 padding 位置的值设为极小值，这样在 max 操作中会被忽略
        # extended_attention_mask: [batch_size, seq_len, 1]
        # 为了正确广播到 [batch_size, seq_len, hidden_size]，我们需要扩展到 [batch_size, seq_len, 1, 1]
        # 然后 PyTorch 会自动广播 hidden_size 维度
        extended_attention_mask_for_max = extended_attention_mask.transpose(-1, -2).unsqueeze(-1) # [batch_size, seq_len, 1, 1]
        # 将 last_hidden_states 中 padding 位置的值设为 -inf
        # last_hidden_states: [batch_size, seq_len, hidden_size]
        # 我们需要将它扩展为 [batch_size, seq_len, hidden_size, 1] 来匹配掩码的广播维度
        # 或者更简单地，我们直接在 [batch_size, seq_len, hidden_size] 上应用 [batch_size, seq_len, 1] 的掩码
        # PyTorch 会自动将 [batch_size, seq_len, 1] 广播到 [batch_size, seq_len, hidden_size]
        # 因此，我们直接使用 unsqueeze(-1) 后的 [batch_size, seq_len, 1] 作为掩码
        # 但是 unsqueeze(-1) 是 [batch_size, seq_len, 1]，我们需要的是 [batch_size, seq_len, 1] 广播到 [batch_size, seq_len, hidden_size]
        # 实际上，unsqueeze(-1) 后是 [batch_size, seq_len, 1]，这正是我们需要的广播形状。
        # 问题在于之前的 unsqueeze(-1) 是在 [batch_size, 1, seq_len] 上做的，得到了 [batch_size, 1, seq_len, 1]
        # 正确的做法是先 transpose(-1, -2) 得到 [batch_size, seq_len, 1]，然后再 unsqueeze(-1) 得到 [batch_size, seq_len, 1, 1]
        # 不过，其实我们只需要 [batch_size, seq_len, 1] 就可以广播到 [batch_size, seq_len, hidden_size]
        # 所以，正确的掩码应该是:
        mask_for_max_pooling = extended_attention_mask.transpose(-1, -2) # [batch_size, seq_len, 1]
        last_hidden_states_masked = last_hidden_states.clone()
        # 这里直接使用 [batch_size, seq_len, 1] 的 mask，PyTorch 会自动广播 hidden_size 维度
        last_hidden_states_masked.masked_fill_(mask_for_max_pooling == 0, float('-inf'))
        # 在序列长度维度 (dim=1) 上取最大值 -> [batch_size, hidden_size]
        batch_max_pool, _ = torch.max(last_hidden_states_masked, dim=1)
        embeddings_max_pool.append(batch_max_pool.cpu().numpy())

    # --- 4. 拼接所有策略的向量 ---
    final_embeddings_cls = np.concatenate(embeddings_cls, axis=0)
    final_embeddings_weighted_avg = np.concatenate(embeddings_weighted_avg, axis=0)
    final_embeddings_max_pool = np.concatenate(embeddings_max_pool, axis=0)

    # 将三种向量拼接在一起，形成最终的增强向量
    enhanced_embeddings = np.concatenate(
        [final_embeddings_cls, final_embeddings_weighted_avg, final_embeddings_max_pool],
        axis=1
    )
    return enhanced_embeddings


# === 3. 行业层次结构处理函数 (针对指定四列) ===
def process_industry_hierarchy(df):
    """
    根据指定的行业层次结构四列，为每个样本生成一个增强的行业嵌入向量。
    规则：优先使用最具体的行业级别（L4 -> L3 -> L2 -> L1）。
    """
    print("Processing industry hierarchy for specified columns...")
    # 定义行业列的优先级和名称 (严格按照 L4 -> L3 -> L2 -> L1)
    industry_hierarchy_cols = ['industry_l4_name', 'industry_l3_name', 'industry_l2_name', 'industry_l1_name']

    # 检查这些列是否在 DataFrame 中
    available_industry_cols = [col for col in industry_hierarchy_cols if col in df.columns]
    if not available_industry_cols:
        print("  None of the specified industry columns found in DataFrame.")
        return None

    # 存储每个级别的嵌入结果
    level_embeddings = {}

    # 为每个可用的行业级别列生成嵌入
    for col in available_industry_cols:
        print(f"  Generating embeddings for {col}...")
        texts = df[col].fillna("").tolist()
        # 为该级别的所有文本生成嵌入
        level_emb = get_enhanced_bert_embedding(texts, batch_size=32)
        level_embeddings[col] = level_emb

    # 初始化最终的行业嵌入矩阵 (样本数 x 增强嵌入维度)
    final_industry_embeddings = np.zeros((len(df), ENHANCED_EMBEDDING_SIZE))

    # 标记哪些样本已经找到了有效的行业嵌入
    assigned = np.zeros(len(df), dtype=bool)

    # 按优先级顺序（从最具体到最宽泛）填充嵌入
    # industry_hierarchy_cols 已经是按 L4->L3->L2->L1 排序
    for col in industry_hierarchy_cols: # 遍历指定的四列
        if col not in available_industry_cols:
             print(f"    Column {col} not found, skipping.")
             continue

        print(f"  Assigning embeddings from {col}...")
        col_data = df[col]
        # 找到该列中非空且尚未被更具体级别填充的样本索引
        # 注意：使用 ~assigned 来确保优先级高的列先赋值
        valid_indices = (~assigned) & (col_data.notna()) & (col_data.str.strip() != "")

        # 将这些样本的嵌入向量赋值给最终结果矩阵
        final_industry_embeddings[valid_indices] = level_embeddings[col][valid_indices]
        # 标记这些样本已被处理
        assigned[valid_indices] = True

    # 对于没有任何有效行业信息的样本，其嵌入向量将保持为零向量

    print("  Industry hierarchy processing for specified columns complete.")
    return final_industry_embeddings


# === 4. 主数据处理函数 ===
def process_data(df, df_name=""):
    """
    处理单个 DataFrame，为指定列生成 BERT 嵌入，并处理行业层次结构。
    """
    print(f"--- Processing {df_name} ---")
    # 需要生成嵌入的中文列名（不包括行业列，行业列单独处理）
    chinese_columns = [
         'province', 'city', 'legal_person',
        'business_scope', 'company_type', 'tags', 'company_on_scale', 'honor_titles',
        'sci_tech_ent_tags', 'top100_tags'
    ]
    # 行业列名 (用于从chinese_columns中排除)
    industry_columns = [
        'industry_l1_name', 'industry_l2_name',
        'industry_l3_name', 'industry_l4_name'
    ]

    # 筛选出实际存在于 DataFrame 中的非行业中文列
    valid_chinese_cols = [col for col in chinese_columns if col in df.columns]
    # 检查行业列是否存在于 DataFrame 中
    available_industry_cols = [col for col in industry_columns if col in df.columns]
    # 选出不需要生成嵌入的其他列 (既不是中文列也不是行业列)
    other_cols = [col for col in df.columns if col not in chinese_columns and col not in industry_columns]

    # 创建一个只包含其他列的 DataFrame 作为基础
    result_df = df[other_cols].copy()

    # --- 处理指定的行业层次结构 ---
    # 只要DataFrame中包含任何指定的行业列，就进行处理
    if available_industry_cols:
        industry_embeddings = process_industry_hierarchy(df)
        if industry_embeddings is not None:
            industry_emb_df = pd.DataFrame(
                industry_embeddings,
                columns=[f"industry_hier_emb_{i}" for i in range(industry_embeddings.shape[1])]
            )
            result_df = pd.concat([result_df, industry_emb_df], axis=1)
    else:
        print("  None of the specified industry columns are present in the data.")

    # --- 为其他中文列生成增强嵌入 ---
    embed_dfs = []
    for col in valid_chinese_cols:
        print(f"  Processing column: {col}")
        texts = df[col].fillna("").tolist()
        embeddings = get_enhanced_bert_embedding(texts, batch_size=32)

        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_enhanced_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embed_dfs.append(embed_df)

    # 如果生成了嵌入，则将它们与基础 DataFrame 拼接起来
    if embed_dfs:
        all_embeddings = pd.concat(embed_dfs, axis=1)
        result_df = pd.concat([result_df, all_embeddings], axis=1)

    return result_df


# === 5. 主执行流程 ===
if __name__ == "__main__":
    # --- 读取数据 ---
    print("Loading data...")
    # 请根据你的实际文件路径修改
    train_csv_path = "../data/train.csv"
    test_csv_path = "../data/test.csv"

    if not os.path.exists(train_csv_path):
         raise FileNotFoundError(f"训练数据文件 {train_csv_path} 不存在。")
    if not os.path.exists(test_csv_path):
         raise FileNotFoundError(f"测试数据文件 {test_csv_path} 不存在。")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # --- 处理训练集 ---
    train_processed = process_data(train_df, "Train Data")

    # --- 处理测试集 ---
    test_processed = process_data(test_df, "Test Data")

    # --- 保存为 Parquet 格式 ---
    print("Saving results to Parquet files...")
    # 输出文件路径
    train_output_path = "train_bert_enhanced_embedded_hier.parquet"
    test_output_path = "test_bert_enhanced_embedded_hier.parquet"

    train_processed.to_parquet(train_output_path, index=False)
    test_processed.to_parquet(test_output_path, index=False)

    print(f"处理完成！")
    print(f"训练集增强嵌入已保存至: {train_output_path} (Shape: {train_processed.shape})")
    print(f"测试集增强嵌入已保存至: {test_output_path} (Shape: {test_processed.shape})")
