# -*- coding: utf-8 -*-
"""增强版机器学习管道，实现了优化的过采样策略、动态特征交互构建和特征分级处理。"""

import os
import pickle
import warnings
import re  # 导入正则表达式库用于特征分级
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
# 注意：KNNImputer 对于分类特征（如目标编码后的）可能不是最佳选择，但在此简化示例中仍使用。
# 实际应用中，可能需要对不同类型的特征使用不同的插补策略。
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, fbeta_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

# 尝试导入 giotto-tda 用于拓扑数据分析
try:
    from gtda.homology import VietorisRipsPersistence

    TOPOLOGY_AVAILABLE = True
except ImportError:
    print("警告: 未找到 gtda (giotto-tda)。将不会计算拓扑特征。")
    TOPOLOGY_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_object(obj, filepath):
    """将Python对象序列化保存到文件"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存对象至: {filepath}")


# === 新增函数：特征分级体系 ===
# 建立特征分级体系
# 注意：这里的特征名是示例，需要根据实际数据集的列名进行调整。
FEATURE_HIERARCHY = {
    "core": ['f0', 'f1', 'f2'],  # 示例核心特征，对应原始数据的前三个特征
    "secondary": [r'f[3-5]'],  # 示例次级特征，匹配 f3, f4, f5
    "contextual": [r'f[6-9]']  # 示例上下文特征，匹配 f6 到 f9
}


# === 新增函数：目标编码器 ===
class TargetEncoder(BaseEstimator, TransformerMixin):
    """简单的均值目标编码器"""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.mapping_ = {}

    def fit(self, X, y):
        # 假设 X 是一维或二维数组，且为分类特征
        X = np.asarray(X).flatten()
        y = np.asarray(y)
        unique_vals = np.unique(X)
        global_mean = np.mean(y)

        for val in unique_vals:
            mask = (X == val)
            count = np.sum(mask)
            mean_val = np.mean(y[mask])
            # 应用平滑
            smoothed_mean = (count * mean_val + self.smoothing * global_mean) / (count + self.smoothing)
            self.mapping_[val] = smoothed_mean
        return self

    def transform(self, X):
        X = np.asarray(X).flatten()
        # 对于未见过的类别，使用全局均值
        default_val = np.mean(list(self.mapping_.values())) if self.mapping_ else 0
        return np.array([self.mapping_.get(x, default_val) for x in X]).reshape(-1, 1)


# === 新增函数：分箱离散化 ===
def bin_features(X, n_bins=5):
    """对特征进行分箱离散化"""
    if X.size == 0:
        return X
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_binned = discretizer.fit_transform(X)
    return X_binned


# === 新增函数：分级特征处理 ===
def hierarchical_processing(X, feature_names, y=None):
    """
    分级特征处理
    核心财务特征：保留原始数值
    次级特征：分箱离散化
    上下文特征：目标编码 (需要标签y)
    """
    print("正在执行分级特征处理...")
    core_idx = [i for i, n in enumerate(feature_names)
                if any(n == p for p in FEATURE_HIERARCHY['core'])]

    sec_idx = [i for i, n in enumerate(feature_names)
               if any(re.match(p, n) for p in FEATURE_HIERARCHY['secondary'])]

    ctx_idx = [i for i, n in enumerate(feature_names)
               if any(re.match(p, n) for p in FEATURE_HIERARCHY['contextual'])]

    processed_parts = []
    processed_names = []

    # 核心特征：直接保留
    if core_idx:
        processed_parts.append(X[:, core_idx])
        processed_names.extend([feature_names[i] for i in core_idx])
        print(f"  - 核心特征 ({len(core_idx)} 个): 已保留原始数值")

    # 次级特征：分箱
    if sec_idx:
        X_sec = bin_features(X[:, sec_idx])
        processed_parts.append(X_sec)
        processed_names.extend([f"{feature_names[i]}_binned" for i in sec_idx])
        print(f"  - 次级特征 ({len(sec_idx)} 个): 已进行分箱离散化")

    # 上下文特征：目标编码 (如果提供了y)
    if ctx_idx:
        if y is None:
            print("  - 警告: 上下文特征存在但未提供标签y，将保留原始数值。")
            processed_parts.append(X[:, ctx_idx])
            processed_names.extend([feature_names[i] for i in ctx_idx])
        else:
            X_ctx_list = []
            ctx_names_out = []
            for i in ctx_idx:
                te = TargetEncoder()
                # 假设上下文特征是字符串或可以转换为字符串的类别
                feat_vals = X[:, i].astype(str)
                X_ctx_enc = te.fit_transform(feat_vals, y)
                X_ctx_list.append(X_ctx_enc.flatten())
                ctx_names_out.append(f"{feature_names[i]}_target_encoded")
                # 保存编码器以便后续使用
                # 在完整管道中，你可能需要将这些编码器保存起来
            if X_ctx_list:
                X_ctx_final = np.column_stack(X_ctx_list)
                processed_parts.append(X_ctx_final)
                processed_names.extend(ctx_names_out)
                print(f"  - 上下文特征 ({len(ctx_idx)} 个): 已进行目标编码")
            else:
                print("  - 上下文特征处理失败。")

    if processed_parts:
        X_processed = np.hstack(processed_parts)
        print(f"分级处理完成，输出特征数: {X_processed.shape[1]}")
        return X_processed, processed_names
    else:
        print("未找到匹配任何分级的特征，返回原始数据。")
        return X, feature_names


# === 新增函数：动态特征交互构建 ===
def generate_interactions(X, feature_names, feature_importances, top_k_ratio=0.2):
    """
    基于特征重要性创建交互项。
    选择重要性排名前 top_k_ratio 的特征进行两两组合。
    """
    print("正在生成特征交互项...")
    num_features = X.shape[1]
    if len(feature_importances) != num_features:
        raise ValueError("特征重要性数组长度与特征数量不匹配。")

    # 确定重要特征的阈值
    k = max(2, int(num_features * top_k_ratio))  # 至少选择2个特征
    important_indices = np.argsort(feature_importances)[-k:]

    interaction_features_X = []
    interaction_feature_names = []

    # 生成重要特征之间的所有组合
    for i, j in combinations(important_indices, 2):
        interaction_term = (X[:, i] * X[:, j]).reshape(-1, 1)
        interaction_features_X.append(interaction_term)
        interaction_feature_names.append(f"{feature_names[i]}_x_{feature_names[j]}")

    if interaction_features_X:
        interaction_features_X = np.hstack(interaction_features_X)
        X_with_interactions = np.hstack([X, interaction_features_X])
        print(f"生成了 {len(interaction_feature_names)} 个交互特征。")
        return X_with_interactions, interaction_feature_names
    else:
        print("未生成交互特征。")
        return X, []


# === 新增函数：动态拓扑尺度选择 ===
def adaptive_scale_selection(X, feature_importances):
    """根据特征重要性动态调整拓扑分析尺度"""
    if len(feature_importances) != X.shape[1]:
        raise ValueError("特征重要性的长度必须与X中的特征数量匹配。")

    scales = []
    for i, imp in enumerate(feature_importances):
        # 高重要性特征使用精细尺度
        if imp > np.quantile(feature_importances, 0.8):
            scales.append(0.05)  # 精细尺度
        # 中等重要性特征使用中等尺度
        elif imp > np.quantile(feature_importances, 0.5):
            scales.append(0.1)  # 中等尺度
        # 低重要性特征使用粗粒度尺度
        else:
            scales.append(0.2)  # 粗粒度尺度
    return np.array(scales)


def extract_topological_features(X_sample, homology_dims=[0, 1], scale_factor=1.0):
    """
    使用 giotto-tda 从单个样本中提取持续同调特征。
    注意：这在实践中效率很低，因为每次只处理一个样本。
    这只是一个概念演示。实际应用中应批量处理或寻找更高效的近似方法。
    新增 scale_factor 参数用于调整距离计算的尺度。
    """
    if not TOPOLOGY_AVAILABLE:
        # 如果拓扑库不可用，则返回固定大小的虚拟特征向量
        return np.zeros(5)  # 示例占位符

    try:
        # 确保输入是二维数组 (n_points, n_dimensions)
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        # 应用尺度因子到数据点上（影响欧几里得距离）
        scaled_X_sample = X_sample * scale_factor

        # gtda 要求输入是三维数组 (n_samples, n_points, n_dimensions)
        # 我们只有一个样本，所以形状是 (1, n_points, n_dimensions)
        # 如果scaled_X_sample已经是 (n_points, n_dimensions)，则需要reshape
        if scaled_X_sample.ndim == 2:
            X_diagram_input = scaled_X_sample.reshape(1, scaled_X_sample.shape[0], scaled_X_sample.shape[1])
        else:
            # 此情况处理如果不知何故它已经是3D但需要是单一样本
            X_diagram_input = scaled_X_sample.reshape(1, -1, scaled_X_sample.shape[-1])

        # 初始化 VietorisRipsPersistence
        persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=homology_dims,
            collapse_edges=True,
            max_edge_length=np.inf,
            infinity_values=None,  # 让 gtda 处理 inf
            n_jobs=1  # 并行处理有时会在循环中引起问题
        )

        # 计算持续同调图
        diagrams = persistence.fit_transform(X_diagram_input)

        # 提取特征：例如，每个维度的条目数，最长寿命等
        topo_features = []
        diagram = diagrams[0]  # 我们只有一个样本

        for dim in homology_dims:
            mask = (diagram[:, 2] == dim)  # 在 gtda 中第三列是同调维数
            births_deaths_dim = diagram[mask][:, :2]  # 前两列是出生/死亡时间

            if len(births_deaths_dim) > 0:
                lifetimes = births_deaths_dim[:, 1] - births_deaths_dim[:, 0]
                # 添加一些基本统计信息作为特征
                topo_features.extend([
                    np.sum(mask),  # 组件/孔洞的数量
                    np.max(lifetimes) if len(lifetimes) > 0 else 0,  # 最大生命周期
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0,  # 平均生命周期
                    np.std(lifetimes) if len(lifetimes) > 1 else 0,  # 生命周期标准差
                ])
            else:
                topo_features.extend([0, 0, 0, 0])  # 如果此维度没有特征则填充零

        # 全局特征：所有维度的最大死亡时间
        max_death_global = np.max(diagram[:, 1]) if diagram.size > 0 else 0
        topo_features.append(max_death_global)

        return np.array(topo_features)

    except Exception as e:
        print(f"警告: 计算样本的拓扑特征时出错: {e}")
        # 出错时返回一致大小的零向量
        # 根据维度数量和每个维度的特征 + 全局特征调整大小
        expected_size = len(homology_dims) * 4 + 1
        return np.zeros(expected_size)


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """执行级联预测逻辑"""
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        pred_or_score = base_model.predict(X)
        if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
            probas_base = np.clip(pred_or_score.flatten(), 0, 1)
        else:
            probas_base = pred_or_score[:, 1]

    preds = (probas_base >= threshold).astype(int)
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

    if meta_model is not None and uncertain_mask.sum() > 0:
        if hasattr(meta_model, 'predict_proba'):
            try:
                meta_probas = meta_model.predict_proba(X[uncertain_mask])[:, 1]
                meta_preds = (meta_probas >= 0.5).astype(int)
            except:
                meta_preds = meta_model.predict(X[uncertain_mask])
        else:
            meta_preds = meta_model.predict(X[uncertain_mask])

        preds[uncertain_mask] = meta_preds.astype(int)

    return preds, probas_base


# --- 修改后的阈值搜索函数 (优化正类权重) ---
def cost_sensitive_threshold(y_true, probas, recall_weight=1.0, precision_weight=1.0, specificity_weight=0.0):
    """
    优化阈值搜索，使正类（少数类）权重更高。
    """
    min_threshold = 0.2
    max_threshold = 0.8
    thresholds = np.linspace(min_threshold, max_threshold, 101)
    print(
        f"正在优化阈值搜索 (范围: [{min_threshold}, {max_threshold}], 权重: Recall={recall_weight}, Precision={precision_weight}, Specificity={specificity_weight})...")

    best_t, best_score = 0.5, -np.inf
    best_metrics = (0.0, 0.0, 0.0)

    for t in thresholds:
        preds = (probas >= t).astype(int)

        try:
            cm = confusion_matrix(y_true, preds)
            tn, fp, fn, tp = cm.ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        except Exception as e:
            print(f"警告: 在阈值={t}处计算指标时发生错误: {e}")
            recall, precision, specificity = 0.0, 0.0, 0.0

        final_score = recall_weight * recall + precision_weight * precision - specificity_weight * (1 - specificity)

        if final_score > best_score:
            best_score, best_t = final_score, t
            best_metrics = (recall, precision, specificity)

    print(f"阈值优化完成。最佳阈值: {best_t:.4f}, 最佳加权得分: {best_score:.4f}")
    print(
        f"  最佳阈值下的指标: Recall={best_metrics[0]:.4f}, Precision={best_metrics[1]:.4f}, Specificity={best_metrics[2]:.4f}")
    return best_t, best_score, best_metrics


# ========== 核心修改：添加生成样本质量控制 ==========
def quality_filter(generated, original, threshold=0.95):
    """基于KNN相似度过滤生成样本"""
    print("正在基于KNN相似度过滤生成的样本...")
    nn = NearestNeighbors(n_neighbors=5).fit(original)
    distances, _ = nn.kneighbors(generated)
    mean_distances = np.mean(distances, axis=1)
    mask = mean_distances < threshold
    filtered_samples = generated[mask]
    print(f"过滤掉了 {len(generated) - len(filtered_samples)} 个样本。保留了 {len(filtered_samples)} 个样本。")
    return filtered_samples


# --- RFA (Recursive Feature Addition) 实现 ---
def recursive_feature_addition(estimator, X, y, cv=None, scoring='roc_auc', min_features=5):
    """
    递归特征添加 (RFA) 算法，自动确定最优特征数量。
    """
    print("开始进行递归特征添加 (RFA)...")
    n_features = X.shape[1]

    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    selected_indices = []
    remaining_indices = list(range(n_features))
    scores_history = []

    # 跟踪所有特征的排名
    ranking_info = np.full(n_features, fill_value=n_features)  # 初始化为最坏可能的排名

    while len(selected_indices) < n_features and len(selected_indices) < 50:  # 限制最大特征数
        scores_with_candidates = []

        # 评估添加每个候选特征
        for i in remaining_indices:
            candidate_indices = selected_indices + [i]
            try:
                score = cross_val_score(estimator, X[:, candidate_indices], y, cv=cv, scoring=scoring).mean()
                scores_with_candidates.append((score, i))
            except Exception as e:
                print(f"警告: 评估特征 {i} 时出错: {e}")
                scores_with_candidates.append((-np.inf, i))

        if not scores_with_candidates:
            break

        scores_with_candidates.sort(reverse=True)
        best_score, best_idx = scores_with_candidates[0]

        # 基于性能提升停止条件
        if len(scores_history) > 0 and best_score <= max(scores_history) and len(selected_indices) >= min_features:
            print(f"检测到性能平台期。停止添加。最终选定特征数: {len(selected_indices)}")
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # 为新添加的特征分配排名
        ranking_info[best_idx] = len(selected_indices)

        scores_history.append(best_score)
        print(f"  已添加特征索引 {best_idx} (得分: {best_score:.4f}), 当前排名: {ranking_info[best_idx]}")

    # 未被选中的剩余特征获得比那些已被选中的更低的排名
    remaining_rank_counter = len(selected_indices) + 1
    for idx_in_remaining, original_idx in enumerate(sorted(remaining_indices)):
        ranking_info[original_idx] = remaining_rank_counter + idx_in_remaining

    print(f"RFA 完成，最终选择了 {len(selected_indices)} 个特征。")
    return np.array(selected_indices), scores_history, ranking_info


# --- WGAN-GP 相关定义 (含自注意力机制) ---

class SelfAttention(nn.Module):
    """自注意力模块"""

    def __init__(self, data_dim):
        super(SelfAttention, self).__init__()
        self.data_dim = data_dim
        self.feature_dim = 1
        self.query = nn.Linear(self.feature_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.value = nn.Linear(self.feature_dim, self.feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.feature_dim])).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.out_proj = nn.Linear(data_dim, data_dim)
        self.layer_norm = nn.LayerNorm(data_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.unsqueeze(-1)  # (B, D, 1)
        Q = self.query(x_reshaped)  # (B, D, 1)
        K = self.key(x_reshaped)  # (B, D, 1)
        V = self.value(x_reshaped)  # (B, D, 1)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, D, D)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, D, D)
        attended_values = torch.matmul(attention_weights, V)  # (B, D, 1)
        attended_values = attended_values.squeeze(-1)  # (B, D)
        attended_values = self.out_proj(attended_values)  # (B, D)

        out = self.layer_norm(x + attended_values)  # 残差连接 + 归一化
        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.data_dim = data_dim
        self.attention = SelfAttention(data_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, data_dim),
        )

    def forward(self, z):
        raw_output = self.model(z)
        attended_output = self.attention(raw_output)
        return attended_output


class Critic(nn.Module):
    def __init__(self, data_dim):
        super(Critic, self).__init__()
        self.data_dim = data_dim
        self.attention = SelfAttention(data_dim)
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, data):
        attended_data = self.attention(data)
        output = self.model(attended_data)
        return output


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgangp_resample(X_train, y_train, minority_class=1, latent_dim=100, epochs=300, batch_size=64, lr=0.0001,
                    device='cpu', n_critic=5, clip_value=0.01, lambda_gp=10, T_max=None, filter_threshold=0.95):
    """
    使用 WGAN-GP 对少数类样本进行过采样，并加入动态学习率调度 (余弦退火)。
    添加了生成样本质量控制过滤器。
    """
    print("开始 WGAN-GP 过采样 (带有余弦退火学习率调度)...")
    X_minority = X_train[y_train == minority_class]
    if len(X_minority) == 0:
        print("警告: 未发现少数类样本。无法执行 WGAN-GP 采样。")
        return X_train, y_train

    data_dim = X_train.shape[1]
    num_minority = len(X_minority)
    num_majority = len(y_train) - num_minority
    num_to_generate = num_majority - num_minority

    if num_to_generate <= 0:
        print("少数类已经平衡或处于主导地位。跳过 WGAN-GP 重采样。")
        return X_train, y_train

    print(f"当前少数类样本: {num_minority}, 多数类样本: {num_majority}")
    print(f"计划生成 {num_to_generate} 个合成少数类样本...")

    X_minority_tensor = torch.tensor(X_minority, dtype=torch.float32).to(device)

    generator = Generator(latent_dim, data_dim).to(device)
    critic = Critic(data_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    if T_max is None:
        T_max = epochs
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=T_max, eta_min=1e-6)
    scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=T_max, eta_min=1e-6)
    print(f"启用 CosineAnnealingLR, T_max={T_max}")

    generator.train()
    critic.train()

    batches_done = 0
    for epoch in range(epochs):

        # 每次生成器更新多次训练判别器
        for _ in range(n_critic):
            idx = np.random.choice(len(X_minority_tensor), size=batch_size,
                                   replace=False if len(X_minority_tensor) >= batch_size else True)
            real_data = X_minority_tensor[idx]

            optimizer_C.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z).detach()

            loss_critic = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))
            gradient_penalty = compute_gradient_penalty(critic, real_data.data, fake_data.data, device)
            c_loss = loss_critic + lambda_gp * gradient_penalty

            c_loss.backward()
            optimizer_C.step()

        # 每 n_critic 步训练一次生成器
        optimizer_G.zero_grad()
        gen_data = generator(torch.randn(batch_size, latent_dim, device=device))
        g_loss = -torch.mean(critic(gen_data))
        g_loss.backward()
        optimizer_G.step()

        # 更新学习率
        scheduler_G.step()
        scheduler_C.step()

        # 定期记录日志
        if (epoch + 1) % 100 == 0 or epoch == 0:
            current_lr_g = optimizer_G.param_groups[0]['lr']
            current_lr_c = optimizer_C.param_groups[0]['lr']
            print(
                f"轮次 [{epoch + 1}/{epochs}], "
                f"C 损失: {loss_critic.item():.4f}, "
                f"梯度惩罚(GP): {gradient_penalty.item():.4f}, "
                f"G 损失: {g_loss.item():.4f}, "
                f"学习率 G: {current_lr_g:.6f}, 学习率 C: {current_lr_c:.6f}"
            )

    # 训练完成后生成新样本
    generator.eval()
    with torch.no_grad():
        z_new = torch.randn(num_to_generate, latent_dim, device=device)
        generated_data = generator(z_new).cpu().numpy()

    # 应用质量过滤
    filtered_generated_data = quality_filter(generated_data, X_minority, threshold=filter_threshold)

    # 合并原始数据和生成的数据
    resampled_X = np.vstack([X_train, filtered_generated_data])
    new_labels = np.full(len(filtered_generated_data), minority_class)
    resampled_y = np.hstack([y_train, new_labels])

    counts_after_sampling = dict(zip(*np.unique(resampled_y, return_counts=True)))
    print(f"WGAN-GP 采样完成。过滤后生成了 {len(filtered_generated_data)} 个新样本。")
    print(f"重采样后的分布: {counts_after_sampling}")
    return resampled_X, resampled_y


# ======== 主程序入口及其它辅助函数 ========
if __name__ == "__main__":
    print("=" * 60)
    print("训练流水线启动:")
    print("- 方差筛选 -> 直接进入模型训练 (取消RFA)")
    print("- 移除特征聚类阶段。")
    print("- 缺失值填补方式变更为 KNNImputer (邻居数=5)。")
    print("- 为WGAN-GP增加了动态学习率调度 (CosineAnnealingLR)。")
    print("- 阈值选择针对阳性类别指标进行了优化。")
    print("- 级联校正器切换为GradientBoostingClassifier。")
    print("- 添加了拓扑数据分析(TDA)特征提取模块。")
    print("- 在拓扑构建中禁用了多尺度识别。")
    print("- 为WGAN-GP生成的样本添加了质量过滤。")
    print("- 添加了基于特征重要性的动态特征交互构建。")
    print("- 添加了特征分级处理 (核心/次级/上下文)。")
    print("=" * 60)

    # 配置区域
    DATA_PATH = "./clean.csv"
    TEST_SIZE = 0.10
    RANDOM_STATE = 42
    VARIANCE_THRESHOLD_VALUE = 0.0
    CV_FOLDS_FOR_RFE = 3
    USE_RFA = False  # 修改：不再使用RFA
    ADD_TOPOLOGICAL_FEATURES = True
    USE_ADAPTIVE_SCALES = False  # 再次禁用自适应缩放逻辑
    USE_HIERARCHICAL_PROCESSING = True  # 启用分级处理

    TOPO_HOMOLOGY_DIMS = [0, 1]

    WGANGP_CONFIG = {
        'latent_dim': 100,
        'epochs': 300,
        'batch_size': 64,
        'lr': 0.0001,
        'lambda_gp': 10,
        'n_critic': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'T_max': None,
        'filter_threshold': 0.95  # 添加过滤阈值
    }

    XGB_PARAMS = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor'
    }

    META_GBM_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 1.0,
        'criterion': 'friedman_mse',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.0,
        'init': None,
        'random_state': RANDOM_STATE + 1,
        'max_features': None,
        'verbose': 0,
        'max_leaf_nodes': None,
        'warm_start': False,
        'validation_fraction': 0.1,
        'n_iter_no_change': None,
        'tol': 1e-4,
        'ccp_alpha': 0.0
    }

    THRESHOLD_WEIGHTS = {
        'recall_weight': 2.0,
        'precision_weight': 1.5,
        'specificity_weight': 0.5
    }

    # 第一步：加载/创建数据集
    if not os.path.exists(DATA_PATH):
        print(f"未找到数据文件 '{DATA_PATH}'。正在创建示例数据集...")
        N_SAMPLES = 1000
        N_FEATURES_TOTAL = 20
        ZERO_VAR_FEATS = 2

        np.random.seed(RANDOM_STATE)
        SAMPLE_X_BASE = np.random.randn(N_SAMPLES, N_FEATURES_TOTAL - ZERO_VAR_FEATS)
        ZEROS_DATA_BLOCK = np.full((N_SAMPLES, ZERO_VAR_FEATS), 5.0)
        SAMPLE_X_FULL = np.hstack([SAMPLE_X_BASE, ZEROS_DATA_BLOCK])

        TARGET_LABELS_RAW = ((SAMPLE_X_FULL[:, 0] + SAMPLE_X_FULL[:, 1] - SAMPLE_X_FULL[:, 2]) > 0).astype(int)
        POSITIVE_INDICES_ALL = np.where(TARGET_LABELS_RAW == 1)[0]

        REMOVE_COUNT_POSITIVES = int(0.8 * len(POSITIVE_INDICES_ALL))
        if REMOVE_COUNT_POSITIVES > 0:
            INDICES_TO_REMOVE = np.random.choice(POSITIVE_INDICES_ALL, size=REMOVE_COUNT_POSITIVES, replace=False)
            SAMPLE_X_FINAL = np.delete(SAMPLE_X_FULL, INDICES_TO_REMOVE, axis=0)
            SAMPLE_Y_CLEANED = np.delete(TARGET_LABELS_RAW, INDICES_TO_REMOVE, axis=0)
        else:
            SAMPLE_X_FINAL = SAMPLE_X_FULL
            SAMPLE_Y_CLEANED = TARGET_LABELS_RAW

        FEATURE_NAMES_LIST = [f"f{i}" for i in range(N_FEATURES_TOTAL - ZERO_VAR_FEATS)] \
                             + [f"zero_var_{j}" for j in range(ZERO_VAR_FEATS)]

        EXAMPLE_DF = pd.DataFrame(SAMPLE_X_FINAL, columns=FEATURE_NAMES_LIST)
        EXAMPLE_DF['target'] = SAMPLE_Y_CLEANED
        EXAMPLE_DF.to_csv(DATA_PATH, index=False)
        print(f"...示例不平衡数据集已保存到 '{DATA_PATH}'")

    df_raw = pd.read_csv(DATA_PATH)
    if 'company_id' in df_raw.columns:
        df_cleaned = df_raw.drop(columns=["company_id"])
    else:
        df_cleaned = df_raw.copy()

    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    assert "target" in df_encoded.columns, "缺少 'target' 列."

    FEATURES_MATRIX_ALL = df_encoded.drop(columns=["target"]).values
    INITIAL_FEATURE_NAMES = df_encoded.drop(columns=["target"]).columns.tolist()
    LABEL_VECTOR_ALL = df_encoded["target"].values

    TOTAL_SAMPLES_BEFORE_PREPROCESSING = FEATURES_MATRIX_ALL.shape[0]
    POSITIVE_CLASS_COUNT_INITIAL = LABEL_VECTOR_ALL.sum()
    NEGATIVE_CLASS_COUNT_INITIAL = (LABEL_VECTOR_ALL == 0).sum()
    print(f"加载的数据形状: 特征({FEATURES_MATRIX_ALL.shape}), 标签({LABEL_VECTOR_ALL.shape}).")
    print(f"预处理前的类别分布: 正类={POSITIVE_CLASS_COUNT_INITIAL}, 负类={NEGATIVE_CLASS_COUNT_INITIAL}")

    # 第二步：方差筛选
    print(f"应用 VarianceFilter，阈值={VARIANCE_THRESHOLD_VALUE} ...")
    selector_variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD_VALUE)
    FEATURES_AFTER_VARIANCE_FILTERING = selector_variance_filter.fit_transform(FEATURES_MATRIX_ALL)
    SELECTED_FEATURE_IDX_POST_VARIANCE = selector_variance_filter.get_support(indices=True)
    SELECTED_FEATURE_NAMES_POST_VARIANCE = [INITIAL_FEATURE_NAMES[i] for i in SELECTED_FEATURE_IDX_POST_VARIANCE]
    print(
        f"经过 VarianceFilter 后: 保留了 {FEATURES_AFTER_VARIANCE_FILTERING.shape[1]} 个特征 (移除了 {len(INITIAL_FEATURE_NAMES) - len(SELECTED_FEATURE_NAMES_POST_VARIANCE)})")

    # ===================== 核心修改：添加特征分级处理 =====================
    print("\n执行增强型流水线: 添加特征分级处理...\n")

    if USE_HIERARCHICAL_PROCESSING:
        try:
            FEATURES_HIERARCHICAL_PROCESSED, NAMES_HIERARCHICAL_PROCESSED = hierarchical_processing(
                FEATURES_AFTER_VARIANCE_FILTERING, SELECTED_FEATURE_NAMES_POST_VARIANCE, LABEL_VECTOR_ALL
            )
            FEATURES_AFTER_HIERARCHICAL = FEATURES_HIERARCHICAL_PROCESSED
            FEATURE_NAMES_AFTER_HIERARCHICAL = NAMES_HIERARCHICAL_PROCESSED
            print(f"分级处理后特征数: {len(FEATURE_NAMES_AFTER_HIERARCHICAL)}")
        except Exception as e:
            print(f"分级处理失败: {e}。回退到原始特征。")
            FEATURES_AFTER_HIERARCHICAL = FEATURES_AFTER_VARIANCE_FILTERING
            FEATURE_NAMES_AFTER_HIERARCHICAL = SELECTED_FEATURE_NAMES_POST_VARIANCE
    else:
        FEATURES_AFTER_HIERARCHICAL = FEATURES_AFTER_VARIANCE_FILTERING
        FEATURE_NAMES_AFTER_HIERARCHICAL = SELECTED_FEATURE_NAMES_POST_VARIANCE

    # 在分级处理后进行插补和标准化
    print("对分级处理后的特征进行插补和标准化...")
    knn_imputer_instance = KNNImputer(n_neighbors=5)
    FEATURES_IMPUTED_NO_MISSING = knn_imputer_instance.fit_transform(FEATURES_AFTER_HIERARCHICAL)

    standard_scaler_instance = StandardScaler()
    FEATURES_SCALED_STANDARDIZED = standard_scaler_instance.fit_transform(FEATURES_IMPUTED_NO_MISSING)

    save_object(knn_imputer_instance, "./imputer.pkl")
    save_object(standard_scaler_instance, "./scaler.pkl")
    save_object(selector_variance_filter, "./variance_selector.pkl")
    # ===================== 特征分级处理结束 =====================

    # 第四步：添加拓扑特征（可选增强）
    FEATURES_WITH_OPTIONAL_TOPOLOGY = FEATURES_SCALED_STANDARDIZED
    TOPOLOGY_FEATURE_NAME_PREFIXES = []

    if ADD_TOPOLOGICAL_FEATURES and TOPOLOGY_AVAILABLE:
        print("使用持久同调提取拓扑签名...")
        TOPOLOGICAL_SIGNATURES_BATCH = []
        NUM_ROWS_IN_DATASET = FEATURES_SCALED_STANDARDIZED.shape[0]

        temp_xgb_estimator_for_importance = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != 'n_estimators'})
        temp_xgb_estimator_for_importance.set_params(n_estimators=100)
        temp_xgb_estimator_for_importance.fit(FEATURES_SCALED_STANDARDIZED, LABEL_VECTOR_ALL)
        ESTIMATED_FEATURE_IMPORTANCES_TEMPORARY = temp_xgb_estimator_for_importance.feature_importances_

        for row_index, ROW_OF_SAMPLE in enumerate(FEATURES_SCALED_STANDARDIZED):
            if (row_index + 1) % 200 == 0 or row_index == NUM_ROWS_IN_DATASET - 1:
                print(f"已完成第 {row_index + 1}/{NUM_ROWS_IN_DATASET} 行的拓扑特征处理.")

            TOPOLOGICAL_FEATURE_VECTOR_SINGLE_ROW = extract_topological_features(
                ROW_OF_SAMPLE,
                homology_dims=TOPO_HOMOLOGY_DIMS
            )
            TOPOLOGICAL_SIGNATURES_BATCH.append(TOPOLOGICAL_FEATURE_VECTOR_SINGLE_ROW)

        ARRAY_OF_TOPOLOGICAL_FEATURES = np.array(TOPOLOGICAL_SIGNATURES_BATCH)
        FEATURES_WITH_OPTIONAL_TOPOLOGY = np.hstack([FEATURES_SCALED_STANDARDIZED, ARRAY_OF_TOPOLOGICAL_FEATURES])

        COUNT_NEWLY_ADDED_TOPOLOGY_FEATURES = ARRAY_OF_TOPOLOGICAL_FEATURES.shape[1]
        TOPOLOGY_FEATURE_NAME_PREFIXES = [f'topo_feat_{idx}' for idx in range(COUNT_NEWLY_ADDED_TOPOLOGY_FEATURES)]

        print(
            f"成功追加了 {COUNT_NEWLY_ADDED_TOPOLOGY_FEATURES} 个拓扑描述符。新的总数: {FEATURES_WITH_OPTIONAL_TOPOLOGY.shape[1]}")


    elif ADD_TOPOLOGICAL_FEATURES and not TOPOLOGY_AVAILABLE:
        print("由于缺少依赖项 ('giotto-tda')，请求的拓扑分析已跳过。请安装软件包以启用该功能。")

    UPDATED_COMPLETE_FEATURE_SET_NAMES = FEATURE_NAMES_AFTER_HIERARCHICAL + TOPOLOGY_FEATURE_NAME_PREFIXES

    # ===================== 核心修改：取消特征交互构建 (根据要求) =====================
    print("\n根据要求，取消动态特征交互构建步骤。\n")
    FEATURES_WITH_INTERACTIONS = FEATURES_WITH_OPTIONAL_TOPOLOGY
    INTERACTION_FEATURE_NAMES = []
    FINAL_FEATURE_NAMES_WITH_INTERACTIONS = UPDATED_COMPLETE_FEATURE_SET_NAMES
    # ===================== 动态特征交互构建结束 =====================

    # ===================== 核心修改：取消RFA，直接使用分级处理后的特征=====================
    print("\n执行增强型流水线: 取消RFA，直接使用分级处理后的特征进行训练...\n")

    # 不再运行RFA，而是直接使用处理后的特征
    FEATURES_SELECTED_FINAL_OUTPUT = FEATURES_WITH_INTERACTIONS
    NAMES_FINAL_SELECTED_FEATURES = FINAL_FEATURE_NAMES_WITH_INTERACTIONS
    INDEX_MAPPING_LOCAL_TO_GLOBAL = np.arange(FEATURES_WITH_INTERACTIONS.shape[1])  # 回退到完整索引

    # 创建一个假的排名数组，表示所有特征都被选中并且排名相同（因为我们没有排序）
    FULL_RANKINGS_INFO_ARRAY = np.ones(FEATURES_WITH_INTERACTIONS.shape[1])

    # 打印最终使用的特征列表
    feature_list_str = '\n'.join(['  ' + str(name) for name in NAMES_FINAL_SELECTED_FEATURES])
    print(f"\n最终特征列表 (共 {len(NAMES_FINAL_SELECTED_FEATURES)} 个):\n{feature_list_str}\n")

    # 保存由修改后的流程得到的信息（模拟RFA的结果）
    save_object(FULL_RANKINGS_INFO_ARRAY, "./rfe_feature_ranking.pkl")
    # ====================== 核心修改结束 =======================

    print(f"特征工程后期总结: 最终特征计数 = {FEATURES_SELECTED_FINAL_OUTPUT.shape[1]}")

    # 继续标准ML工作流...
    TRAIN_INPUT_SPLIT, HOLDOUT_EVAL_SPLIT, TRAIN_TARGET_SPLIT, HOLDOUT_TARGET_SPLIT = train_test_split(
        FEATURES_SELECTED_FINAL_OUTPUT, LABEL_VECTOR_ALL, test_size=TEST_SIZE, stratify=LABEL_VECTOR_ALL,
        random_state=RANDOM_STATE
    )
    print(f"分割为训练集 ({TRAIN_INPUT_SPLIT.shape}) 和留出验证集 ({HOLDOUT_EVAL_SPLIT.shape})。")

    print(f"\n启动 WGAN-GP 数据扩充过程...")
    DISTRIBUTION_BEFORE_SAMPLING_TRAINSET = dict(Counter(TRAIN_TARGET_SPLIT))
    print(f"训练集中采样前的标签均衡状况: {DISTRIBUTION_BEFORE_SAMPLING_TRAINSET}")

    try:
        RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS = wgangp_resample(TRAIN_INPUT_SPLIT, TRAIN_TARGET_SPLIT,
                                                                           **WGANGP_CONFIG)
    except Exception as wg_error:
        print(f"WGAN-GP 异常失败: {wg_error}. 改为继续使用原始不平衡数据。")
        RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS = TRAIN_INPUT_SPLIT, TRAIN_TARGET_SPLIT

    print("建立主分类器 (XGBoost)...")
    DMATRIX_RESAMPLED_TRAIN = xgb.DMatrix(RESAMPLED_TRAIN_FEATURES, label=RESAMPLED_TRAIN_LABELS)
    DMATRIX_HOLDOUT_VALIDATION = xgb.DMatrix(HOLDOUT_EVAL_SPLIT, label=HOLDOUT_TARGET_SPLIT)

    EVALUATION_METRICS_CONTAINER = {}
    MODEL_TRAINED_USING_XGB_API = xgb.train(
        XGB_PARAMS,
        dtrain=DMATRIX_RESAMPLED_TRAIN,
        num_boost_round=XGB_PARAMS['n_estimators'],
        evals=[(DMATRIX_HOLDOUT_VALIDATION, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=EVALUATION_METRICS_CONTAINER
    )


    class WrappedXGBPredictiveModel:
        def __init__(self, booster_obj_ref):
            self.booster_internal_reference = booster_obj_ref

        def predict_proba(self, input_features_array_like):
            dmatrix_format_input = xgb.DMatrix(input_features_array_like)
            probabilities_positives_only = self.booster_internal_reference.predict(dmatrix_format_input)
            return np.column_stack([1 - probabilities_positives_only, probabilities_positives_only])

        def predict(self, input_features_array_like):
            dmatrix_format_input = xgb.DMatrix(input_features_array_like)
            predicted_probabilities = self.booster_internal_reference.predict(dmatrix_format_input)
            return (predicted_probabilities > 0.5).astype(int)


    WRAPPED_MODEL_WRAPPER_INSTANCE = WrappedXGBPredictiveModel(MODEL_TRAINED_USING_XGB_API)
    save_object(WRAPPED_MODEL_WRAPPER_INSTANCE, "./base_model_xgboost.pkl")
    print("基础学习器训练完毕并已存储。")

    print("识别难以分类的例子以供二级修正层使用...")
    _, PROBABILITIES_ON_TRAIN_SET_PREDICTED_BY_BASELINE = cascade_predict_single_model(WRAPPED_MODEL_WRAPPER_INSTANCE,
                                                                                       None, RESAMPLED_TRAIN_FEATURES,
                                                                                       threshold=0.5)
    PREDICTIONS_ON_TRAIN_SET_FROM_BASELINE = (PROBABILITIES_ON_TRAIN_SET_PREDICTED_BY_BASELINE >= 0.5).astype(int)

    MASK_MISCLASSIFIED_EXAMPLES = (PREDICTIONS_ON_TRAIN_SET_FROM_BASELINE != RESAMPLED_TRAIN_LABELS)
    HARD_CASE_FEATURE_SUBSET, HARD_CASE_TRUE_LABELS = RESAMPLED_TRAIN_FEATURES[MASK_MISCLASSIFIED_EXAMPLES], \
        RESAMPLED_TRAIN_LABELS[MASK_MISCLASSIFIED_EXAMPLES]

    _, BASELINE_PROBA_HOLDOUT_STAGE_ONE = cascade_predict_single_model(WRAPPED_MODEL_WRAPPER_INSTANCE, None,
                                                                       HOLDOUT_EVAL_SPLIT, threshold=0.5)

    if len(HARD_CASE_FEATURE_SUBSET) > 0 and len(np.unique(HARD_CASE_TRUE_LABELS)) > 1:
        print("利用误分类实例构建纠正模型...")
        CORRECTIVE_LEARNER_SECONDARY = GradientBoostingClassifier(**META_GBM_PARAMS)
        CORRECTIVE_LEARNER_SECONDARY.fit(HARD_CASE_FEATURE_SUBSET, HARD_CASE_TRUE_LABELS)
    else:
        print("遇到困难案例不足；无论如何初始化备用GBM。")
        CORRECTIVE_LEARNER_SECONDARY = GradientBoostingClassifier(**META_GBM_PARAMS)
        CORRECTIVE_LEARNER_SECONDARY.fit(RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS)

    save_object(CORRECTIVE_LEARNER_SECONDARY, "./meta_model.pkl")
    print("二次纠正组件成功构建。")

    print("通过定制准则确定最佳决策边界...")
    _, PROBABILITIES_HOLDOUT_COMBINED_PIPELINE = cascade_predict_single_model(WRAPPED_MODEL_WRAPPER_INSTANCE,
                                                                              CORRECTIVE_LEARNER_SECONDARY,
                                                                              HOLDOUT_EVAL_SPLIT, threshold=0.5)

    OPTIMAL_DECISION_BOUNDARY, BEST_ACHIEVED_SCORE, METRICS_AT_OPTIMAL_POINT = cost_sensitive_threshold(
        HOLDOUT_TARGET_SPLIT,
        PROBABILITIES_HOLDOUT_COMBINED_PIPELINE,
        **THRESHOLD_WEIGHTS
    )

    RECALL_BEST, PRECISION_BEST, SPECIFICITY_BEST = METRICS_AT_OPTIMAL_POINT

    PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH = (
                PROBABILITIES_HOLDOUT_COMBINED_PIPELINE >= OPTIMAL_DECISION_BOUNDARY).astype(int)

    ROC_AREA_UNDER_CURVE = roc_auc_score(HOLDOUT_TARGET_SPLIT, PROBABILITIES_HOLDOUT_COMBINED_PIPELINE)
    ACCURACY_BEST = accuracy_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH)
    RECALL_CALCULATED_AGAIN = recall_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH, zero_division=0)
    FBETA_MEASURE_ENHANCED_SENSITIVITY = fbeta_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH,
                                                     beta=1.2, zero_division=0)

    # <<<--- 关键修改在这里：计算精确率 Precision --->>>
    PRECISION_CALCULATED = precision_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH, zero_division=0)
    # <<<--------------------------------------------------->>>

    # 自定义最终评价分数公式 (注意这里也做了修改)
    # CUSTOM_EVALUATION_SCORE_FINAL = 50*ROC_AREA_UNDER_CURVE + 20*ACCURACY_BEST + 30*RECALL_CALCULATED_AGAIN
    CUSTOM_EVALUATION_SCORE_FINAL = 50 * ROC_AREA_UNDER_CURVE + 20 * PRECISION_CALCULATED + 30 * RECALL_CALCULATED_AGAIN

    print(f"\n选定的最佳操作点 (阈值): {OPTIMAL_DECISION_BOUNDARY:.4f}")
    print("关联的评估指标:")
    print(f"  AUC = {ROC_AREA_UNDER_CURVE:.5f}")
    print(f"  Accuracy = {ACCURACY_BEST:.5f}")
    # <<<--- 关键修改在这里：打印精确率 Precision --->>>
    print(f"  Precision = {PRECISION_CALCULATED:.5f}")
    # <<<------------------------------------------>>>
    print(f"  Recall/Sensitivity = {RECALL_CALCULATED_AGAIN:.5f}")
    # <<<--- 关键修改在这里：更新打印的加权总分说明 --->>>
    print(f"  加权总分为 (50*AUC + 20*Precision + 30*Recall): {CUSTOM_EVALUATION_SCORE_FINAL:.5f}")

    PIPELINE_ARTIFACT_DICTIONARY = {
        "imputer": knn_imputer_instance,
        "scaler": standard_scaler_instance,
        "variance_selector": selector_variance_filter,
        "tree_feature_ranking": FULL_RANKINGS_INFO_ARRAY,
        "selected_feature_names": NAMES_FINAL_SELECTED_FEATURES,
        "base_models": WRAPPED_MODEL_WRAPPER_INSTANCE,
        "base_types": ["XGBoost"],
        "meta_nb": CORRECTIVE_LEARNER_SECONDARY,
        "threshold": float(OPTIMAL_DECISION_BOUNDARY),
        "use_rfa": USE_RFA,
        "add_topological_features": ADD_TOPOLOGICAL_FEATURES,
        "topo_homology_dims": TOPO_HOMOLOGY_DIMS,
        "threshold_weights": THRESHOLD_WEIGHTS,
        "use_adaptive_scales": USE_ADAPTIVE_SCALES,
        "use_hierarchical_processing": USE_HIERARCHICAL_PROCESSING  # 保存配置
    }
    save_object(PIPELINE_ARTIFACT_DICTIONARY, "./model.pkl")
    print("\n完整的端到端建模产物已导出至 './model.pkl'")
    print("=" * 60)




