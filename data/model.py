# -*- coding: utf-8 -*-
"""增强版机器学习管道，实现了带特征聚类的递归特征添加(RFA)优化。"""

import os
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# 尝试导入 giotto-tda 用于拓扑数据分析
try:
    from gtda.homology import VietorisRipsPersistence

    TOPOLOGY_AVAILABLE = True
except ImportError:
    print("Warning: gtda (giotto-tda) not found. Topological features will NOT be computed.")
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
    print(f"Saved object to: {filepath}")


# === 新增函数：动态拓扑尺度选择 ===
def adaptive_scale_selection(X, feature_importances):
    """根据特征重要性动态调整拓扑分析尺度"""
    if len(feature_importances) != X.shape[1]:
        raise ValueError("Feature importance length must match number of features in X.")

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
        # Return a dummy feature vector of fixed size if topology library is not available
        return np.zeros(5)  # Example placeholder

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
            # This case handles if somehow it's already 3D but needs to be single sample
            X_diagram_input = scaled_X_sample.reshape(1, -1, scaled_X_sample.shape[-1])

        # 初始化 VietorisRipsPersistence
        persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=homology_dims,
            collapse_edges=True,
            max_edge_length=np.inf,
            infinity_values=None,  # Let gtda handle inf
            n_jobs=1  # Parallel processing can sometimes cause issues in loops
        )

        # 计算持续同调图
        diagrams = persistence.fit_transform(X_diagram_input)

        # 提取特征：例如，每个维度的条目数，最长寿命等
        topo_features = []
        diagram = diagrams[0]  # We have only one sample

        for dim in homology_dims:
            mask = (diagram[:, 2] == dim)  # The third column is the homology dimension in gtda
            births_deaths_dim = diagram[mask][:, :2]  # First two columns are birth/death

            if len(births_deaths_dim) > 0:
                lifetimes = births_deaths_dim[:, 1] - births_deaths_dim[:, 0]
                # Add some basic stats as features
                topo_features.extend([
                    np.sum(mask),  # Number of components/holes
                    np.max(lifetimes) if len(lifetimes) > 0 else 0,  # Max lifetime
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0,  # Mean lifetime
                    np.std(lifetimes) if len(lifetimes) > 1 else 0,  # Std lifetime
                ])
            else:
                topo_features.extend([0, 0, 0, 0])  # Fill with zeros if no features for this dim

        # Global feature: maximum death time across all dimensions
        max_death_global = np.max(diagram[:, 1]) if diagram.size > 0 else 0
        topo_features.append(max_death_global)

        return np.array(topo_features)

    except Exception as e:
        print(f"Warning: Error computing topological features for a sample: {e}")
        # Return a zero vector of consistent size on failure
        # Adjust size based on number of dims and features per dim + global feature
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
        f"Optimizing threshold search (range: [{min_threshold}, {max_threshold}], weights: Recall={recall_weight}, Precision={precision_weight}, Specificity={specificity_weight})...")

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
            print(f"Warning during metrics calculation at threshold={t}: {e}")
            recall, precision, specificity = 0.0, 0.0, 0.0

        final_score = recall_weight * recall + precision_weight * precision - specificity_weight * (1 - specificity)

        if final_score > best_score:
            best_score, best_t = final_score, t
            best_metrics = (recall, precision, specificity)

    print(f"Threshold optimization completed. Best threshold: {best_t:.4f}, Best weighted score: {best_score:.4f}")
    print(
        f"  Metrics at best threshold: Recall={best_metrics[0]:.4f}, Precision={best_metrics[1]:.4f}, Specificity={best_metrics[2]:.4f}")
    return best_t, best_score, best_metrics


# ========== 核心修改：引入特征聚类 ==========
def feature_clustering(X, n_clusters_ratio=0.5):
    """
    基于特征间相关性的聚类，以减少冗余特征。

    Args:
        X: 输入特征矩阵 (numpy array)。
        n_clusters_ratio: 目标簇数量相对于原始特征数的比例。

    Returns:
        cluster_labels: 每个特征所属的簇标签。
        representative_indices: 每个簇中选出的一个代表性特征索引。
    """
    print("Performing feature clustering to reduce redundancy...")
    corr_matrix = np.abs(np.corrcoef(X.T))  # Compute absolute correlation matrix
    distance_matrix = 1 - corr_matrix  # Convert correlation to distance

    # Determine number of clusters
    n_original_features = X.shape[1]
    n_clusters_target = max(1, int(n_original_features * n_clusters_ratio))
    print(f"Target number of clusters after agglomeration: {n_clusters_target}")

    # Perform Agglomerative Clustering using precomputed distances
    agglo = FeatureAgglomeration(
        n_clusters=n_clusters_target,
        linkage="complete",  # Complete linkage tends to produce compact clusters
        affinity="precomputed"  # Use our custom distance matrix
    )

    # Fit the model and get cluster labels for each original feature
    cluster_labels = agglo.fit_predict(distance_matrix)
    print(f"Clustering resulted in {len(set(cluster_labels))} distinct clusters.")

    # Select one representative feature per cluster (the one with highest variance)
    representative_indices = []
    for cluster_id in set(cluster_labels):
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]

        # Calculate variances for features within the current cluster
        vars_in_cluster = np.var(X[:, indices_in_cluster], axis=0)

        # Find index (within the subset) that has maximum variance
        idx_of_max_var_within_cluster = np.argmax(vars_in_cluster)

        # Map back to original feature space index
        selected_feature_index = indices_in_cluster[idx_of_max_var_within_cluster]
        representative_indices.append(selected_feature_index)

    print(f"Selected {len(representative_indices)} representative features from clusters.")
    return cluster_labels, np.array(representative_indices)


# --- RFA (Recursive Feature Addition) 实现 ---
def recursive_feature_addition(estimator, X, y, cv=None, scoring='roc_auc', min_features=5):
    """
    递归特征添加 (RFA) 算法，自动确定最优特征数量。
    修改后：这是唯一的特征选择步骤。
    """
    print("Starting Recursive Feature Addition (RFA)...")
    n_features = X.shape[1]

    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    selected_indices = []
    remaining_indices = list(range(n_features))
    scores_history = []

    # Track rankings for all features
    ranking_info = np.full(n_features, fill_value=n_features)  # Initialize with worst possible rank

    while len(selected_indices) < n_features and len(selected_indices) < 50:  # Limit max features
        scores_with_candidates = []

        # Evaluate adding each candidate feature
        for i in remaining_indices:
            candidate_indices = selected_indices + [i]
            try:
                score = cross_val_score(estimator, X[:, candidate_indices], y, cv=cv, scoring=scoring).mean()
                scores_with_candidates.append((score, i))
            except Exception as e:
                print(f"Warning evaluating feature {i}: {e}")
                scores_with_candidates.append((-np.inf, i))

        if not scores_with_candidates:
            break

        scores_with_candidates.sort(reverse=True)
        best_score, best_idx = scores_with_candidates[0]

        # Stop condition based on performance gain
        if len(scores_history) > 0 and best_score <= max(scores_history) and len(selected_indices) >= min_features:
            print(f"Performance plateau detected. Stopping addition. Optimal features count: {len(selected_indices)}")
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Assign rank to newly added feature
        ranking_info[best_idx] = len(selected_indices)

        scores_history.append(best_score)
        print(f"  Added feature index {best_idx} (Score: {best_score:.4f}), Current Rank: {ranking_info[best_idx]}")

    # Remaining unselected features get ranks lower than those selected
    remaining_rank_counter = len(selected_indices) + 1
    for idx_in_remaining, original_idx in enumerate(sorted(remaining_indices)):
        ranking_info[original_idx] = remaining_rank_counter + idx_in_remaining

    print(f"RFA completed, finally selected {len(selected_indices)} features.")
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

        out = self.layer_norm(x + attended_values)  # Residual connection + Norm
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
                    device='cpu', n_critic=5, clip_value=0.01, lambda_gp=10, T_max=None):
    """
    使用 WGAN-GP 对少数类样本进行过采样，并加入动态学习率调度 (余弦退火)。
    """
    print("Starting WGAN-GP oversampling (with cosine annealing LR schedule)...")
    X_minority = X_train[y_train == minority_class]
    if len(X_minority) == 0:
        print("Warning: No minority class samples found. Cannot perform WGAN-GP sampling.")
        return X_train, y_train

    data_dim = X_train.shape[1]
    num_minority = len(X_minority)
    num_majority = len(y_train) - num_minority
    num_to_generate = num_majority - num_minority

    if num_to_generate <= 0:
        print("Minority class already balanced or dominant. Skipping WGAN-GP resampling.")
        return X_train, y_train

    print(f"Current minority samples: {num_minority}, majority samples: {num_majority}")
    print(f"Planning to generate {num_to_generate} synthetic minority samples...")

    X_minority_tensor = torch.tensor(X_minority, dtype=torch.float32).to(device)

    generator = Generator(latent_dim, data_dim).to(device)
    critic = Critic(data_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    if T_max is None:
        T_max = epochs
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=T_max, eta_min=1e-6)
    scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=T_max, eta_min=1e-6)
    print(f"CosineAnnealingLR enabled, T_max={T_max}")

    generator.train()
    critic.train()

    batches_done = 0
    for epoch in range(epochs):

        # Train Critic multiple times per generator update
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

        # Train Generator once every n_critic steps
        optimizer_G.zero_grad()
        gen_data = generator(torch.randn(batch_size, latent_dim, device=device))
        g_loss = -torch.mean(critic(gen_data))
        g_loss.backward()
        optimizer_G.step()

        # Update learning rates
        scheduler_G.step()
        scheduler_C.step()

        # Log periodically
        if (epoch + 1) % 100 == 0 or epoch == 0:
            current_lr_g = optimizer_G.param_groups[0]['lr']
            current_lr_c = optimizer_C.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"C Loss: {loss_critic.item():.4f}, "
                f"GP: {gradient_penalty.item():.4f}, "
                f"G Loss: {g_loss.item():.4f}, "
                f"LR_G: {current_lr_g:.6f}, LR_C: {current_lr_c:.6f}"
            )

    # Generate new samples after training completes
    generator.eval()
    with torch.no_grad():
        z_new = torch.randn(num_to_generate, latent_dim, device=device)
        generated_data = generator(z_new).cpu().numpy()

    # Combine original and generated data
    resampled_X = np.vstack([X_train, generated_data])
    new_labels = np.full(num_to_generate, minority_class)
    resampled_y = np.hstack([y_train, new_labels])

    counts_after_sampling = dict(zip(*np.unique(resampled_y, return_counts=True)))
    print(f"WGAN-GP sampling finished. Generated {num_to_generate} new samples.")
    print(f"Distribution after resampling: {counts_after_sampling}")
    return resampled_X, resampled_y


# ======== 主程序入口及其它辅助函数 ========
if __name__ == "__main__":
    print("=" * 60)
    print("Training Pipeline Started:")
    print("- Variance Filter -> Feature Clustering -> Representative Selection -> RFA (Automatic Feature Count)")
    print("- Removed Balanced Random Forest stage.")
    print("- Missing value imputation changed to KNNImputer (n_neighbors=5).")
    print("- Dynamic learning rate scheduling added to WGAN-GP (CosineAnnealingLR).")
    print("- Threshold selection optimized for positive class metrics.")
    print("- Cascade corrector switched to GradientBoostingClassifier.")
    print("- Added Topological Data Analysis (TDA) feature extraction module.")
    print("- Disabled multi-scale recognition in topology construction.")
    print("=" * 60)

    # Configuration Section
    DATA_PATH = "./clean.csv"
    TEST_SIZE = 0.10
    RANDOM_STATE = 42
    VARIANCE_THRESHOLD_VALUE = 0.0
    CV_FOLDS_FOR_RFE = 3
    USE_RFA = True  # Always true now
    ADD_TOPOLOGICAL_FEATURES = True
    USE_ADAPTIVE_SCALES = False  # Disable adaptive scaling logic again

    CLUSTERING_RATIO = 0.7  # Retain ~70% of features via clustering representatives

    TOPO_HOMOLOGY_DIMS = [0, 1]

    WGANGP_CONFIG = {
        'latent_dim': 100,
        'epochs': 300,
        'batch_size': 64,
        'lr': 0.0001,
        'lambda_gp': 10,
        'n_critic': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'T_max': None
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

    # Step 1: Load/Create Dataset
    if not os.path.exists(DATA_PATH):
        print(f"Data file '{DATA_PATH}' not found. Creating example dataset...")
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
        print(f"...Example imbalanced dataset saved to '{DATA_PATH}'")

    df_raw = pd.read_csv(DATA_PATH)
    if 'company_id' in df_raw.columns:
        df_cleaned = df_raw.drop(columns=["company_id"])
    else:
        df_cleaned = df_raw.copy()

    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    assert "target" in df_encoded.columns, "'target' column missing."

    FEATURES_MATRIX_ALL = df_encoded.drop(columns=["target"]).values
    INITIAL_FEATURE_NAMES = df_encoded.drop(columns=["target"]).columns.tolist()
    LABEL_VECTOR_ALL = df_encoded["target"].values

    TOTAL_SAMPLES_BEFORE_PREPROCESSING = FEATURES_MATRIX_ALL.shape[0]
    POSITIVE_CLASS_COUNT_INITIAL = LABEL_VECTOR_ALL.sum()
    NEGATIVE_CLASS_COUNT_INITIAL = (LABEL_VECTOR_ALL == 0).sum()
    print(f"Loaded data shape: Features({FEATURES_MATRIX_ALL.shape}), Labels({LABEL_VECTOR_ALL.shape}).")
    print(
        f"Class distribution before preprocessing: Positive={POSITIVE_CLASS_COUNT_INITIAL}, Negative={NEGATIVE_CLASS_COUNT_INITIAL}")

    # Step 2: Variance Filtering
    print(f"Applying VarianceFilter with threshold={VARIANCE_THRESHOLD_VALUE} ...")
    selector_variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD_VALUE)
    FEATURES_AFTER_VARIANCE_FILTERING = selector_variance_filter.fit_transform(FEATURES_MATRIX_ALL)
    SELECTED_FEATURE_IDX_POST_VARIANCE = selector_variance_filter.get_support(indices=True)
    SELECTED_FEATURE_NAMES_POST_VARIANCE = [INITIAL_FEATURE_NAMES[i] for i in SELECTED_FEATURE_IDX_POST_VARIANCE]
    print(
        f"After VarianceFilter: retained {FEATURES_AFTER_VARIANCE_FILTERING.shape[1]} features "
        f"(removed {len(INITIAL_FEATURE_NAMES) - len(SELECTED_FEATURE_NAMES_POST_VARIANCE)})")

    # Step 3: Imputation & Scaling
    knn_imputer_instance = KNNImputer(n_neighbors=5)
    FEATURES_IMPUTED_NO_MISSING = knn_imputer_instance.fit_transform(FEATURES_AFTER_VARIANCE_FILTERING)

    standard_scaler_instance = StandardScaler()
    FEATURES_SCALED_STANDARDIZED = standard_scaler_instance.fit_transform(FEATURES_IMPUTED_NO_MISSING)

    save_object(knn_imputer_instance, "./imputer.pkl")
    save_object(standard_scaler_instance, "./scaler.pkl")
    save_object(selector_variance_filter, "./variance_selector.pkl")

    # Step 4: Add Topological Features (Optional Enhancement)
    FEATURES_WITH_OPTIONAL_TOPOLOGY = FEATURES_SCALED_STANDARDIZED
    TOPOLOGY_FEATURE_NAME_PREFIXES = []

    if ADD_TOPOLOGICAL_FEATURES and TOPOLOGY_AVAILABLE:
        print("Extracting topological signatures using persistent homology...")
        TOPOLOGICAL_SIGNATURES_BATCH = []
        NUM_ROWS_IN_DATASET = FEATURES_SCALED_STANDARDIZED.shape[0]

        temp_xgb_estimator_for_importance = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != 'n_estimators'})
        temp_xgb_estimator_for_importance.set_params(n_estimators=100)
        temp_xgb_estimator_for_importance.fit(FEATURES_SCALED_STANDARDIZED, LABEL_VECTOR_ALL)
        ESTIMATED_FEATURE_IMPORTANCES_TEMPORARY = temp_xgb_estimator_for_importance.feature_importances_

        for row_index, ROW_OF_SAMPLE in enumerate(FEATURES_SCALED_STANDARDIZED):
            if (row_index + 1) % 200 == 0 or row_index == NUM_ROWS_IN_DATASET - 1:
                print(f"Processed topological features for {row_index + 1}/{NUM_ROWS_IN_DATASET} rows.")

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
            f"Successfully appended {COUNT_NEWLY_ADDED_TOPOLOGY_FEATURES} topological descriptors. New total: {FEATURES_WITH_OPTIONAL_TOPOLOGY.shape[1]}")

    elif ADD_TOPOLOGICAL_FEATURES and not TOPOLOGY_AVAILABLE:
        print(
            "Requested topological analysis skipped due to missing dependency ('giotto-tda'). Install package to enable.")

    UPDATED_COMPLETE_FEATURE_SET_NAMES = SELECTED_FEATURE_NAMES_POST_VARIANCE + TOPOLOGY_FEATURE_NAME_PREFIXES

    # ===================== Core Modification Begins Here =====================
    print("\nExecuting enhanced pipeline: Feature Clustering followed by RFA...\n")

    # Stage A: Cluster Features to Reduce Redundancy
    _, REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING = feature_clustering(
        FEATURES_WITH_OPTIONAL_TOPOLOGY,
        n_clusters_ratio=CLUSTERING_RATIO
    )

    # Create reduced view of data containing only these representative features
    FEATURES_REDUCED_BY_CLUSTER_SELECTION = FEATURES_WITH_OPTIONAL_TOPOLOGY[:,
                                            REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING]
    NAMES_REPRESENTATIVES_ONLY = [UPDATED_COMPLETE_FEATURE_SET_NAMES[i] for i in
                                  REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING]

    # Stage B: Apply RFA Only On These Representatives
    estimator_used_by_rfa = xgb.XGBClassifier(**XGB_PARAMS)

    try:
        FINAL_SELECTED_FEATURE_INDICES_RELATIVE_TO_REPRESENTATIVES, SCORE_HISTORY_PER_STEP, FULL_RANKINGS_INFO_ARRAY = recursive_feature_addition(
            estimator_used_by_rfa,
            FEATURES_REDUCED_BY_CLUSTER_SELECTION,
            LABEL_VECTOR_ALL,
            cv=StratifiedKFold(n_splits=CV_FOLDS_FOR_RFE, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )

        # Map local indices back to full feature space (before reduction step)
        ABSOLUTE_INDICES_OF_CHOSEN_FEATURES = REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING[
            FULL_RANKINGS_INFO_ARRAY.argsort()]
        INDEX_MAPPING_LOCAL_TO_GLOBAL = REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING[
            FINAL_SELECTED_FEATURE_INDICES_RELATIVE_TO_REPRESENTATIVES]

        FEATURES_SELECTED_FINAL_OUTPUT = FEATURES_REDUCED_BY_CLUSTER_SELECTION[:,
                                         FINAL_SELECTED_FEATURE_INDICES_RELATIVE_TO_REPRESENTATIVES]
        NAMES_FINAL_SELECTED_FEATURES = [NAMES_REPRESENTATIVES_ONLY[i] for i in
                                         FINAL_SELECTED_FEATURE_INDICES_RELATIVE_TO_REPRESENTATIVES]

        print(
            f"\nFinal feature selection result: {len(NAMES_FINAL_SELECTED_FEATURES)} chosen.\nList:\n{'\\n'.join(['  ' + str(name) for name in NAMES_FINAL_SELECTED_FEATURES])}\n")

    except Exception as err_msg:
        print(f"Error occurred during RFA execution phase: {err_msg}")
        print("Falling back to clustered-reduced feature pool without further refinement.")
        FEATURES_SELECTED_FINAL_OUTPUT = FEATURES_REDUCED_BY_CLUSTER_SELECTION
        NAMES_FINAL_SELECTED_FEATURES = NAMES_REPRESENTATIVES_ONLY
        INDEX_MAPPING_LOCAL_TO_GLOBAL = REPRESENTATIVE_FEATURE_INDICES_FROM_CLUSTERING

    # Save comprehensive ranking info obtained from modified RFA function
    save_object(FULL_RANKINGS_INFO_ARRAY, "./rfe_feature_ranking.pkl")
    # ====================== Core Modification Ends =======================

    print(f"Post-feature engineering summary: Final feature count = {FEATURES_SELECTED_FINAL_OUTPUT.shape[1]}")

    # Continue with standard ML workflow...
    TRAIN_INPUT_SPLIT, HOLDOUT_EVAL_SPLIT, TRAIN_TARGET_SPLIT, HOLDOUT_TARGET_SPLIT = train_test_split(
        FEATURES_SELECTED_FINAL_OUTPUT, LABEL_VECTOR_ALL, test_size=TEST_SIZE, stratify=LABEL_VECTOR_ALL,
        random_state=RANDOM_STATE
    )
    print(f"Splitted into Training ({TRAIN_INPUT_SPLIT.shape}) and Holdout ({HOLDOUT_EVAL_SPLIT.shape}) sets.")

    print(f"\nInitiating WGAN-GP augmentation process...")
    DISTRIBUTION_BEFORE_SAMPLING_TRAINSET = dict(Counter(TRAIN_TARGET_SPLIT))
    print(f"Pre-WGAN-GP label balance in training split: {DISTRIBUTION_BEFORE_SAMPLING_TRAINSET}")

    try:
        RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS = wgangp_resample(TRAIN_INPUT_SPLIT, TRAIN_TARGET_SPLIT,
                                                                           **WGANGP_CONFIG)
    except Exception as wg_error:
        print(f"WGAN-GP failed unexpectedly: {wg_error}. Proceeding with raw imbalanced data instead.")
        RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS = TRAIN_INPUT_SPLIT, TRAIN_TARGET_SPLIT

    print("Building primary classifier (XGBoost)...")
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
    print("Base learner successfully trained and persisted.")

    print("Identifying hard-to-classify examples for secondary correction layer...")
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
        print("Constructing corrective ensemble using misclassified instances...")
        CORRECTIVE_LEARNER_SECONDARY = GradientBoostingClassifier(**META_GBM_PARAMS)
        CORRECTIVE_LEARNER_SECONDARY.fit(HARD_CASE_FEATURE_SUBSET, HARD_CASE_TRUE_LABELS)
    else:
        print("Insufficient difficult cases encountered; initializing fallback GBM anyway.")
        CORRECTIVE_LEARNER_SECONDARY = GradientBoostingClassifier(**META_GBM_PARAMS)
        CORRECTIVE_LEARNER_SECONDARY.fit(RESAMPLED_TRAIN_FEATURES, RESAMPLED_TRAIN_LABELS)

    save_object(CORRECTIVE_LEARNER_SECONDARY, "./meta_model.pkl")
    print("Secondary corrective component built successfully.")

    print("Determining optimal decision boundary via customized criterion...")
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
    F1_MACRO_BALANCED_ACCURACY = f1_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH, zero_division=0)
    FBETA_MEASURE_ENHANCED_SENSITIVITY = fbeta_score(HOLDOUT_TARGET_SPLIT, PREDICTIONS_HOLDOUT_AT_OPTIMAL_THRESH,
                                                     beta=1.2, zero_division=0)

    CUSTOM_EVALUATION_SCORE_FINAL = (
            THRESHOLD_WEIGHTS['recall_weight'] * RECALL_BEST +
            THRESHOLD_WEIGHTS['precision_weight'] * PRECISION_BEST -
            THRESHOLD_WEIGHTS['specificity_weight'] * (1 - SPECIFICITY_BEST)
    )

    print(f"\nChosen optimal operating point (threshold): {OPTIMAL_DECISION_BOUNDARY:.4f}")
    print("Associated evaluation metrics:")
    print(f"  Recall/Sensitivity = {RECALL_BEST:.5f}")
    print(f"  Area Under ROC Curve = {ROC_AREA_UNDER_CURVE:.5f}")
    print(f"  Precision/Positive Predictive Value = {PRECISION_BEST:.5f}")
    print(f"  Specificity/TNR = {SPECIFICITY_BEST:.5f}")
    print(f"  Harmonic Mean (F1-Score) = {F1_MACRO_BALANCED_ACCURACY:.4f}")
    print(f"  Weighted F-Beta Metric (β=1.2) = {FBETA_MEASURE_ENHANCED_SENSITIVITY:.4f}")
    print(f"  Composite Score Formula Applied: ")
    formula_str = (
        f"Recall×{THRESHOLD_WEIGHTS['recall_weight']} + "
        f"Precision×{THRESHOLD_WEIGHTS['precision_weight']} − "
        f"(1−Specificity)×{THRESHOLD_WEIGHTS['specificity_weight']}"
    )
    print(f"    {formula_str} = {CUSTOM_EVALUATION_SCORE_FINAL:.5f}")

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
        "use_adaptive_scales": USE_ADAPTIVE_SCALES
    }
    save_object(PIPELINE_ARTIFACT_DICTIONARY, "./model.pkl")
    print("\nComplete end-to-end modeling artifact exported to './model.pkl'")
    print("=" * 60)




