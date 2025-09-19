import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.neighbors import NearestNeighbors

# 设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ------------------- 阈值搜索函数（优化 G-Mean）-------------------
def find_optimal_threshold(y_true, y_proba, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_threshold = 0.5
    best_gmean = 0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        gmean = np.sqrt(rec * prec) if rec > 0 and prec > 0 else 0
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = th
    return best_threshold, best_gmean

# ------------------- 类别特异性特征重要性（近似）-------------------
def compute_class_specific_feature_importance(X, y, model, feature_names=None):
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_importances_"):
        return None, None

    proba = model.predict_proba(X)[:, 1]
    importances = model.feature_importances_

    pos_idx = y == 1
    neg_idx = y == 0
    if pos_idx.sum() == 0 or neg_idx.sum() == 0:
        return None, None

    pos_weighted = np.mean(proba[pos_idx][:, np.newaxis] * importances, axis=0)
    neg_weighted = np.mean((1 - proba[neg_idx])[:, np.newaxis] * importances, axis=0)

    pos_weighted /= (np.sum(pos_weighted) + 1e-8)
    neg_weighted /= (np.sum(neg_weighted) + 1e-8)

    if feature_names is not None:
        pos_df = pd.DataFrame({'feature': feature_names, 'importance': pos_weighted}).sort_values('importance', ascending=False)
        neg_df = pd.DataFrame({'feature': feature_names, 'importance': neg_weighted}).sort_values('importance', ascending=False)
        return pos_df, neg_df
    return pos_weighted, neg_weighted

# ------------------- 模块1：MDWA —— 基于马氏距离与DBSCAN的特征权重计算（简化版）-------------------
def estimate_epsilon_with_k_distance(X, k=5):
    """使用 k-distance 曲线估计 DBSCAN 的 ε 参数"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)
    # 取第90百分位作为ε（避免噪声影响）
    epsilon = np.percentile(k_distances_sorted, 90)
    return max(epsilon, 1e-3)  # 避免过小

def get_dbscan_neighbors(x, samples, epsilon, min_samples=3):
    """使用 DBSCAN 获取 x 在 samples 中的密度可达邻居"""
    if len(samples) < min_samples:
        return []
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(samples)
    labels = clustering.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    # 找到与 x 同簇的样本（包括边界点）
    if len(samples) == 0:
        return []
    x_reshaped = x.reshape(1, -1)
    x_label = clustering.fit_predict(x_reshaped)[0]
    if x_label == -1:  # x 是噪声点
        return []
    neighbors_mask = labels == x_label
    neighbors = samples[neighbors_mask]
    return neighbors.tolist()

def MDWA(X, y, N=500):
    """
    计算每个特征的 Relief-F 改进权重（使用马氏距离 + DBSCAN）
    X: (n_samples, n_features)
    y: (n_samples,)
    N: 抽样次数，默认为 500
    """
    n_samples, n_features = X.shape
    if N is None:
        N = max(1, int(0.15 * n_samples))

    # 按类别划分样本
    classes = np.unique(y)
    class_samples = {c: X[y == c] for c in classes}

    # 初始化权重
    weights = np.zeros(n_features)

    # 为每个类别预计算协方差逆矩阵（加正则化）
    cov_inv_dict = {}
    reg = 1e-6
    for c in classes:
        samples_c = class_samples[c]
        if len(samples_c) < 2:
            cov_inv_dict[c] = np.eye(n_features)
        else:
            cov = np.cov(samples_c, rowvar=False)
            cov_reg = cov + reg * np.eye(n_features)
            try:
                cov_inv_dict[c] = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv_dict[c] = np.eye(n_features)

    # 开始抽样
    for _ in range(N):
        idx = np.random.randint(0, n_samples)
        x = X[idx]
        y_x = y[idx]

        # 获取同类近邻（使用DBSCAN）
        same_class_samples = class_samples[y_x]
        if len(same_class_samples) < 3:
            same_neighbors = []
        else:
            epsilon = estimate_epsilon_with_k_distance(same_class_samples, k=5)
            same_neighbors = get_dbscan_neighbors(x, same_class_samples, epsilon)

        # 获取异类近邻（每类分别处理）
        diff_neighbors_all = []
        for c in classes:
            if c == y_x:
                continue
            diff_samples = class_samples[c]
            if len(diff_samples) < 3:
                continue
            epsilon = estimate_epsilon_with_k_distance(diff_samples, k=5)
            diff_neighbors = get_dbscan_neighbors(x, diff_samples, epsilon)
            diff_neighbors_all.extend(diff_neighbors)

        # 更新每个特征的权重
        for i in range(n_features):
            penalty = 0.0
            reward = 0.0

            # 同类样本差异惩罚
            for neighbor in same_neighbors:
                penalty += abs(x[i] - neighbor[i])

            # 异类样本差异奖励
            for neighbor in diff_neighbors_all:
                reward += abs(x[i] - neighbor[i])

            # 平均更新
            n_same = len(same_neighbors) if len(same_neighbors) > 0 else 1
            n_diff = len(diff_neighbors_all) if len(diff_neighbors_all) > 0 else 1
            weights[i] += (reward / n_diff - penalty / n_same) / N

    return weights

# ------------------- 模块3：SSFS —— 分层抽样 + Gini 评分最终筛选（简化版）-------------------
def random_subset(X, y, sample_ratio=0.7, feature_ratio=0.9):
    """随机抽取样本和特征的子集"""
    n_samples, n_features = X.shape
    sample_size = max(1, int(sample_ratio * n_samples))
    feature_size = max(1, int(feature_ratio * n_features))

    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    feature_indices = np.random.choice(n_features, size=feature_size, replace=False)

    X_sub = X[sample_indices][:, feature_indices]
    y_sub = y[sample_indices]
    return X_sub, y_sub, feature_indices  # 返回特征映射

def random_forest_gini_importance(X, y, n_estimators=50, max_depth=7):
    """训练随机森林并返回 Gini 重要性"""
    if X.shape[1] == 0:
        return np.array([])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    return rf.feature_importances_

def SSFS_simplified(X, y, n1=3, gini_percentile=40):
    """
    简化版 SSFS 主算法：多轮随机子集 + Gini 筛选
    X: (n_samples, n_features)
    y: (n_samples,)
    n1: 子表数量
    gini_percentile: Gini 阈值百分位数（默认中位数50）
    """
    n_features = X.shape[1]
    R_combined = set()

    for i in range(n1):
        # 生成随机子表
        X_sub, y_sub, feat_map = random_subset(X, y, sample_ratio=0.7, feature_ratio=0.9)

        # 在子表上运行 MDWA
        weights_sub = MDWA(X_sub, y_sub, N=500)
        selected_sub_indices = np.argsort(weights_sub)[-10:]  # 保留 Top 10 特征（局部索引）

        # 映射回全局特征索引
        selected_global = [feat_map[idx] for idx in selected_sub_indices]
        R_combined.update(selected_global)

        # 转为列表用于训练
        R_list = list(R_combined)
        if len(R_list) == 0:
            continue

        # 计算 Gini 重要性
        X_selected = X[:, R_list]
        gini_scores = random_forest_gini_importance(X_selected, y, n_estimators=30, max_depth=5)

        if len(gini_scores) == 0:
            continue

        # 应用 Gini 阈值（百分位数）
        threshold = np.percentile(gini_scores, gini_percentile)
        keep_mask = gini_scores >= threshold
        R_combined = set(np.array(R_list)[keep_mask])

        print(f"SSFS 轮次 {i+1}/{n1}: 当前特征数 = {len(R_combined)}")

    return list(R_combined)

test_data = pd.read_csv(r'testClean.csv')

# 2. 加载完整的级联模型（已包含所有预处理步骤）
cascaded_model = joblib.load('ssfs_cascaded_model.pkl')

# 3. 应用预处理和模型预测
# 删除 'company_id' 列（如果存在）
X_test = test_data.drop(columns=['company_id'], errors='ignore')

# 直接使用级联模型进行预测
y_proba = cascaded_model.predict_proba(X_test)[:, 1]  # 类别1的概率
y_pred = cascaded_model.predict(X_test)  # 类别预测

# 4. 创建结果数据框
results_df = pd.DataFrame({
    'uuid': test_data['company_id'],  # 保留原始company_id
    'proba': y_proba,
    'prediction': y_pred
})

# 5. 保存结果
results_df.to_csv(r'C:\Users\YKSHb\Desktop\submit_template.csv', index=False)

print("预测结果已保存到 submit_template.csv")
