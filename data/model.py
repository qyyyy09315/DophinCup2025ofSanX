# train_ensemble_cascade.py
import os
import pickle
import warnings
import math

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN  # 仅保留 import，如果不需要可删除
from imblearn.ensemble import BalancedRandomForestClassifier  # 保留以防后续替换
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def load_object(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练：负类分组 -> 多个 AdaBoost -> 级联 GaussianNB（元分类器）")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    assert os.path.exists(data_path), f"未找到数据文件: {data_path}"
    test_size = 0.10
    random_state = 42

    # AdaBoost 超参（可以调整）
    adaboost_n_estimators = 300
    adaboost_learning_rate = 0.5

    # SelectKBest 候选
    candidate_k = [30, 50, 80, 100]

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path).drop(columns=["company_id"])
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all==0).sum()}")

    # 处理缺失与缩放（并保存）
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # ----- 2. 特征选择（交叉验证选择 k） -----
    print("开始用交叉验证选择特征数 k ...")
    best_k = None
    best_auc = -np.inf
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for k in candidate_k:
        k_use = min(k, X_scaled.shape[1])
        selector_tmp = SelectKBest(score_func=f_classif, k=k_use)
        X_tmp = selector_tmp.fit_transform(X_scaled, y_all)
        # 用一个简单的 AdaBoost 评估 AUC（快速）
        from sklearn.ensemble import AdaBoostClassifier
        clf_tmp = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=random_state)
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(clf_tmp, X_tmp, y_all, cv=skf, scoring="roc_auc", n_jobs=2)
            mean_auc = np.mean(scores)
        except Exception:
            mean_auc = -np.inf
        print(f"  k={k_use}, CV AUC={mean_auc:.5f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_k = k_use

    print(f"选定特征数 k = {best_k}, CV AUC={best_auc:.5f}")
    selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = selector.fit_transform(X_scaled, y_all)
    save_object(selector, "./feature_selector.pkl")
    print(f"特征选择后维度: {X_selected.shape}")

    # ----- 3. 划分训练/测试集（用于最终评估与阈值选择） -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集(holdout): {X_holdout.shape}")

    # ----- 4. 负类分组并训练多个 AdaBoost（每组与所有正类组合） -----
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    print(f"训练集中正类数={n_pos}, 负类数={n_neg}")

    # 按与正类相同数量分组负类
    group_size = max(1, n_pos)  # 防止 n_pos = 0 的极端情况（若无正类需特殊处理）
    n_groups = math.ceil(n_neg / group_size)
    print(f"将负类分为 {n_groups} 组，每组最多 {group_size} 个样本")

    adaboost_list = []
    group_indices_list = []

    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, n_neg)
        neg_group_idx = neg_idx[start:end]
        # 合并正类与该负组，构建训练子集
        subset_idx = np.concatenate([pos_idx, neg_group_idx])
        np.random.shuffle(subset_idx)

        X_sub = X_train_full[subset_idx]
        y_sub = y_train_full[subset_idx]

        clf = AdaBoostClassifier(
            n_estimators=adaboost_n_estimators,
            learning_rate=adaboost_learning_rate,
            random_state=random_state
        )
        clf.fit(X_sub, y_sub)
        adaboost_list.append(clf)
        group_indices_list.append(subset_idx)
        print(f" 已训练 AdaBoost #{g+1}，子集大小={len(subset_idx)}，正例={sum(y_sub==1)}, 负例={sum(y_sub==0)}")

    # 保存 adaboost 列表（可能比较大）
    save_object(adaboost_list, "./adaboost_list.pkl")
    save_object(group_indices_list, "./adaboost_group_indices.pkl")

    # ----- 5. 构建元特征（用每个 AdaBoost 对全训练集和 holdout 产生概率） -----
    def stack_probas(adaboost_models, X):
        """返回 shape=(n_samples, n_models) 的正类概率矩阵"""
        proba_mat = np.zeros((X.shape[0], len(adaboost_models)))
        for i, m in enumerate(adaboost_models):
            try:
                proba = m.predict_proba(X)[:, 1]
            except Exception:
                # 万一 predict_proba 不可用（极端），退回 predict
                proba = m.predict(X).astype(float)
            proba_mat[:, i] = proba
        return proba_mat

    print("为训练集与 holdout 构建元概率特征...")
    X_train_meta = stack_probas(adaboost_list, X_train_full)  # (n_train, n_models)
    X_holdout_meta = stack_probas(adaboost_list, X_holdout)  # (n_holdout, n_models)
    print(f"元特征矩阵形状: train {X_train_meta.shape}, holdout {X_holdout_meta.shape}")

    # ----- 6. 在元特征上训练级联 GaussianNB（元分类器） -----
    meta_clf = GaussianNB()
    meta_clf.fit(X_train_meta, y_train_full)
    save_object(meta_clf, "./meta_nb.pkl")
    print("训练并保存元分类器 GaussianNB。")

    # ----- 7. 在 holdout 上评估并选择最佳阈值（以 F1 最大化为例） -----
    print("在 holdout 上评估以选择最佳阈值（使用 F1 最大化）...")
    holdout_probas = meta_clf.predict_proba(X_holdout_meta)[:, 1]
    # 搜索阈值
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (holdout_probas >= t).astype(int)
        f1 = f1_score(y_holdout, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"在 holdout 上选定阈值 (F1 最大): {best_thresh:.4f}, 对应 F1 = {best_f1:.4f}")

    # 也计算 recall/auc/precision 等供参考
    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout)
    final_score = 30 * recall + 50 * auc + 20 * precision
    print("holdout 评估：")
    print(f"  Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, final_score={final_score:.5f}")

    # ----- 8. 保存整体模型（包含所有必要组件与阈值） -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "adaboost_list": adaboost_list,
        "group_indices_list": group_indices_list,
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
        "feature_names": None  # 如果需要可以保存训练时的列名
    }
    save_object(model_dict, "./model_pipeline_ensemble.pkl")
    print("全部模型组件已保存为 ./model_pipeline_ensemble.pkl")

    # ----- 9. 在 holdout 上报告最终指标（再次打印） -----
    print("\n最终 holdout 指标（使用选定阈值）:")
    print(f"  Threshold = {best_thresh:.4f}")
    print(f"  Recall = {recall:.6f}")
    print(f"  AUC = {auc:.6f}")
    print(f"  Precision = {precision:.6f}")
    print(f"  F1 = {best_f1:.6f}")
    print("=" * 60)
