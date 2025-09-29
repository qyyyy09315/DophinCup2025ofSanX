# train_ensemble_cascade_smoteen_catboost.py
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict(base_models, meta_model, X, threshold=0.5):
    """
    级联预测：
    1. 所有基模型预测概率均值 -> 初步预测
    2. 如果样本预测分歧较大 -> 交给元分类器修正
    """
    probas = np.zeros((X.shape[0], len(base_models)))
    for i, m in enumerate(base_models):
        try:
            probas[:, i] = m.predict_proba(X)[:, 1]
        except Exception:
            probas[:, i] = m.predict(X).astype(float)
    avg_proba = probas.mean(axis=1)

    preds = (avg_proba >= threshold).astype(int)

    # 判断不确定样本：各模型预测分歧
    disagreement = (probas > 0.5).sum(axis=1)
    uncertain_mask = (disagreement > 0) & (disagreement < len(base_models))

    if uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, avg_proba


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练：SMOTEENN 重采样 -> CatBoost + 平衡随机森林 -> 级联修正 GaussianNB")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    assert os.path.exists(data_path), f"未找到数据文件: {data_path}"
    test_size = 0.10
    random_state = 42

    # CatBoost 超参
    cat_params = {
        "iterations": 500,
        "learning_rate": 0.001,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_seed": random_state
    }

    # BalancedRandomForest 超参
    brf_n_estimators = 300
    brf_max_depth = 8
    brf_max_features = "sqrt"

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

    # 缺失值处理 & 标准化
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # ----- 2. 特征选择 -----
    print("开始用交叉验证选择特征数 k ...")
    best_k, best_auc = None, -np.inf
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for k in candidate_k:
        k_use = min(k, X_scaled.shape[1])
        selector_tmp = SelectKBest(score_func=f_classif, k=k_use)
        X_tmp = selector_tmp.fit_transform(X_scaled, y_all)
        clf_tmp = CatBoostClassifier(**cat_params)
        try:
            scores = cross_val_score(clf_tmp, X_tmp, y_all, cv=skf, scoring="roc_auc", n_jobs=2)
            mean_auc = np.mean(scores)
        except Exception:
            mean_auc = -np.inf
        print(f"  k={k_use}, CV AUC={mean_auc:.5f}")
        if mean_auc > best_auc:
            best_auc, best_k = mean_auc, k_use

    print(f"选定特征数 k = {best_k}, CV AUC={best_auc:.5f}")
    selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = selector.fit_transform(X_scaled, y_all)
    save_object(selector, "./feature_selector.pkl")
    print(f"特征选择后维度: {X_selected.shape}")

    # ----- 3. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # ----- 4. SMOTEENN 重采样 -----
    print("正在进行 SMOTEENN 重采样 ...")
    smoteenn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train_full, y_train_full)
    print(f"重采样后: X={X_resampled.shape}, 正类={y_resampled.sum()}, 负类={(y_resampled==0).sum()}")

    # ----- 5. 训练基模型 -----
    model_list = []
    model_types = []

    # --- CatBoost ---
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_resampled, y_resampled)
    model_list.append(cat)
    model_types.append("CatBoost")

    # --- BalancedRandomForest ---
    brf = BalancedRandomForestClassifier(
        n_estimators=brf_n_estimators,
        max_depth=brf_max_depth,
        max_features=brf_max_features,
        random_state=random_state,
        n_jobs=2
    )
    brf.fit(X_resampled, y_resampled)
    model_list.append(brf)
    model_types.append("BalancedRandomForest")

    save_object(model_list, "./base_model_list.pkl")
    save_object(model_types, "./base_model_types.pkl")
    print("已训练基模型：CatBoost + BalancedRandomForest")

    # ----- 6. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器...")
    # 在训练集（重采样后）上预测，找到误分类样本
    base_probas = np.zeros((X_resampled.shape[0], len(model_list)))
    for i, m in enumerate(model_list):
        base_probas[:, i] = m.predict_proba(X_resampled)[:, 1]
    avg_train_proba = base_probas.mean(axis=1)
    base_train_preds = (avg_train_proba >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0:
        meta_clf = GaussianNB()
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器训练完成，困难样本数={len(X_hard)}")
    else:
        meta_clf = GaussianNB()
        meta_clf.fit(X_resampled, y_resampled)
        print("未发现误分类样本，使用全部训练集训练修正器")

    # ----- 7. 阈值选择（F1 最大化） -----
    final_preds, final_probas = cascade_predict(model_list, meta_clf, X_holdout, threshold=0.5)
    holdout_probas = final_probas
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (holdout_probas >= t).astype(int)
        f1 = f1_score(y_holdout, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"最佳阈值={best_thresh:.4f}, F1={best_f1:.4f}")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, FinalScore={final_score:.5f}")

    # ----- 8. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "base_models": model_list,
        "base_types": model_types,
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model_pipeline_cascade.pkl")
    print("已保存完整模型 ./model_pipeline_cascade.pkl")
    print("=" * 60)
