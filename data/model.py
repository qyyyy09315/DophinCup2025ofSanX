# train_ensemble_cascade_smoteen_catboost_dnn_torch.py
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict(base_models, meta_model, X, threshold=0.5):
    probas = np.zeros((X.shape[0], len(base_models)))
    for i, m in enumerate(base_models):
        try:
            probas[:, i] = m.predict_proba(X)[:, 1]
        except Exception:
            probas[:, i] = m.predict(X).astype(float)
    avg_proba = probas.mean(axis=1)

    preds = (avg_proba >= threshold).astype(int)

    disagreement = (probas > 0.5).sum(axis=1)
    uncertain_mask = (disagreement > 0) & (disagreement < len(base_models))

    if uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, avg_proba


class DeepFeatureSelector(nn.Module):
    """更深的全连接网络"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=200, batch_size=64, lr=1e-3, device="cpu"):
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")

    # 提取第一层权重
    first_layer = None
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            first_layer = layer
            break
    weights = first_layer.weight.detach().cpu().numpy()  # shape=(1024, input_dim)
    feature_importance = np.mean(np.abs(weights), axis=0)

    return feature_importance


def youden_threshold(y_true, probas):
    thresholds = np.linspace(0, 1, 101)
    best_t, best_j = 0.5, -1
    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        J = sensitivity + specificity - 1
        if J > best_j:
            best_j, best_t = J, t
    return best_t


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练：SMOTEENN -> CatBoost + 平衡随机森林 -> 级联 GaussianNB (Torch DNN 特征权重)")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    assert os.path.exists(data_path), f"未找到数据文件: {data_path}"
    test_size = 0.10
    random_state = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前计算设备: {device}")

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

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path).drop(columns=["company_id"])
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all==0).sum()}")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # ----- 2. DNN 特征权重 -----
    print("使用 Torch DNN 训练获取特征权重 ...")
    feature_importance = train_dnn_feature_selector_torch(X_scaled, y_all, input_dim=X_scaled.shape[1], device=device)
    ranked_idx = np.argsort(-feature_importance)
    X_selected = X_scaled[:, ranked_idx]
    print(f"Torch DNN 特征加权完成，特征总数={X_selected.shape[1]}")

    save_object(ranked_idx, "./dnn_feature_ranking.pkl")

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

    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_resampled, y_resampled)
    model_list.append(cat)
    model_types.append("CatBoost")

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

    # ----- 7. 阈值选择（Youden's J） -----
    _, holdout_probas = cascade_predict(model_list, meta_clf, X_holdout, threshold=0.5)
    best_thresh = youden_threshold(y_holdout, holdout_probas)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout)
    f1 = f1_score(y_holdout, y_pred_holdout)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"最佳阈值={best_thresh:.4f}, F1={f1:.4f}")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, FinalScore={final_score:.5f}")

    # ----- 8. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "dnn_feature_ranking": ranked_idx,
        "base_models": model_list,
        "base_types": model_types,
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model_pipeline_cascade_dnn_torch.pkl")
    print("已保存完整模型 ./model_pipeline_cascade_dnn_torch.pkl")
    print("=" * 60)
