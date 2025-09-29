# train_ensemble_cascade_smoteen_adaboost_dnn_torch_fixed_with_residual.py
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from imblearn.ensemble import BalancedRandomForestClassifier
# 注意：AdaBoostClassifier 的 base_estimator 参数在 sklearn 新版本中已被弃用，应使用 estimator 参数替代
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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


# 定义一个带残差连接的块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # 主路径
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 残差连接路径 (如果输入输出维度不同)
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.relu1(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.linear2(out)
        # 残差连接
        out += identity
        out = self.relu2(out)
        if self.dropout2:
            out = self.dropout2(out)

        return out


class DeepFeatureSelector(nn.Module):
    """更深的全连接网络，包含残差连接"""

    def __init__(self, input_dim):
        super().__init__()
        # 使用残差块构建网络
        self.block1 = ResidualBlock(input_dim, 1024, 0.4)
        self.block2 = ResidualBlock(1024, 512, 0.3)
        self.block3 = ResidualBlock(512, 256, 0.3)
        self.block4 = ResidualBlock(256, 128, 0.2)

        # 输出层
        self.output_layer = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=200, batch_size=128, lr=1e-3, device="cpu"):
    # 确保模型在指定设备上
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()
    # 可以考虑使用 weight_decay 进行 L2 正则化
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将数据也移动到指定设备
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for xb, yb in loader:
            # 数据已经在设备上了，无需再次移动
            # xb, yb = xb.to(device), yb.to(device)
            yb = yb.unsqueeze(1)  # 调整标签形状
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        # 可选：打印平均损失
        # if (epoch + 1) % 50 == 0:
        #     avg_loss = total_loss / num_batches
        #     print(f"Epoch {epoch+1}/{epochs}, Avg Loss={avg_loss:.4f}")

    # 提取第一层权重以计算特征重要性
    # 注意：现在第一层是 ResidualBlock，我们需要访问其内部的 linear1 层
    first_linear_layer = None
    if hasattr(model, 'block1') and hasattr(model.block1, 'linear1'):
        first_linear_layer = model.block1.linear1
    else:
        # 如果结构改变，尝试查找第一个 nn.Linear 层
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear_layer = module
                break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(1024, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0)
    else:
        # 如果找不到线性层，则返回零重要性（理论上不应发生）
        print("警告：无法找到第一层线性层以计算特征重要性。")
        feature_importance = np.zeros(input_dim)

    return feature_importance


def youden_threshold(y_true, probas):
    thresholds = np.linspace(0, 1, 101)
    best_t, best_j = 0.5, -1
    for t in thresholds:
        preds = (probas >= t).astype(int)
        # 处理只有一个类别的情况
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # 所有预测都是同一类
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # All predicted negative
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # All predicted positive
        else:
            # 不规则矩阵，填充缺失项
            padded_cm = np.pad(cm, ((0, 2 - cm.shape[0]), (0, 2 - cm.shape[1])), mode='constant')
            tn, fp, fn, tp = padded_cm.ravel()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        J = sensitivity + specificity - 1
        if J > best_j:
            best_j, best_t = J, t
    return best_t


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练：SMOTEENN -> AdaBoost(DecisionTree) + 平衡随机森林 -> 级联 GaussianNB (Torch DNN 特征权重)")
    print("-> DNN 使用残差网络并利用 GPU (如果可用)")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 创建示例数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到数据文件 '{data_path}'。正在创建示例数据...")
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.rand(1000) * 100,
            'category_A': ['Y'] * 300 + ['N'] * 700,
            'target': ([1] * 150 + [0] * 150) + ([1] * 50 + [0] * 650)  # 制造不平衡
        })
        sample_data = pd.get_dummies(sample_data, drop_first=True)
        sample_data.to_csv(data_path, index=False)
        print(f"已生成示例数据并保存至 {data_path}")

    test_size = 0.10
    random_state = 42
    # 关键修改: 确定设备，优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # AdaBoost 超参 (关键修改点: 将 base_estimator 替换为 estimator)
    ada_params = {
        # 原始参数: "base_estimator": DecisionTreeClassifier(max_depth=3, random_state=random_state),
        # 修改后参数:
        "estimator": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "n_estimators": 300,
        "learning_rate": 0.05,
        "random_state": random_state,
        "algorithm": "SAMME"  # 显式指定算法以兼容新版本sklearn和DecisionTreeClassifier
    }

    # BalancedRandomForest 超参
    brf_n_estimators = 300
    brf_max_depth = 8
    brf_max_features = "sqrt"

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path)
    # 删除 company_id 列（如果存在）
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # 进行独热编码
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all == 0).sum()}")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # ----- 2. DNN 特征权重 -----
    print("使用 Torch DNN (带残差连接) 训练获取特征权重 ...")
    # 关键修改: 将设备传递给训练函数
    feature_importance = train_dnn_feature_selector_torch(
        X_scaled, y_all,
        input_dim=X_scaled.shape[1],
        device=device  # 使用指定的设备
    )
    ranked_idx = np.argsort(-feature_importance)
    # 根据特征重要性排序选择所有特征
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
    print(f"重采样后: X={X_resampled.shape}, 正类={y_resampled.sum()}, 负类={(y_resampled == 0).sum()}")

    # ----- 5. 训练基模型 -----
    model_list = []
    model_types = []

    # 关键修改: 使用更新后的参数初始化 AdaBoostClassifier
    ada = AdaBoostClassifier(**ada_params)
    ada.fit(X_resampled, y_resampled)
    model_list.append(ada)
    model_types.append("AdaBoost(DecisionTree)")

    brf = BalancedRandomForestClassifier(
        n_estimators=brf_n_estimators,
        max_depth=brf_max_depth,
        max_features=brf_max_features,
        random_state=random_state,
        n_jobs=-1  # 使用所有可用CPU核心
    )
    brf.fit(X_resampled, y_resampled)
    model_list.append(brf)
    model_types.append("BalancedRandomForest")

    save_object(model_list, "./base_model_list.pkl")
    save_object(model_types, "./base_model_types.pkl")
    print("已训练基模型：AdaBoost(DecisionTree) + BalancedRandomForest")

    # ----- 6. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器...")
    base_probas = np.zeros((X_resampled.shape[0], len(model_list)))
    for i, m in enumerate(model_list):
        base_probas[:, i] = m.predict_proba(X_resampled)[:, 1]
    avg_train_proba = base_probas.mean(axis=1)
    base_train_preds = (avg_train_proba >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:  # 确保困难样本集中至少有两个类别
        meta_clf = GaussianNB()
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器训练完成，困难样本数={len(X_hard)}")
    else:
        # 如果没有误分类样本或困难样本只属于一个类别，则使用全部重采样后的数据训练
        meta_clf = GaussianNB()
        meta_clf.fit(X_resampled, y_resampled)
        print("未发现足够的误分类样本，使用全部重采样数据训练修正器")

    # ----- 7. 阈值选择（Youden's J） -----
    _, holdout_probas = cascade_predict(model_list, meta_clf, X_holdout, threshold=0.5)
    best_thresh = youden_threshold(y_holdout, holdout_probas)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # 自定义评分公式 (注意：原注释中的系数总和不是100%，这里保持原样)
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
