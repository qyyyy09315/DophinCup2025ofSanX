import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 导入 XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    print("警告: 未找到 xgboost 库。请运行 'pip install xgboost' 进行安装。")
    XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 引入进度条库

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """
    修改版cascade_predict，适用于单个基模型。
    修复了当meta_model为None时的错误处理
    """
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        probas_base = base_model.predict(X).astype(float)

    preds = (probas_base >= threshold).astype(int)

    # 简化的不确定性判断：这里我们假设当概率接近0.5时不确定
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)  # 可调整阈值

    # 只有当meta_model不为None且存在不确定样本时才使用meta_model
    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


# 定义一个带残差连接的块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # 主路径
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)  # 添加批归一化
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)  # 添加批归一化
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 残差连接路径 (如果输入输出维度不同)
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.bn1(out)  # 批归一化
        out = self.relu1(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)  # 批归一化
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
        # 使用残差块构建网络，修改为512-256-256-128-64结构
        self.block1 = ResidualBlock(input_dim, 512, 0.3)
        self.block2 = ResidualBlock(512, 256, 0.2)
        self.block3 = ResidualBlock(256, 256, 0.2)
        self.block4 = ResidualBlock(256, 128, 0.2)
        self.block5 = ResidualBlock(128, 64, 0.1)

        # 输出层
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=1000, batch_size=256, lr=1e-3, device="cpu", patience=50):
    """
    训练带有早停和学习率调度的深度特征选择器

    参数:
        X : array-like, shape (n_samples, n_features)
            输入特征.
        y : array-like, shape (n_samples,)
            目标变量.
        input_dim : int
            输入特征维度.
        epochs : int, optional (default=1000)
            最大训练轮数.
        batch_size : int, optional (default=128)
            批次大小.
        lr : float, optional (default=1e-3)
            初始学习率.
        device : str or torch.device, optional (default="cpu")
            训算设备 ("cpu" 或 "cuda").
        patience : int, optional (default=20)
            早停耐心值.

    返回:
        feature_importance : ndarray, shape (input_dim,)
            特征重要性分数.
    """
    # 确保模型在指定设备上
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()

    # 使用 AdamW 优化器，通常比 Adam 更好
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 添加学习率调度器，在验证损失停滞时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2
                                                     )

    # 将数据也移动到指定设备
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化早停相关变量
    best_loss = float('inf')
    trigger_times = 0
    early_stop = False

    model.train()

    # 使用tqdm显示训练进度条
    pbar = tqdm(range(epochs), desc="Training DNN Feature Selector")

    for epoch in pbar:
        total_loss = 0.0
        num_batches = 0

        for xb, yb in loader:
            yb = yb.unsqueeze(1)  # 调整标签形状
            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            # 添加梯度裁剪防止爆炸梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # 更新学习率调度器
        scheduler.step(avg_loss)

        # 更新进度条信息
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'LR': f'{current_lr:.2e}'})

        # --- Early Stopping Logic ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            # 在这里可以保存最佳模型状态字典，但因为我们只需要最终权重来计算特征重要性，
            # 并且不会恢复到这个检查点，所以省略了保存步骤。
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    # 训练结束后关闭进度条
    pbar.close()

    # 提取第一层权重以计算特征重要性
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
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(512, input_dim)
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
    if not XGB_AVAILABLE:
        print("错误: 缺少必要依赖库 xgboost。程序退出。")
        exit(1)

    print("=" * 60)
    print("开始训练：SMOTEENN -> XGBoost -> 级联 GaussianNB (Torch DNN 特征权重)")
    print("-> DNN 使用残差网络并利用 GPU (如果可用)，新增早停、学习率调度和进度条")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 创建示例数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到数据文件 '{data_path}'。正在创建示例数据...")
        # 示例数据生成代码（简化版）
        sample_n = 1000
        sample_features = 20
        np.random.seed(42)
        sample_X = np.random.randn(sample_n, sample_features)
        # 创建一些有意义的相关性
        sample_y = ((sample_X[:, 0] + sample_X[:, 1] - sample_X[:, 2]) > 0).astype(int)
        sample_df = pd.DataFrame(sample_X, columns=[f"f{i}" for i in range(sample_features)])
        sample_df['target'] = sample_y
        sample_df.to_csv(data_path, index=False)
        print(f"...示例数据已保存至 '{data_path}'")

    test_size = 0.10
    random_state = 42
    # 关键修改: 确定设备，优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # XGBoost 超参
    xgb_params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # 用于早停
        'random_state': random_state,
        'n_jobs': -1,
        # 启用 GPU (如果可用且配置了 GPU 支持的 XGBoost)
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
    }

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
    # 关键修改: 将设备传递给训练函数，并增加迭代次数
    feature_importance = train_dnn_feature_selector_torch(
        X_scaled, y_all,
        input_dim=X_scaled.shape[1],
        epochs=1000,  # 设置为1000次迭代
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

    # ----- 5. 训练基模型 (改为 XGBoost) -----
    print("训练 XGBoost 基模型 ...")
    # 准备 XGBoost 数据集以便使用早停功能（可选）
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    # 训练 XGBoost 模型
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,  # 可选：启用早停
        verbose_eval=False,  # 设置为True可以看到训练过程
        evals_result=evals_result
    )


    # 为了兼容后续的 predict_proba 接口，我们可以包装一下
    class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            # 返回二维数组 [[prob_0, prob_1], ...]
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost.pkl")
    print("已训练基模型：XGBoost")

    # ----- 6. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器...")
    # 使用新的预测方法，传入None作为meta_model
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

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
    # 使用新的预测函数
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
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
        "base_models": wrapped_xgb_model,  # 单一模型
        "base_types": ["XGBoost"],  # 类型列表
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model_pipeline_cascade_dnn_torch.pkl")
    print("已保存完整模型 ./model_pipeline_cascade_dnn_torch.pkl")
    print("=" * 60)



