import os
import pickle
import warnings

import numpy as np
import pandas as pd
# 注意：如果环境中没有安装 imblearn，需要先通过 pip install imbalanced-learn 安装
from sklearn.metrics import confusion_matrix

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

# -----------------------------
# 工具函数
# -----------------------------
def load_object(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"已加载: {path}")
        return obj
    else:
        print(f"警告: {path} 不存在，跳过。")
        return None

def load_model(filename):
    """从文件加载模型"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"模型已从 '{filename}' 加载, 类型: {type(model)}")
    return model

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
# -----------------------------
# 1. 加载测试数据
# -----------------------------
test_data_path = "testClean.csv"
print(f"正在加载测试数据: {test_data_path}")
test_data = pd.read_csv(test_data_path)
print(f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1]}")

# -----------------------------
# 2. 加载预处理器（按训练时保存的）
# -----------------------------
imputer = load_object("./imputer.pkl")
scaler = load_object("./scaler.pkl")
selector = load_object("./feature_selector.pkl")

# -----------------------------
# 3. 加载模型管道
# -----------------------------
model_path = "model_pipeline.pkl"
print(f"正在加载模型管道: {model_path}")
loaded_pipeline = load_model(model_path)

# -----------------------------
# 4. 读取阈值
# -----------------------------
if hasattr(loaded_pipeline, "threshold"):
    threshold = loaded_pipeline.threshold
    print(f"从模型对象读取阈值: {threshold}")
else:
    threshold = 0.5
    print(f"警告: 模型对象中未找到阈值属性，使用默认值 {threshold}")

# -----------------------------
# 5. 测试数据处理
# -----------------------------
# 去掉 ID 列
feature_columns = [col for col in test_data.columns if col != "company_id"]
X_test_raw = test_data[feature_columns]
print(f"用于预测的原始特征数: {X_test_raw.shape[1]}")

# 依次应用预处理
X_test_proc = X_test_raw.copy()
if imputer is not None:
    X_test_proc = imputer.transform(X_test_proc)
    print("已应用缺失值填充。")

if scaler is not None:
    X_test_proc = scaler.transform(X_test_proc)
    print("已应用标准化。")

if selector is not None:
    X_test_proc = selector.transform(X_test_proc)
    print(f"已应用特征选择，最终特征数: {X_test_proc.shape[1]}")

# -----------------------------
# 6. 预测
# -----------------------------
print("正在进行预测...")
y_proba = loaded_pipeline.predict_proba(X_test_proc)[:, 1]
print(f"预测完成，共 {len(y_proba)} 个概率值。")

print(f"应用分类阈值: {threshold}")
y_pred = (y_proba >= threshold).astype(int)
print("分类完成。")

# -----------------------------
# 7. 结果输出
# -----------------------------
if "company_id" in test_data.columns:
    uuid_column = test_data["company_id"]
else:
    print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
    uuid_column = test_data.index

results_df = pd.DataFrame({
    "uuid": uuid_column,
    "proba": y_proba,
    "prediction": y_pred
})
print("结果数据框创建成功。")

output_path = r"C:\Users\YKSHb\Desktop\submit_template.csv"
print(f"正在保存结果到: {output_path}")
results_df.to_csv(output_path, index=False)
print(f"预测完成，阈值 {threshold}，结果已保存到 {output_path}")
