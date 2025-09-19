import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier  # 替换 GaussianNB
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度器

# 设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)

# ------------------- Focal Loss（添加 epsilon 防止数值下溢） -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        # 添加 epsilon 防止 exp(-BCE_loss) 下溢为 0
        pt = torch.exp(-BCE_loss.clamp(max=50))  # clamp 防止数值过大，也可用 torch.clamp_min(pt, self.epsilon)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# ------------------- 自注意力模块 -------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = (Q * K) / self.scale
        weights = torch.softmax(scores, dim=-1)
        out = weights * V
        return out

# ------------------- 特征提取网络（仅保留自注意力）-------------------
class FeatureAttentionNet(nn.Module):
    def __init__(self, input_dim, weight_decay=1e-4):
        super().__init__()
        self.attention = SelfAttention(input_dim)
        self.classifier = nn.Linear(input_dim, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.attention(x)
        logits = self.classifier(x).squeeze(-1)
        return logits, x

# ------------------- 特征提取器 -------------------
class FeatureExtractor:
    def __init__(self, input_dim, epochs=500, lr=1e-3, batch_size=128,
                 weight_decay=1e-4, device=None, verbose=True):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.verbose = verbose

        self.net = FeatureAttentionNet(input_dim=input_dim, weight_decay=weight_decay).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)  # 添加学习率调度器
        self.criterion = FocalLoss()
        self.early_stopper = EarlyStopping(patience=50, min_delta=1e-4, monitor='loss')
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False

        if self.device.type == 'cuda':
            print(f"[FeatureExtractor] 模型已加载到 GPU {self.device}")
        else:
            print("[FeatureExtractor] 模型将使用 CPU")

        if self.verbose:
            print(f"[FeatureExtractor] 模型参数设备: {next(self.net.parameters()).device}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=0)

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.net(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * xb.size(0)
                count += xb.size(0)

            avg_loss = epoch_loss / max(1, count)
            self.scheduler.step()  # 更新学习率

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[FeatureExtractor] Epoch {epoch + 1}/{self.epochs}, loss = {avg_loss:.6f}, lr = {current_lr:.2e}")

            if self.early_stopper.step(avg_loss):
                if self.verbose:
                    print(f"[FeatureExtractor] Early stopping at epoch {epoch + 1}, loss={avg_loss:.6f}")
                break

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("FeatureExtractor 尚未 fit，请先 fit 后再 transform。")
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        embeddings = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                _, emb = self.net(xb)
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

# ------------------- EarlyStopping -------------------
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = np.inf if monitor == 'loss' else -np.inf
        self.wait = 0

    def step(self, value):
        stop = False
        if self.monitor == 'loss':
            if (self.best - value) > self.min_delta:
                self.best = value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
        else:
            if (value - self.best) > self.min_delta:
                self.best = value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
        return stop

# ------------------- 级联模型类 -------------------
class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, scaler, feature_extractor, classifiers):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.scaler = scaler
        self.feature_extractor = feature_extractor
        self.classifiers = classifiers  # [adaboost, xgboost]

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_scaled = self.scaler.transform(X_sel)
        return X_scaled

    def predict(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.feature_extractor.transform(X_scaled)
        preds = [clf.predict(emb) for clf in self.classifiers]
        final_pred = np.round(np.mean(preds, axis=0)).astype(int)  # Soft Voting
        return final_pred.ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.feature_extractor.transform(X_scaled)
        probas = [clf.predict_proba(emb)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])

# ------------------- 主流程 -------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，将使用 CPU 训练。")

    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)

    # 填补缺失值
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)

    # 删除常量特征
    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    # VarianceThreshold
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)

    # ADASYN 过采样
    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_selected, y_train_all)

    # 标准化
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # 特征提取器（基于自注意力机制）
    input_dim = X_resampled_scaled.shape[1]
    feature_extractor = FeatureExtractor(
        input_dim=input_dim,
        batch_size=128,
        verbose=True
    )
    feature_extractor.fit(X_resampled_scaled, y_resampled)

    # 生成 embedding（经过自注意力处理）
    X_emb_train = feature_extractor.transform(X_resampled_scaled)

    # AdaBoost 分类器
    adaboost = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=SEED
    )
    adaboost.fit(X_emb_train, y_resampled)

    # 替换为 XGBoost 分类器
    xgboost = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgboost.fit(X_emb_train, y_resampled)

    # 预测与评估
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_scaled = scaler.transform(X_test_selected)
    X_emb_test = feature_extractor.transform(X_test_scaled)

    # 使用两个分类器进行 soft voting
    y_proba_adaboost = adaboost.predict_proba(X_emb_test)[:, 1]
    y_proba_xgboost = xgboost.predict_proba(X_emb_test)[:, 1]
    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= 0.5).astype(int)

    # 评估指标
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    # 保存模型与预处理器
    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        scaler=scaler,
        feature_extractor=feature_extractor,
        classifiers=[adaboost, xgboost]  # 更新为 xgboost
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")  # 更新保存文件名

    print("模型和预处理器已保存。")

    # 最终模型是否在 GPU 上
    if torch.cuda.is_available():
        print("[Final Model] 模型是否在 GPU 上:", next(feature_extractor.net.parameters()).is_cuda)
    else:
        print("[Final Model] 当前不支持 GPU")

if __name__ == "__main__":
    main()