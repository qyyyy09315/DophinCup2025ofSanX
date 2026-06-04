import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import ADASYN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# 新增导入 AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# -------------------- 环境与随机种子 --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU (如有多卡可修改)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(8)


# -------------------- 损失 / 工具类 --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs: logits, targets: float {0,1}
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, monitor='loss'):
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


# -------------------- 自注意力模块 --------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5

    def forward(self, x):
        # x: (batch, dim)
        Q = self.query(x)  # (batch, dim)
        K = self.key(x)  # (batch, dim)
        V = self.value(x)  # (batch, dim)
        scores = (Q * K) / self.scale  # (batch, dim)
        weights = torch.softmax(scores, dim=-1)  # (batch, dim)
        out = weights * V  # (batch, dim)
        return out


# -------------------- DeepFeatureNet：简化结构 + 自注意力 --------------------
class DeepFeatureNet(nn.Module):
    def __init__(self, input_dim, embed_dim=32, weight_decay=1e-4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # attention layer on 128-d
        self.att1 = SelfAttention(128)

        self.final = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # head for classification
        self.head = nn.Linear(embed_dim, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.body(x)  # (batch, 128)
        x = x + self.att1(x)  # residual-style
        embed = self.final(x)  # (batch, embed_dim)
        logits = self.head(embed).squeeze(-1)  # (batch,)
        return logits, embed

    def get_optimizer(self, lr=1e-3):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)


# -------------------- TorchFeatureExtractor: sklearn 风格的 transformer --------------------
class TorchFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, embed_dim=32, epochs=100, lr=1e-3, batch_size=256, weight_decay=1e-4, device=None,
                 verbose=True):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.verbose = verbose

        self.net = DeepFeatureNet(input_dim=input_dim, embed_dim=embed_dim, weight_decay=weight_decay).to(self.device)
        self.optimizer = self.net.get_optimizer(lr=self.lr)
        self.criterion = FocalLoss()
        self.early_stopper = EarlyStopping(patience=20, min_delta=1e-4, monitor='loss')
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        X: ndarray (n_samples, n_features)
        y: ndarray-like (n_samples,) with {0,1}
        """
        self.classes_ = np.unique(y)
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
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"[TorchFeatureExtractor] Epoch {epoch + 1}/{self.epochs}, loss = {avg_loss:.6f}")

            if self.early_stopper.step(avg_loss):
                if self.verbose:
                    print(f"[TorchFeatureExtractor] Early stopping at epoch {epoch + 1}, loss={avg_loss:.6f}")
                break

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("TorchFeatureExtractor 尚未 fit，请先 fit 后再 transform。")
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        embeddings = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                _, emb = self.net(xb)  # logits, embed
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    # 便捷函数：直接返回 logits 概率
    def predict_proba_from_X(self, X):
        if not self.is_fitted_:
            raise RuntimeError("TorchFeatureExtractor 尚未 fit。")
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        probs = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                logits, _ = self.net(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs, axis=0)
        return np.column_stack([1 - probs, probs])


# -------------------- 级联模型类（移到全局作用域）--------------------
class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, scaler, torch_extractor, lda, naive_bayes):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.scaler = scaler
        self.torch_extractor = torch_extractor
        self.lda = lda
        self.naive_bayes = naive_bayes

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_scaled = self.scaler.transform(X_sel)
        return X_scaled

    def predict(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.torch_extractor.transform(X_scaled)
        emb_lda = self.lda.transform(emb)
        return self.naive_bayes.predict(emb_lda).ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.torch_extractor.transform(X_scaled)
        emb_lda = self.lda.transform(emb)
        return self.naive_bayes.predict_proba(emb_lda)


# -------------------- 整体流程 --------------------
def main():
    # ---------- 读取数据 ----------
    data = pd.read_csv("../clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"])
    y = data["target"]

    # ---------- 填补 ----------
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # ---------- 去除常量特征 ----------
    std_devs = np.std(X_imputed, axis=0)
    non_constant_features = std_devs > 0
    X_non_constant = X_imputed[:, non_constant_features]

    # ---------- VarianceThreshold ----------
    selector = VarianceThreshold()
    X_selected = selector.fit_transform(X_non_constant)

    # ---------- 划分数据集 ----------
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=SEED, stratify=y)

    # ---------- ADASYN 过采样（只对训练集） ----------
    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")

    # ---------- 标准化（在 resampled 上 fit scaler） ----------
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Torch 特征提取器（自注意力 + DNN） ----------
    input_dim = X_resampled_scaled.shape[1]
    torch_extractor = TorchFeatureExtractor(input_dim=input_dim, embed_dim=32, epochs=200, lr=1e-3, batch_size=512,
                                            weight_decay=1e-4, verbose=True)
    print("Start training Torch feature extractor ...")
    torch_extractor.fit(X_resampled_scaled, y_resampled)
    print("Torch feature extractor training finished.")

    # ---------- 生成 embedding（训练集与测试集） ----------
    X_emb_train = torch_extractor.transform(X_resampled_scaled)
    X_emb_test = torch_extractor.transform(X_test_scaled)

    print(f"Embedding shapes: train {X_emb_train.shape}, test {X_emb_test.shape}")

    # ---------- LDA 降维 ----------
    # 二分类问题中，LDA 的 n_components 最大为 1
    lda = LDA(n_components=1, solver='svd')
    X_emb_train_lda = lda.fit_transform(X_emb_train, y_resampled)
    X_emb_test_lda = lda.transform(X_emb_test)

    # ---------- Adaboost 分类器 ----------
    adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=SEED)
    print("Training Adaboost on LDA-embedded features ...")
    adaboost.fit(X_emb_train_lda, y_resampled)
    print("Adaboost training finished.")

    # ---------- 预测 ----------
    y_proba = adaboost.predict_proba(X_emb_test_lda)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # ---------- 评估 ----------
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    # ---------- 保存模型与预处理器 ----------
    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        scaler=scaler,
        torch_extractor=torch_extractor,
        lda=lda,
        naive_bayes=adaboost  # 此处替换为 Adaboost 模型
    )

    joblib.dump(cascaded, "../best_cascaded_model.pkl")
    # joblib.dump(imputer, "imputer.pkl")
    # joblib.dump(selector, "variance_threshold_selector.pkl")
    # joblib.dump(non_constant_features, "non_constant_features.pkl")
    # joblib.dump(scaler, "scaler.pkl")
    # joblib.dump(adasyn, "adasyn_sampler.pkl")
    # joblib.dump(lda, "lda.pkl")
    # joblib.dump(adaboost, "naive_bayes.pkl")  # 保存为 naive_bayes.pkl，但内容为 Adaboost 模型

    print(
        "模型和预处理器已保存：best_cascaded_model.pkl, imputer.pkl, variance_threshold_selector.pkl, non_constant_features.pkl, "
        "scaler.pkl, adasyn_sampler.pkl, lda.pkl, naive_bayes.pkl")


if __name__ == "__main__":
    main()