import os
import random
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.validation import validate_data  # 用于替代 _validate_data

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

# ------------------- Focal Loss -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
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

# ------------------- DeepFeatureNet -------------------
class DeepFeatureNet(nn.Module):
    def __init__(self, input_dim, embed_dim=32, weight_decay=1e-4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.att1 = SelfAttention(128)

        self.final = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(embed_dim, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.body(x)
        x = x + self.att1(x)
        embed = self.final(x)
        logits = self.head(embed).squeeze(-1)
        return logits, embed

    def get_optimizer(self, lr=1e-3):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

# ------------------- TorchFeatureExtractor -------------------
class TorchFeatureExtractor:
    def __init__(self, input_dim, embed_dim=32, epochs=100, lr=1e-3, batch_size=512, weight_decay=1e-4, device=None, verbose=True):
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

        # 验证设备是否为 GPU
        if self.device.type == 'cuda':
            print(f"[TorchFeatureExtractor] 模型已加载到 GPU {self.device}")
        else:
            print("[TorchFeatureExtractor] 模型将使用 CPU")

        if self.verbose:
            print(f"[TorchFeatureExtractor] 模型参数设备: {next(self.net.parameters()).device}")

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

                if self.verbose and epoch == 0 and count == 0:
                    print(f"[TorchFeatureExtractor] 数据已移动到设备: xb.device={xb.device}, yb.device={yb.device}")

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
                _, emb = self.net(xb)
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

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

# ------------------- EarlyStopping -------------------
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

# ------------------- 级联模型类 -------------------
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

# ------------------- 目标函数用于贝叶斯优化 -------------------
def objective(trial):
    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)

    # 填补缺失值
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    # 删除常量特征
    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]
    X_val_non_constant = X_val_imp[:, non_constant_features]

    # VarianceThreshold
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)
    X_val_selected = selector.transform(X_val_non_constant)

    # ADASYN 过采样
    adasyn_neighbors = trial.suggest_int("adasyn_neighbors", 2, 10)
    adasyn = ADASYN(random_state=SEED, n_neighbors=adasyn_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_selected, y_train)

    # 标准化
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_val_scaled = scaler.transform(X_val_selected)

    # TorchFeatureExtractor 参数
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 50, 200)

    torch_extractor = TorchFeatureExtractor(
        input_dim=X_resampled_scaled.shape[1],
        embed_dim=embed_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=512,
        verbose=False
    )
    torch_extractor.fit(X_resampled_scaled, y_resampled)

    # 生成 embedding
    X_emb_train = torch_extractor.transform(X_resampled_scaled)
    X_emb_val = torch_extractor.transform(X_val_scaled)

    # LDA 降维
    lda = LDA(n_components=1, solver='svd')
    X_emb_train_lda = lda.fit_transform(X_emb_train, y_resampled)
    X_emb_val_lda = lda.transform(X_emb_val)

    # AdaBoost 分类器
    ada_n_estimators = trial.suggest_int("ada_n_estimators", 50, 200)
    ada_lr = trial.suggest_float("ada_lr", 0.01, 1.0)
    adaboost = AdaBoostClassifier(
        n_estimators=ada_n_estimators,
        learning_rate=ada_lr,
        random_state=SEED
    )
    adaboost.fit(X_emb_train_lda, y_resampled)

    # 评估 AUC
    y_proba = adaboost.predict_proba(X_emb_val_lda)[:, 1]
    auc = roc_auc_score(y_val, y_proba)

    return auc

# ------------------- 主流程 -------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
    else:
        print("CUDA 不可用，将使用 CPU 训练。")

    # 启动 Optuna 优化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # 使用最佳参数重新训练模型
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
    adasyn_neighbors = best_params["adasyn_neighbors"]
    adasyn = ADASYN(random_state=SEED, n_neighbors=adasyn_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_selected, y_train_all)

    # 标准化
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # TorchFeatureExtractor
    input_dim = X_resampled_scaled.shape[1]
    torch_extractor = TorchFeatureExtractor(
        input_dim=input_dim,
        embed_dim=best_params["embed_dim"],
        epochs=best_params["epochs"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        batch_size=512,
        verbose=True
    )
    torch_extractor.fit(X_resampled_scaled, y_resampled)

    # 生成 embedding
    X_emb_train = torch_extractor.transform(X_resampled_scaled)

    # LDA 降维
    lda = LDA(n_components=1, solver='svd')
    X_emb_train_lda = lda.fit_transform(X_emb_train, y_resampled)

    # AdaBoost 分类器
    adaboost = AdaBoostClassifier(
        n_estimators=best_params["ada_n_estimators"],
        learning_rate=best_params["ada_lr"],
        random_state=SEED
    )
    adaboost.fit(X_emb_train_lda, y_resampled)

    # 预测与评估
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_scaled = scaler.transform(X_test_selected)
    X_emb_test = torch_extractor.transform(X_test_scaled)
    X_emb_test_lda = lda.transform(X_emb_test)

    y_proba = adaboost.predict_proba(X_emb_test_lda)[:, 1]
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
        torch_extractor=torch_extractor,
        lda=lda,
        naive_bayes=adaboost
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(lda, "lda.pkl")
    joblib.dump(adaboost, "naive_bayes.pkl")

    print("模型和预处理器已保存。")

    # 最终模型是否在 GPU 上
    if torch.cuda.is_available():
        print("[Final Model] 模型是否在 GPU 上:", next(torch_extractor.net.parameters()).is_cuda)
    else:
        print("[Final Model] 当前不支持 GPU")

if __name__ == "__main__":
    main()