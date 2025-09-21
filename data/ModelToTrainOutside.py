import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.cuda.amp as amp  # 混合精度
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # 根据 CPU 核心数调整
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False  # 为了性能关闭 determinism
    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
torch.set_num_threads(16)


# ------------------- 自适应激活函数 -------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ------------------- 残差块 -------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1, activation='swish'):
        super().__init__()
        self.activation_fn = Swish() if activation == 'swish' else Mish()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            self.activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate)
        )
        self.skip_connection = nn.Linear(dim, dim) if dim != dim else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.block(x)
        return self.activation_fn(out + residual)  # 残差连接


# ------------------- 增强型多头自注意力模块 -------------------
class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        # 添加相对位置编码（虽然对于单点特征不是必须，但可以增强表达能力）
        self.relative_position_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # 添加层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape  # B: batch, N: seq_len=1, C: dim
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_position_bias  # 添加相对位置偏置
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)

        # 残差连接
        x = x + F.dropout(x_attn, p=0.1, training=self.training)

        # FFN + 残差连接
        x = x + self.ffn(self.norm2(x))

        return x.squeeze(1)  # 回到 (B, C)


# ------------------- 特征交互模块 -------------------
class FeatureInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.bilinear = nn.Bilinear(input_dim, input_dim, hidden_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.combine = nn.Linear(hidden_dim * 2, input_dim)
        self.activation = Swish()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 双线性交互
        bilinear_out = self.bilinear(x, x)
        # 线性变换
        linear_out = self.linear(x)
        # 合并
        combined = torch.cat([bilinear_out, linear_out], dim=-1)
        out = self.activation(self.combine(combined))
        return self.dropout(out)


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
        pt = torch.exp(-BCE_loss.clamp(max=50))
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# ------------------- 增强版特征提取网络 -------------------
class EnhancedFeatureNet(nn.Module):
    def __init__(self, input_dim, weight_decay=1e-4, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = input_dim * 2  # 增加隐藏层维度

        # 输入层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Dropout(dropout_rate)
        )

        # 注意力模块
        self.attention = EnhancedMultiHeadAttention(hidden_dim, num_heads=8, dropout=dropout_rate)

        # 特征交互模块
        self.feature_interaction = FeatureInteraction(hidden_dim)

        # 深度残差网络
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate=dropout_rate, activation='swish'),
            ResidualBlock(hidden_dim, dropout_rate=dropout_rate, activation='swish'),
            ResidualBlock(hidden_dim, dropout_rate=dropout_rate, activation='swish')
        ])

        # 输出层
        self.classifier = nn.Linear(hidden_dim, 1)
        self.weight_decay = weight_decay

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)

        # 注意力机制
        x_attn = self.attention(x.unsqueeze(1))  # (B, C) -> (B, 1, C) for attention

        # 特征交互
        x_inter = self.feature_interaction(x_attn)

        # 深度残差网络
        x_deep = x_inter
        for block in self.res_blocks:
            x_deep = block(x_deep)

        # 分类器
        logits = self.classifier(x_deep).squeeze(-1)
        return logits, x_deep


# ------------------- 类间类内距离特征选择器 -------------------
class InterIntraDistanceSelector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y):
        """
        基于类间类内距离进行特征选择。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("目前只支持二分类任务")

        class_0_indices = (y == classes[0])
        class_1_indices = (y == classes[1])
        X_0 = X[class_0_indices]
        X_1 = X[class_1_indices]

        # 计算每个特征的类内方差之和
        intra_class_var_0 = np.var(X_0, axis=0)
        intra_class_var_1 = np.var(X_1, axis=0)
        intra_class_variance_sum = intra_class_var_0 + intra_class_var_1

        # 计算每个特征的类间距离平方
        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        inter_class_distance_sq = (mean_1 - mean_0) ** 2

        # 计算判别分数 (类间距离 / 类内方差和)
        # 避免除以零
        scores = np.divide(inter_class_distance_sq, intra_class_variance_sum,
                           out=np.zeros_like(inter_class_distance_sq),
                           where=intra_class_variance_sum != 0)

        # 选择分数高于阈值的特征
        self.selected_features_ = scores > self.threshold
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# ------------------- 改进的K-means聚类器 -------------------
class ImprovedKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """
        使用改进的策略拟合 K-means。
        """
        X = np.asarray(X)
        np.random.seed(self.random_state)

        # 1. 初始化聚类中心 (使用标准 K-means++ 初始化)
        n_samples, n_features = X.shape
        centers = []
        # 随机选择第一个中心
        centers.append(X[np.random.randint(n_samples)])

        for _ in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])
            # 选择下一个中心的概率与距离平方成正比
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers.append(X[j])
                    break
        centers = np.array(centers)

        # 迭代优化
        for iteration in range(self.max_iter):
            # 分配每个点到最近的中心
            distances = cdist(X, centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # 计算中心变化
            center_shift = np.linalg.norm(new_centers - centers)

            centers = new_centers

            if center_shift < self.tol:
                print(f"K-means 收敛于第 {iteration + 1} 次迭代")
                break
        else:
            print(f"K-means 达到最大迭代次数 {self.max_iter}")

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        """
        预测样本所属的簇。
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans has not been fitted yet.")
        X = np.asarray(X)
        distances = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


# ------------------- 特征提取器（支持混合精度训练）-------------------
class FeatureExtractor:
    def __init__(self, input_dim, epochs=2000, lr=1e-3, batch_size=4096,  # <<< 增大 batch_size
                 weight_decay=1e-4, device=None, verbose=True):  # ✅ 修复：新增 verbose 参数，默认 True
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.verbose = verbose  # ✅ 修复：初始化 self.verbose

        self.net = EnhancedFeatureNet(input_dim=input_dim, weight_decay=weight_decay).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                           betas=(0.9, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20)
        self.criterion = FocalLoss()
        self.early_stopper = EarlyStopping(patience=50, min_delta=1e-4, monitor='loss')
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False
        self.scaler = amp.GradScaler()  # 混合精度梯度缩放器

        if self.device.type == 'cuda':
            print(f"[FeatureExtractor] 模型已加载到 GPU {self.device}")
            print(f"[FeatureExtractor] 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        else:
            print("[FeatureExtractor] 模型将使用 CPU")

        if self.verbose:
            print(f"[FeatureExtractor] 模型参数设备: {next(self.net.parameters()).device}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False, num_workers=8, pin_memory=True)  # <<< 多进程 + pin_memory

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)  # 异步传输
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # 混合精度训练
                with amp.autocast():
                    logits, _ = self.net(xb)
                    loss = self.criterion(logits, yb)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)  # 为梯度裁剪做准备
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)  # 防止梯度爆炸
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item() * xb.size(0)
                count += xb.size(0)

            avg_loss = epoch_loss / max(1, count)

            # 使用 ReduceLROnPlateau 调度器
            self.scheduler.step(avg_loss)

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"[FeatureExtractor] Epoch {epoch + 1}/{self.epochs}, loss = {avg_loss:.6f}, lr = {current_lr:.2e}")

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
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)  # <<< 加速推理

        embeddings = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device, non_blocking=True)
                with amp.autocast():  # 推理也用混合精度
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
        self.stopped_epoch = 0

    def step(self, value):
        stop = False
        improved = False

        if self.monitor == 'loss':
            if (self.best - value) > self.min_delta:
                self.best = value
                self.wait = 0
                improved = True
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
                    self.stopped_epoch = self.wait
        else:
            if (value - self.best) > self.min_delta:
                self.best = value
                self.wait = 0
                improved = True
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop = True
                    self.stopped_epoch = self.wait

        return stop


# ------------------- 级联模型类 -------------------
class CascadedModel:
    def __init__(self, imputer, non_constant_features, selector, inter_intra_selector, kmeans, scaler,
                 feature_extractor, emb_imputer, classifiers):
        self.imputer = imputer
        self.non_constant_features = non_constant_features
        self.selector = selector
        self.inter_intra_selector = inter_intra_selector  # <<< 新增：类间类内距离选择器
        self.kmeans = kmeans  # <<< 新增：K-means 聚类器
        self.scaler = scaler
        self.feature_extractor = feature_extractor
        self.emb_imputer = emb_imputer  # <<< 新增：嵌入层插补器
        self.classifiers = classifiers

    def _preprocess(self, X_raw):
        X_arr = np.asarray(X_raw)
        X_imp = self.imputer.transform(X_arr)
        X_masked = X_imp[:, self.non_constant_features]
        X_sel = self.selector.transform(X_masked)
        X_inter_intra = self.inter_intra_selector.transform(X_sel)  # <<< 新增：应用类间类内距离选择
        # 注意：K-means聚类主要用于识别离群点，这里我们不直接用它变换数据，
        # 而是假设它在训练时帮助识别了离群点并进行了处理（例如过滤或加权）。
        # 如果需要在预测时也应用聚类，逻辑会更复杂。
        X_scaled = self.scaler.transform(X_inter_intra)
        return X_scaled

    def predict(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.feature_extractor.transform(X_scaled)
        emb_clean = self.emb_imputer.transform(emb)  # <<< 插补 NaN
        preds = [clf.predict(emb_clean) for clf in self.classifiers]
        final_pred = np.round(np.mean(preds, axis=0)).astype(int)
        return final_pred.ravel()

    def predict_proba(self, X_raw):
        X_scaled = self._preprocess(X_raw)
        emb = self.feature_extractor.transform(X_scaled)
        emb_clean = self.emb_imputer.transform(emb)  # <<< 插补 NaN
        probas = [clf.predict_proba(emb_clean)[:, 1] for clf in self.classifiers]
        final_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - final_proba, final_proba])


# ------------------- 主流程 -------------------
def main():
    print("CUDA 可用性:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 CUDA 设备:", torch.cuda.current_device())
        print("CUDA 设备名称:", torch.cuda.get_device_name(0))
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"显存已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
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

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)

    std_devs = np.std(X_train_imp, axis=0)
    non_constant_features = std_devs > 0
    X_train_non_constant = X_train_imp[:, non_constant_features]

    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_non_constant)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 新增：基于类间类内距离的特征选择
    inter_intra_selector = InterIntraDistanceSelector(threshold=0.5)  # 可调整阈值
    X_train_inter_intra_selected = inter_intra_selector.fit_transform(X_train_selected, y_train_all)
    print(
        f"[Feature Selection] 原始特征数: {X_train_selected.shape[1]}, 选择后特征数: {X_train_inter_intra_selected.shape[1]}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 新增：使用改进的 K-means 识别离群点 (在训练集上)
    # 这里我们简单地拟合K-means并打印中心，实际应用中可以用来过滤或加权样本
    kmeans = ImprovedKMeans(n_clusters=2, random_state=SEED)
    # 假设我们先用原始数据拟合K-means来识别离群点
    # 或者用选择后的特征拟合
    cluster_labels = kmeans.fit_predict(X_train_inter_intra_selected)
    print(f"[KMeans] 聚类中心:\n{kmeans.cluster_centers_}")
    # 这里可以添加逻辑来处理离群点，例如：
    # 1. 找到每个簇的马氏距离
    # 2. 标记距离簇中心过远的点为离群点
    # 3. 在后续训练中过滤或降低这些点的权重
    # 为简化，我们这里只做拟合，不改变训练数据
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    adasyn = ADASYN(random_state=SEED, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_inter_intra_selected, y_train_all)

    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    input_dim = X_resampled_scaled.shape[1]
    feature_extractor = FeatureExtractor(
        input_dim=input_dim,
        batch_size=4096,  # <<< 关键：增大 batch_size 吃满显存
        lr=5e-4,  # 降低学习率以获得更稳定的训练
        weight_decay=1e-5,  # 适当降低权重衰减
        verbose=True  # ✅ 修复：显式传入 verbose
    )
    feature_extractor.fit(X_resampled_scaled, y_resampled)

    X_emb_train = feature_extractor.transform(X_resampled_scaled)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ✅ 新增：训练嵌入层插补器（在训练集嵌入上拟合）
    emb_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    X_emb_train_clean = emb_imputer.fit_transform(X_emb_train)  # 拟合并转换
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    adaboost = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=SEED
    )
    adaboost.fit(X_emb_train_clean, y_resampled)  # <<< 使用清洗后的嵌入训练

    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgboost.fit(X_emb_train_clean, y_resampled)  # <<< 使用清洗后的嵌入训练

    # 测试阶段预处理
    X_test_imp = imputer.transform(X_test)
    X_test_non_constant = X_test_imp[:, non_constant_features]
    X_test_selected = selector.transform(X_test_non_constant)
    X_test_inter_intra_selected = inter_intra_selector.transform(X_test_selected)  # <<< 新增：测试集特征选择
    # 注意：测试集不应用 K-means 聚类处理
    X_test_scaled = scaler.transform(X_test_inter_intra_selected)
    X_emb_test = feature_extractor.transform(X_test_scaled)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ✅ 新增：对测试嵌入进行插补
    X_emb_test_clean = emb_imputer.transform(X_emb_test)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    y_proba_adaboost = adaboost.predict_proba(X_emb_test_clean)[:, 1]  # <<< 使用清洗后的嵌入预测
    y_proba_xgboost = xgboost.predict_proba(X_emb_test_clean)[:, 1]  # <<< 使用清洗后的嵌入预测
    y_proba = (y_proba_adaboost + y_proba_xgboost) / 2
    y_pred = (y_proba >= 0.5).astype(int)

    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print(f"Recall: {recall:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.4f}")

    cascaded = CascadedModel(
        imputer=imputer,
        non_constant_features=non_constant_features,
        selector=selector,
        inter_intra_selector=inter_intra_selector,  # <<< 保存选择器
        kmeans=kmeans,  # <<< 保存聚类器
        scaler=scaler,
        feature_extractor=feature_extractor,
        emb_imputer=emb_imputer,  # <<< 保存插补器
        classifiers=[adaboost, xgboost]
    )

    joblib.dump(cascaded, "best_cascaded_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(selector, "variance_threshold_selector.pkl")
    joblib.dump(inter_intra_selector, "inter_intra_distance_selector.pkl")  # <<< 保存选择器
    joblib.dump(kmeans, "kmeans_model.pkl")  # <<< 保存聚类器
    joblib.dump(non_constant_features, "non_constant_features.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(adasyn, "adasyn_sampler.pkl")
    joblib.dump(adaboost, "adaboost.pkl")
    joblib.dump(xgboost, "xgboost.pkl")
    joblib.dump(emb_imputer, "emb_imputer.pkl")  # <<< 保存插补器

    print("模型和预处理器已保存。")

    if torch.cuda.is_available():
        print("[Final Model] 模型是否在 GPU 上:", next(feature_extractor.net.parameters()).is_cuda)
        print(f"[Final Model] 显存已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print(f"[Final Model] 显存峰值: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
    else:
        print("[Final Model] 当前不支持 GPU")


if __name__ == "__main__":
    main()



