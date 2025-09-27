import multiprocessing
import os
import pickle
import random
import time
import warnings

# --- 导入绘图库 ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# --- 导入先进的特征工程库 ---
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, f_classif, \
    mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (recall_score, precision_score, roc_auc_score, f1_score, accuracy_score,
                             precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, MinMaxScaler)

# --- 导入聚类和异常检测用于特征工程 ---
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# --- 导入稀疏矩阵支持 ---
from scipy import sparse
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# 环境配置
NUM_CORES = multiprocessing.cpu_count()
print(f"检测到 {NUM_CORES} 个CPU核心")
os.environ["LOKY_MAX_CPU_COUNT"] = str(NUM_CORES)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 神经网络模型定义 ---
class SimpleNN(nn.Module):
    """一个简单的前馈神经网络"""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出一个logit
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiScaleNNEnsemble:
    """多尺度神经网络集群"""

    def __init__(self, base_model_class, model_params_list, device='cpu'):
        self.base_model_class = base_model_class
        self.model_params_list = model_params_list  # 每个模型的参数字典列表
        self.device = device
        self.models = []
        self.weights = []  # 存储每个模型的权重

    def fit(self, X_train_list, y_train, X_val_list, y_val, epochs=50, batch_size=1024, lr=0.001):
        """训练所有模型"""
        self.models = []
        self.weights = []
        # 确保 y_train 是 numpy 数组，并且形状正确
        if not isinstance(y_train, np.ndarray):
            y_train_np = np.array(y_train)
        else:
            y_train_np = y_train

        if not isinstance(y_val, np.ndarray):
            y_val_np = np.array(y_val)
        else:
            y_val_np = y_val

        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(self.device)

        for i, (X_train, X_val, model_params) in enumerate(zip(X_train_list, X_val_list, self.model_params_list)):
            print(f"训练模型 {i + 1}/{len(self.model_params_list)}...")
            input_dim = X_train.shape[1]
            model = self.base_model_class(input_dim, **model_params).to(self.device)

            # 数据加载器 - 确保 X_train 也是 numpy 数组
            if not isinstance(X_train, np.ndarray):
                X_train_np = np.array(X_train)
            else:
                X_train_np = X_train

            if not isinstance(X_val, np.ndarray):
                X_val_np = np.array(X_val)
            else:
                X_val_np = X_val

            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train_np, dtype=torch.float32), y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            for epoch in range(epochs):
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # 在验证集上评估以确定权重
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(
                    model(torch.tensor(X_val_np, dtype=torch.float32).to(self.device))).cpu().numpy().flatten()
            try:
                val_auc = roc_auc_score(y_val_np, val_preds)
            except:
                val_auc = 0.5  # 防止AUC计算错误
            print(f"  -> 模型 {i + 1} 验证集 AUC: {val_auc:.4f}")

            self.models.append(model)
            self.weights.append(val_auc)  # 使用AUC作为权重

    def predict_proba(self, X_test_list):
        """预测概率，返回正类概率"""
        if not self.models:
            raise ValueError("模型尚未训练，请先调用 fit 方法。")

        all_probs = []
        normalized_weights = np.array(self.weights) / np.sum(self.weights)  # 归一化权重

        for model, X_test, weight in zip(self.models, X_test_list, normalized_weights):
            model.eval()
            with torch.no_grad():
                # 确保 X_test 是 numpy 数组
                if not isinstance(X_test, np.ndarray):
                    X_test_np = np.array(X_test)
                else:
                    X_test_np = X_test
                test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(self.device)
                probs = torch.sigmoid(model(test_tensor)).cpu().numpy().flatten()
            all_probs.append(probs * weight)  # 加权概率

        # 对所有加权概率求和得到最终概率
        final_probs = np.sum(np.array(all_probs), axis=0)
        # 确保概率在 [0, 1] 范围内
        final_probs = np.clip(final_probs, 0, 1)
        return final_probs

    def predict(self, X_test_list, threshold=0.5):
        """预测类别"""
        probs = self.predict_proba(X_test_list)
        return (probs >= threshold).astype(int)


def moo_smotetomek_func(X, y):
    """
    使用一组固定的参数应用 SMOTETomek 进行重采样。
    """
    print("  -> 应用固定参数的 SMOTETomek...")

    if len(np.unique(y)) < 2:
        print("    -> 标签类别不足，跳过SMOTETomek.")
        return X, y

    try:
        if sparse.issparse(X):
            print("    -> 将稀疏矩阵转换为密集矩阵以进行 SMOTETomek...")
            X_dense = X.toarray()
        else:
            X_dense = X

        smotetomek = SMOTETomek(
            smote=SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=SEED, n_jobs=-1),
            tomek=TomekLinks(sampling_strategy='majority', n_jobs=-1),
            random_state=SEED
        )

        X_resampled_dense, y_resampled = smotetomek.fit_resample(X_dense, y)

        print(f"    -> 应用 SMOTETomek 后，样本数: {X_resampled_dense.shape[0]}")
        return X_resampled_dense, y_resampled

    except Exception as e:
        print(f"    -> 应用 SMOTETomek 失败: {e}。回退到原始数据。")
        return X, y


class AdvancedFeatureEngineer:
    """
    简化版的特征工程管道，仅包含：
    1. 多种预处理技术
    2. 智能特征选择 (方差过滤, 单变量选择, 基于模型的选择)
    """

    def __init__(self,
                 # 预处理参数
                 scaler_type='robust',  # 'standard', 'robust', 'quantile', 'power', 'minmax'
                 power_method='yeo-johnson',  # PowerTransformer method

                 # 特征选择参数
                 variance_threshold=0.01,
                 univariate_k_best=1000,
                 rf_n_features_to_select=800,

                 # 输出格式
                 force_sparse_output=False):

        # 存储参数
        self.scaler_type = scaler_type
        self.power_method = power_method
        self.variance_threshold = variance_threshold
        self.univariate_k_best = univariate_k_best
        self.rf_n_features_to_select = rf_n_features_to_select
        self.force_sparse_output = force_sparse_output

        # 初始化组件
        self.imputer_num = None
        self.scaler = None
        self.power_transformer = None
        self.variance_selector = None
        self.univariate_selector = None
        self.rf_selector = None

        # 记录特征信息
        self.numerical_features = None
        self.original_feature_count = None
        self.feature_names = []  # 特征名称列表

    def fit(self, X, y=None):
        """拟合特征工程管道"""
        print("开始简化版特征工程拟合 (仅预处理+特征选择)...")

        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X_dense = X.values
        elif sparse.issparse(X):
            X_dense = X.toarray()
            self.numerical_features = list(range(X.shape[1]))
        else:
            X_dense = X
            self.numerical_features = list(range(X.shape[1]))

        self.original_feature_count = X_dense.shape[1]
        print(f"-> 原始特征数: {self.original_feature_count}")

        # 1. 缺失值处理
        print("-> 步骤 1: 缺失值处理...")
        self.imputer_num = SimpleImputer(strategy='median')
        X_dense = self.imputer_num.fit_transform(X_dense)

        # 2. 数据预处理和变换
        print("-> 步骤 2: 数据预处理...")

        # 选择缩放器
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=SEED)
        elif self.scaler_type == 'power':
            self.power_transformer = PowerTransformer(method=self.power_method, standardize=True)
            X_dense = self.power_transformer.fit_transform(X_dense)
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()

        X_dense = self.scaler.fit_transform(X_dense)

        # 3. 智能特征选择
        print("-> 步骤 3: 智能特征选择...")

        # 方差过滤
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_dense = self.variance_selector.fit_transform(X_dense)
        print(f"  -> 方差过滤后特征数: {X_dense.shape[1]}")

        # 单变量特征选择
        if y is not None and self.univariate_k_best:
            k_best = min(self.univariate_k_best, X_dense.shape[1])
            self.univariate_selector = SelectKBest(score_func=f_classif, k=k_best)
            X_dense = self.univariate_selector.fit_transform(X_dense, y)
            print(f"  -> 单变量选择后特征数: {X_dense.shape[1]}")

        # 随机森林特征选择
        if y is not None and self.rf_n_features_to_select:
            n_rf_features = min(self.rf_n_features_to_select, X_dense.shape[1])
            temp_rf = BalancedRandomForestClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, random_state=SEED)
            self.rf_selector = SelectFromModel(
                temp_rf, max_features=n_rf_features, threshold=-np.inf)
            X_dense = self.rf_selector.fit_transform(X_dense, y)
            print(f"  -> 随机森林选择后特征数: {X_dense.shape[1]}")

        print("特征工程拟合完成！")
        return self

    def transform(self, X):
        """应用特征工程变换"""
        # 处理输入格式
        if isinstance(X, pd.DataFrame):
            X_dense = X.values
        elif sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 1. 缺失值处理
        X_dense = self.imputer_num.transform(X_dense)

        # 2. 数据预处理
        if self.power_transformer is not None:
            X_dense = self.power_transformer.transform(X_dense)
        X_dense = self.scaler.transform(X_dense)

        # 3. 特征选择
        X_dense = self.variance_selector.transform(X_dense)
        if self.univariate_selector is not None:
            X_dense = self.univariate_selector.transform(X_dense)
        if self.rf_selector is not None:
            X_dense = self.rf_selector.transform(X_dense)

        # 输出格式控制
        if self.force_sparse_output:
            return sparse.csr_matrix(X_dense)
        else:
            return X_dense

    def fit_transform(self, X, y=None):
        """组合fit和transform"""
        return self.fit(X, y).transform(X)


def save_model(model, filename):
    """保存模型到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存为 '{filename}'")

# 其他函数保持不变


def find_best_f1_threshold(y_true, y_proba):
    """在验证集上寻找最佳阈值以优化 F1 分数"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_f_score = f_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print(f"最佳 F1 阈值: {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f_score:.4f}")

    return best_threshold


def find_threshold_for_max_recall(y_true, y_proba, min_precision=0.4):
    """寻找能获得最高召回率且精确率不低于 min_precision 的阈值"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid_indices) == 0:
        print(f"警告: 没有阈值能满足精确率 >= {min_precision}。返回默认阈值 0.5。")
        return 0.5

    best_valid_idx = valid_indices[np.argmax(recalls[valid_indices])]
    best_threshold = thresholds[best_valid_idx]
    best_precision = precisions[best_valid_idx]
    best_recall = recalls[best_valid_idx]

    print(f"高召回率阈值 (Precision >= {min_precision}): {best_threshold:.4f}")
    print(f"  对应 Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

    return best_threshold


def evaluate_and_plot(y_true, y_proba, threshold_f1, threshold_hr, model_name="Model"):
    """评估模型并绘制 ROC, PR, Confusion Matrix"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='b', lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")

    # Confusion Matrix
    y_pred_f1 = (y_proba >= threshold_f1).astype(int)
    cm = confusion_matrix(y_true, y_pred_f1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title(f'Confusion Matrix (Threshold={threshold_f1:.4f})')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    print(f"\n=== {model_name} 详细评估报告 ===")
    print(classification_report(y_true, y_pred_f1, target_names=['Negative', 'Positive']))
    print("-" * 40)


def train_and_evaluate_advanced(X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer_params=None):
    """使用简化特征工程的训练评估流程，模型替换为多尺度神经网络集群"""
    print("\n=== 构建简化特征工程管道 (多尺度神经网络集群) ===")

    # 特征工程 - 只创建一个基础配置的特征工程器
    print("\n=== 创建简化特征工程器 ===")
    fe = AdvancedFeatureEngineer(**feature_engineer_params)
    fe.fit(X_train, y_train)

    X_train_engineered = fe.transform(X_train)
    X_val_engineered = fe.transform(X_val)
    X_test_engineered = fe.transform(X_test)
    print(f"特征工程器输出特征数: {X_train_engineered.shape[1]}")

    # 2. 应用 SMOTETomek
    print("\n=== 应用 SMOTETomek 重采样 ===")
    X_train_resampled, y_train_resampled = moo_smotetomek_func(X_train_engineered, y_train)

    # 更新训练集
    X_train_engineered_final = X_train_resampled

    # 3. 定义神经网络模型集群 (保持不变)
    print("\n=== 定义神经网络模型集群 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ensemble_model_params_list = [
        {'hidden_dims': [256, 128, 64], 'dropout_rate': 0.3},
        {'hidden_dims': [128, 64], 'dropout_rate': 0.2},
        {'hidden_dims': [512, 256, 128, 64], 'dropout_rate': 0.4},
        {'hidden_dims': [64, 32], 'dropout_rate': 0.1},
    ]

    classifier = MultiScaleNNEnsemble(SimpleNN, ensemble_model_params_list, device=device)

    # 4. 模型训练 (保持不变)
    print("\n=== 开始模型集群训练 ===")
    start_time = time.time()
    classifier.fit([X_train_engineered_final]*len(ensemble_model_params_list), y_train_resampled,
                   [X_val_engineered]*len(ensemble_model_params_list), y_val, epochs=50, batch_size=1024,
                   lr=0.001)
    end_time = time.time()
    print(f"模型集群训练耗时: {end_time - start_time:.2f} 秒")

    # 5. 保存模型 (保存特征工程器和分类器)
    trained_pipeline = {
        'feature_engineer': fe,
        'classifier': classifier
    }
    save_model(trained_pipeline, 'simplified_feature_engineering_nn_ensemble_model.pkl')

    # 6. 验证集阈值选择 (保持不变)
    print("\n=== 验证集阈值选择 ===")
    val_proba = classifier.predict_proba([X_val_engineered]*len(ensemble_model_params_list))

    threshold_f1 = find_best_f1_threshold(y_val, val_proba)
    threshold_high_recall = find_threshold_for_max_recall(y_val, val_proba, min_precision=0.4)

    # 7. 测试集评估 (保持不变)
    print("\n=== 测试集评估 ===")
    test_proba = classifier.predict_proba([X_test_engineered]*len(ensemble_model_params_list))

    # F1优化阈值评估
    print("\n--- 使用 F1 优化阈值评估 ---")
    y_pred_f1 = (test_proba >= threshold_f1).astype(int)
    recall_f1 = recall_score(y_test, y_pred_f1)
    auc_f1 = roc_auc_score(y_test, test_proba)
    precision_f1 = precision_score(y_test, y_pred_f1)
    f1_f1 = f1_score(y_test, y_pred_f1)
    accuracy_f1 = accuracy_score(y_test, y_pred_f1)
    print(f"阈值: {threshold_f1:.4f}")
    print(f"Recall: {recall_f1:.4f}")
    print(f"AUC: {auc_f1:.4f}")
    print(f"Precision: {precision_f1:.4f}")
    print(f"F1 Score: {f1_f1:.4f}")
    print(f"Accuracy: {accuracy_f1:.4f}")
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision_f1 + 50 * auc_f1 + 30 * recall_f1:.4f}")

    # 高召回率阈值评估
    print("\n--- 使用高召回率阈值评估 ---")
    y_pred_hr = (test_proba >= threshold_high_recall).astype(int)
    recall_hr = recall_score(y_test, y_pred_hr)
    auc_hr = roc_auc_score(y_test, test_proba)
    precision_hr = precision_score(y_test, y_pred_hr)
    f1_hr = f1_score(y_test, y_pred_hr)
    accuracy_hr = accuracy_score(y_test, y_pred_hr)
    print(f"阈值: {threshold_high_recall:.4f}")
    print(f"Recall: {recall_hr:.4f}")
    print(f"AUC: {auc_hr:.4f}")
    print(f"Precision: {precision_hr:.4f}")
    print(f"F1 Score: {f1_hr:.4f}")
    print(f"Accuracy: {accuracy_hr:.4f}")
    print(f"TestScore (20*P+50*AUC+30*R): {20 * precision_hr + 50 * auc_hr + 30 * recall_hr:.4f}")

    # 可视化评估结果 (保持不变)
    print("\n=== 绘制评估图表 ===")
    evaluate_and_plot(y_test, test_proba, threshold_f1, threshold_high_recall,
                      model_name="Simplified Feature Engineering NN Ensemble")

    return {
        'recall': recall_f1,
        'auc': auc_f1,
        'precision': precision_f1,
        'f1': f1_f1,
        'accuracy': accuracy_f1,
        'y_proba': test_proba,
        'y_pred': y_pred_f1,
        'threshold': threshold_f1,
        'high_recall_results': {
            'recall': recall_hr,
            'precision': precision_hr,
            'f1': f1_hr,
            'threshold': threshold_high_recall
        },
        'trained_model': trained_pipeline  # 保存的是包含特征工程器和分类器的字典
    }



# 以上填写对象
import pandas as pd
import pickle

import pickle
import pandas as pd
import numpy as np

def load_model(filename):
    """从文件加载模型"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"模型已从 '{filename}' 加载")
    return model

# 1. 加载测试数据
# 请确保 'testClean.csv' 文件与脚本在同一目录下，或提供完整路径
test_data_path = 'testClean.csv'
print(f"正在加载测试数据: {test_data_path}")
test_data = pd.read_csv(test_data_path)
print(f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1]}")

# 2. 加载完整的级联模型 (实际上是包含特征工程和分类器的管道)
# 请确保 'simplified_feature_engineering_nn_ensemble_model.pkl' 文件与脚本在同一目录下，或提供完整路径
model_path = 'simplified_feature_engineering_nn_ensemble_model.pkl'
print(f"正在加载模型管道: {model_path}")
# 加载的是一个字典，包含 'feature_engineer' 和 'classifier'
loaded_pipeline = load_model(model_path)
loaded_fe = loaded_pipeline['feature_engineer']
loaded_classifier = loaded_pipeline['classifier'] # 这是 MultiScaleNNEnsemble 实例
print("模型管道加载成功。")

# 3. 准备测试特征并应用模型预测
# 确保 X_test 不包含 'company_id' 列
feature_columns = [col for col in test_data.columns if col != 'company_id']
X_test_raw = test_data[feature_columns]
print(f"用于预测的原始特征数: {X_test_raw.shape[1]}")

# 应用训练好的特征工程
print("正在进行特征工程变换...")
X_test_engineered = loaded_fe.transform(X_test_raw) # 应用训练好的特征工程
print(f"特征工程变换完成，变换后特征数: {X_test_engineered.shape[1]}")

print("正在进行预测...")
# 获取预测概率
# 注意：MultiScaleNNEnsemble 的 predict_proba 需要输入一个列表，
# 其中每个元素都是对应模型的特征矩阵。在这里，我们对所有模型使用相同的工程化特征。
y_proba = loaded_classifier.predict_proba([X_test_engineered] * len(loaded_classifier.model_params_list))
print(f"预测完成，获得 {len(y_proba)} 个概率值。")

# 应用固定阈值进行分类
threshold = 0.7436  # 这里明确定义分类阈值
print(f"应用分类阈值: {threshold}")
y_pred = (y_proba >= threshold).astype(int)
print(f"分类完成。")

# 4. 创建结果数据框
# 假设原始测试数据中有 'company_id' 列用于标识
if 'company_id' in test_data.columns:
    uuid_column = test_data['company_id']
else:
    # 如果没有 'company_id'，可以使用索引或其他方式生成唯一标识
    print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
    uuid_column = test_data.index

results_df = pd.DataFrame({
    'uuid': uuid_column,
    'proba': y_proba,
    'prediction': y_pred
})
print("结果数据框创建成功。")

# 5. 保存结果
# 请根据您的实际需求修改保存路径
output_path = r'C:\Users\YKSHb\Desktop\submit_template.csv' # Windows 路径示例
print(f"正在保存结果到: {output_path}")
results_df.to_csv(output_path, index=False)
print(f"预测完成，使用阈值 {threshold} 进行分类，结果已保存到 {output_path}")