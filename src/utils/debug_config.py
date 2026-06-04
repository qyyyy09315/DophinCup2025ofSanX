import joblib
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from deepforest import CascadeForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, x):
        weights = torch.softmax(self.attention_weights, dim=0)
        return x * weights


# 深度模型
class DeepModel(nn.Module):
    def __init__(self, input_dim, weight_decay=1e-4):
        super(DeepModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            Attention(16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.weight_decay = weight_decay  # 用于 L2 正则化

    def forward(self, x):
        return self.model(x).squeeze()

    def get_optimizer(self, lr=0.001):
        return optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=self.weight_decay  # 添加 L2 正则化
        )


# 早停机制
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0

    def __call__(self, current_loss):
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False


# 自定义 PyTorch 分类器
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=1000, lr=0.01, weight_decay=1e-4):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = DeepModel(input_dim, weight_decay=self.weight_decay).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = self.model.get_optimizer(lr=self.lr)
        self.early_stopping = EarlyStopping()
        self.classes_ = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.classes_ = np.unique(y)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                if self.device.type == 'cuda':
                    print(f'GPU memory allocated: {torch.cuda.memory_allocated(self.device)} bytes')

            if self.early_stopping(loss.item()):
                print(f'Early stop at epoch {epoch}')
                break

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            return (outputs > 0.5).float().cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.cat([1 - outputs.unsqueeze(1), outputs.unsqueeze(1)], dim=1)
            return probabilities.cpu().numpy()


# 深度森林分类器
class CustomCascadeForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.classes_ = None
        self.cascade_forest = CascadeForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.cascade_forest.fit(X, y)
        return self

    def predict(self, X):
        return self.cascade_forest.predict(X)

    def predict_proba(self, X):
        return self.cascade_forest.predict_proba(X)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.cascade_forest.fit(X, y)
        return self

    def predict(self, X):
        return self.cascade_forest.predict(X)

    def predict_proba(self, X):
        return self.cascade_forest.predict_proba(X)


def load_models():
    try:
        stacking_model = joblib.load('best_stacking_model.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('variance_threshold_selector.pkl')
        non_constant_features = joblib.load('non_constant_features.pkl')
        selected_feature_names = joblib.load('selected_feature_names.pkl')
        return stacking_model, imputer, scaler, selector, non_constant_features, selected_feature_names
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


stacking_model, imputer, scaler, selector, non_constant_features, selected_feature_names = load_models()

def preprocess_data(file_path, imputer, train_columns):
    try:
        # 读取新的测试数据
        test_data = pd.read_csv(file_path)

        # 去除 'company_id' 列，只保留特征数据
        X_test = test_data.drop(columns=['company_id'])

        # 将数值和非数值数据分开
        num_data = X_test.select_dtypes(include=[np.number])
        non_num_data = X_test.select_dtypes(exclude=[np.number])

        # 打印调试信息：数值数据列名
        num_data_columns = num_data.columns
        print(f"Num data columns in test set: {num_data_columns.tolist()}")

        # 删除多余的特征
        num_data = num_data[train_columns]

        # 打印调试信息：处理后的数值数据列名
        print(f"Processed num data columns: {num_data.columns.tolist()}")

        # 合并处理后的数据
        X_test = num_data.join(non_num_data)

        # 打印调试信息：最终特征列名
        print(f"Final X_test columns: {X_test.columns.tolist()}")

        return X_test

    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        raise


file_path = r'C:\Users\YKSHb\Desktop\2408DAback\data\test.csv'
X_test = preprocess_data(file_path, imputer, imputer.feature_names_in_)

# 确保特征列一致
test_columns = X_test.columns
train_columns = imputer.feature_names_in_  # 从训练中获得的特征列名

missing_features = set(train_columns) - set(test_columns)
extra_features = set(test_columns) - set(train_columns)

if missing_features:
    print(f"Missing features in test set: {missing_features}")

if extra_features:
    print(f"Extra features in test set: {extra_features}")

# 对于 num_data，确保其列与训练时一致
X_test = X_test[train_columns]

# 如果有缺失的特征，则填充为0
for feature in missing_features:
    X_test[feature] = 0

# 删除额外的特征
X_test = X_test[train_columns]


def make_predictions(X_test, model):
    try:
        # 最后一步再将数据转换为 numpy 数组
        X_test = X_test.values

        # 进行预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_proba
    except Exception as e:
        print(f"Error in making predictions: {e}")
        raise


y_pred, y_proba = make_predictions(X_test, stacking_model)


def save_results(test_data, y_pred, y_proba, output_file):
    try:
        results_df = pd.DataFrame({
            'uuid': test_data['company_id'],  # 使用未删除的原始数据中的 'company_id'
            'proba': y_proba,
            'prediction': y_pred,
        })

        # 保存结果到 CSV 文件
        results_df.to_csv(output_file, index=False)
        print(f"Predictions and probabilities have been saved to {output_file}")
    except Exception as e:
        print(f"Error in saving results: {e}")
        raise


output_file = 'submit_template.csv'
save_results(pd.read_csv(file_path), y_pred, y_proba, output_file)
