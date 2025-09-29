import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

# --- 设置环境变量以控制底层库的线程数 ---
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

warnings.filterwarnings("ignore")


class BalancedRandomForestCascade(BaseEstimator, ClassifierMixin):
    """
    决策树级联的平衡随机森林网络（改进版）
    每一层使用 BalancedRandomForestClassifier，并将预测概率作为下一层的输入特征。
    参数：
        n_estimators: 每层的树数量（可调整以防过拟合）
        max_layers: 最大层数
        random_state: 随机种子
        max_depth: 单棵树最大深度（控制复杂度）
        max_features: 每棵树随机特征数（例如 "sqrt"）
        n_jobs: BalancedRandomForestClassifier 的并行度
        debug: 是否打印每层输入维度（用于调试）
    """
    def __init__(self,
                 n_estimators=100,
                 max_layers=3,
                 random_state=None,
                 max_depth=None,
                 max_features="sqrt",
                 n_jobs=1,
                 debug=False):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.debug = debug

        self.layers_ = []
        self.classes_ = None
        self.initial_n_features_ = None
        self.layer_input_dims_ = []  # 记录每层训练时的输入维度

    def _make_layer_clf(self, layer_idx):
        rs = (self.random_state + layer_idx) if (self.random_state is not None) else layer_idx
        return BalancedRandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=rs,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=self.n_jobs
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_features=1)
        self.classes_ = np.unique(y)
        self.initial_n_features_ = X.shape[1]

        current_input = X.copy()

        for layer_idx in range(self.max_layers):
            # 记录训练时该层的输入维度（用于后续predict时一致性检查）
            self.layer_input_dims_.append(current_input.shape[1])

            layer_clf = self._make_layer_clf(layer_idx)
            layer_clf.fit(current_input, y)
            self.layers_.append(layer_clf)

            # 取得概率特征并拼接（用于下一层训练）
            if len(self.classes_) == 2:
                probas = layer_clf.predict_proba(current_input)[:, 1:].reshape(-1, 1)
            else:
                probas = layer_clf.predict_proba(current_input)

            current_input = np.hstack((current_input, probas))

            if self.debug:
                print(f"[fit] 层 {layer_idx} 训练时输入维度: {self.layer_input_dims_[-1]}, "
                      f"拼接后维度: {current_input.shape[1]}")

        return self

    def predict_proba(self, X):
        if not hasattr(self, "layers_") or len(self.layers_) == 0:
            raise RuntimeError("必须先调用 fit() 训练模型！")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"期望2D数组，得到{X.ndim}D数组")

        if X.shape[1] != self.initial_n_features_:
            raise ValueError(f"输入特征数 {X.shape[1]} 与训练时初始特征数 {self.initial_n_features_} 不匹配")

        current_input = X.copy()

        for layer_idx, layer_clf in enumerate(self.layers_):
            expected_dim = self.layer_input_dims_[layer_idx]
            if current_input.shape[1] != expected_dim:
                raise ValueError(
                    f"预测时第 {layer_idx} 层输入维度 {current_input.shape[1]} 与训练时的 {expected_dim} 不一致"
                )

            if len(self.classes_) == 2:
                probas = layer_clf.predict_proba(current_input)[:, 1:].reshape(-1, 1)
            else:
                probas = layer_clf.predict_proba(current_input)

            current_input = np.hstack((current_input, probas))

            if self.debug:
                print(f"[predict] 层 {layer_idx} 预测时输入维度: {expected_dim}, 拼接后维度: {current_input.shape[1]}")

        # 注意：最后一层的训练输入维度记录在 layer_input_dims_[-1]
        # layers_[-1] 是在其对应输入（即 current_input 切片到前 layer_input_dims_[-1]）上训练的
        last_input_dim = self.layer_input_dims_[-1]
        last_layer_input = current_input[:, :last_input_dim]
        return self.layers_[-1].predict_proba(last_layer_input)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_params(self, deep=True):
        # 使其在 sklearn 管道中可调
        return {
            "n_estimators": self.n_estimators,
            "max_layers": self.max_layers,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "debug": self.debug
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class CascadeForestWrapper(ClassifierMixin):
    """包装器用于让 StackingClassifier 能够识别该自定义分类器"""
    def __init__(self, cascade_forest):
        self.cascade_forest = cascade_forest
        self.classes_ = None

    def fit(self, X, y):
        self.cascade_forest.fit(X, y)
        self.classes_ = self.cascade_forest.classes_
        return self

    def predict(self, X):
        return self.cascade_forest.predict(X)

    def predict_proba(self, X):
        return self.cascade_forest.predict_proba(X)

    def get_params(self, deep=True):
        return {"cascade_forest": self.cascade_forest}

    def set_params(self, **params):
        if "cascade_forest" in params:
            self.cascade_forest = params["cascade_forest"]
        return self

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
