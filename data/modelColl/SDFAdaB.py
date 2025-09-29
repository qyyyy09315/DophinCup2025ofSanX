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


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def load_object(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("开始数据预处理与训练流程（含过拟合控制）")
    print("=" * 60)

    # 1) 读取数据
    data_path = "./clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到 {data_path}，请确认路径正确。")

    data = pd.read_csv(data_path).drop(columns=["company_id"])
    print(f"原始数据形状: {data.shape}")

    # 2) 类别编码（one-hot）
    data = pd.get_dummies(data, drop_first=True)
    print(f"编码后数据形状: {data.shape}")

    # 3) 分离特征与标签
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列。")
    X = data.drop(columns=["target"])
    y = data["target"].values
    print(f"特征形状: {X.shape}, 标签分布: {np.bincount(y)}")

    # 4) 缺失值处理与缩放
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 保存预处理器
    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # 5) 处理类别不平衡（注意：过采样会引入噪声，需谨慎）
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
    print(f"SMOTEENN 后数据形状: {X_resampled.shape}, 标签分布: {np.bincount(y_resampled)}")

    # 6) 自动选择最优 k（候选 k 列表），用一个较稳健的弱分类器评估 AUC
    candidate_k = [30, 50, 80, 100]
    best_k = min(candidate_k)
    best_score = -np.inf

    # 在小数据集上可增大 cv，否则保持 cv=5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("开始选择最佳特征数 k（使用简单 BRF 做 cross-val 评估）...")
    for k in candidate_k:
        k_tmp = min(k, X_resampled.shape[1])
        selector_tmp = SelectKBest(score_func=f_classif, k=k_tmp)
        X_tmp = selector_tmp.fit_transform(X_resampled, y_resampled)
        # 用一个受限参数的 BRF 做评估，避免过拟合评估器本身
        eval_clf = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=2
        )
        try:
            scores = cross_val_score(eval_clf, X_tmp, y_resampled, cv=cv, scoring="roc_auc", n_jobs=2)
            mean_score = np.mean(scores)
        except Exception as e:
            mean_score = -np.inf
        print(f"  k={k_tmp} -> CV AUC = {mean_score:.5f}")
        if mean_score > best_score:
            best_score = mean_score
            best_k = k_tmp

    print(f"选择到的最佳 k = {best_k}（CV AUC={best_score:.5f}）")

    # 7) 用最佳 k 做最终特征选择
    selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    print(f"特征选择后维度: {X_selected.shape}")

    save_object(selector, "./feature_selector.pkl")

    # 8) 划分训练/测试集（最终评估使用 held-out test）
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_resampled, test_size=0.1, stratify=y_resampled, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 9) 初始化 Cascade（限制复杂度）
    base_balanced_cascade = BalancedRandomForestCascade(
        n_estimators=150,   # 减少树数
        max_layers=3,
        random_state=42,
        max_depth=12,       # 限制树深
        max_features="sqrt",
        n_jobs=2,
        debug=False         # 设置 True 可打印每层维度以便调试
    )
    balanced_cascade = CascadeForestWrapper(base_balanced_cascade)

    # 10) AdaBoost（降低学习率与迭代数）
    adaboost = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.5,
        algorithm="SAMME",
        random_state=42
    )

    # 11) Stacking（增加 cv 稳定性）
    stacking_model = StackingClassifier(
        estimators=[
            ("balanced_cascade", balanced_cascade),
            ("adaboost", adaboost),
        ],
        final_estimator=GaussianNB(),
        n_jobs=2,
        cv=5
    )

    # 12) 在训练集上先做一次交叉验证评估（观察是否还存在过拟合）
    print("在训练集上进行交叉验证评估（观察过拟合迹象）...")
    try:
        cv_scores = cross_val_score(clone(stacking_model), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=2)
        print(f"训练集 CV AUC (5-fold): mean={np.mean(cv_scores):.5f}, std={np.std(cv_scores):.5f}")
    except Exception as e:
        print("交叉验证评估出错：", e)

    # 13) 训练最终模型
    print("开始拟合最终模型（可能耗时）...")
    stacking_model.fit(X_train, y_train)
    print("训练完成。")

    # 保存模型
    save_object(stacking_model, "./model_pipeline.pkl")

    # 14) 在 held-out 测试集上评估
    y_pred_test = stacking_model.predict(X_test)
    # 如果 predict_proba 不可用（极端情况），处理异常
    try:
        y_proba_test = stacking_model.predict_proba(X_test)[:, 1]
    except Exception:
        # 退回使用预测标签的 0/1 （这会使 AUC 无法正确计算）
        y_proba_test = (y_pred_test == 1).astype(float)

    recall = recall_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print("\n最终模型在测试集上的评估结果：")
    print(f"召回率 (Recall):    {recall:.6f}")
    print(f"AUC:               {auc:.6f}")
    print(f"精确率 (Precision): {precision:.6f}")
    print(f"最终评分:          {final_score:.6f}")
    print("=" * 60)
