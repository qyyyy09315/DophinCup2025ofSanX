import os
import pickle
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

# --- 设置环境变量以控制底层库的线程数 ---
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


class BalancedRandomForestCascade(BaseEstimator, ClassifierMixin):
    """
    决策树级联的平衡随机森林网络
    每一层使用 BalancedRandomForestClassifier，并将预测概率作为下一层的输入特征。
    """

    def __init__(self,
                 n_estimators=100,
                 max_layers=5,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.random_state = random_state
        self.layers_ = []
        self.classes_ = None
        self.initial_n_features_ = None
        self.layer_input_dims_ = []  # 记录每层输入维度

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_features=1)
        self.classes_ = np.unique(y)
        self.initial_n_features_ = X.shape[1]

        current_input = X.copy()

        for layer_idx in range(self.max_layers):
            self.layer_input_dims_.append(current_input.shape[1])  # 记录输入维度

            layer_clf = BalancedRandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=(self.random_state + layer_idx) if self.random_state else layer_idx,
                n_jobs=2
            )
            layer_clf.fit(current_input, y)
            self.layers_.append(layer_clf)

            if len(self.classes_) == 2:
                probas = layer_clf.predict_proba(current_input)[:, 1:]
            else:
                probas = layer_clf.predict_proba(current_input)

            current_input = np.hstack((current_input, probas))

        return self

    def predict_proba(self, X):
        if not hasattr(self, "layers_") or len(self.layers_) == 0:
            raise RuntimeError("必须先调用fit训练模型！")

        X = np.asarray(X)
        if X.shape[1] != self.initial_n_features_:
            raise ValueError(f"输入特征数 {X.shape[1]} 与训练时初始特征数 {self.initial_n_features_} 不匹配")

        current_input = X.copy()

        for layer_idx, layer_clf in enumerate(self.layers_):
            expected_dim = self.layer_input_dims_[layer_idx]
            if current_input.shape[1] != expected_dim:
                raise ValueError(f"预测时第 {layer_idx} 层输入维度 {current_input.shape[1]} "
                                 f"与训练时的 {expected_dim} 不一致")

            if len(self.classes_) == 2:
                probas = layer_clf.predict_proba(current_input)[:, 1:]
            else:
                probas = layer_clf.predict_proba(current_input)

            current_input = np.hstack((current_input, probas))

        # 返回最后一层的预测概率
        return self.layers_[-1].predict_proba(current_input[:, :self.layer_input_dims_[-1]])

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class CascadeForestWrapper(ClassifierMixin):
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
        print(f"模型已保存至: {filepath}")


def load_object(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("=" * 50)
    print("开始数据预处理和模型训练...")
    print("=" * 50)

    data = pd.read_csv("./clean.csv").drop(columns=["company_id"])
    print(f"原始数据形状: {data.shape}")

    data = pd.get_dummies(data, drop_first=True)
    print(f"编码后数据形状: {data.shape}")

    X = data.drop(columns=["target"])
    y = data["target"]
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
    print(f"平衡后数据形状: {X_resampled.shape}")

    fixed_k = min(100, X_resampled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=fixed_k)
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    print(f"特征选择后数量: {X_selected.shape[1]}")

    save_object(selector, "./feature_selector_xgboost_cemmdan.pkl")
    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_resampled, test_size=0.1, random_state=42
    )

    base_balanced_cascade = BalancedRandomForestCascade(
        n_estimators=300,
        max_layers=3,
        random_state=42
    )
    balanced_cascade = CascadeForestWrapper(base_balanced_cascade)

    adaboost = AdaBoostClassifier(n_estimators=700, algorithm="SAMME", random_state=42)
    stacking_model = StackingClassifier(
        estimators=[
            ("balanced_cascade", balanced_cascade),
            ("adaboost", adaboost),
        ],
        final_estimator=GaussianNB(),
        n_jobs=2,
        cv=3
    )

    print("开始训练模型...")
    stacking_model.fit(X_train, y_train)
    print("模型训练完成！")

    save_object(stacking_model, "./model_pipeline_xgboost_cemmdan.pkl")

    y_pred_test = stacking_model.predict(X_test)
    y_proba_test = stacking_model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print("\n模型评估结果:")
    print(f"召回率 (Recall):    {recall:.6f}")
    print(f"AUC:               {auc:.6f}")
    print(f"精确率 (Precision): {precision:.6f}")
    print(f"最终评分:          {final_score:.6f}")
    print("=" * 50)
