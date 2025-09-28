import os
import pickle
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

# --- 设置环境变量以控制底层库的线程数 ---
# 针对 8 核 CPU，设置为 6 是一个相对安全且能利用多核优势的选择
# 避免设置过高导致系统资源冲突
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# --- 修改后的深度森林：不生成新特征 ---
class EnhancedDeepForest(BaseEstimator, ClassifierMixin):
    """
    增强版深度森林（不生成新特征，每层独立使用原始输入）
    """

    def __init__(self,
                 n_estimators=100,
                 max_layers=5,
                 min_delta=0.001,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.min_delta = min_delta
        self.random_state = random_state
        self.layers_ = []  # 存储每层的分类器列表
        self.classes_ = None
        self.n_features_in_ = None

    def _create_base_estimators(self):
        # 显式设置 n_jobs=2，避免使用全部核心
        # 在8核CPU上，两个这样的基学习器并行运行比较安全
        return [
            RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=2),
            ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=2)
        ]

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_features=1)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # 所有层都使用原始 X，不生成新特征
        base_X = X.copy()

        for layer_idx in range(self.max_layers):
            layer_estimators = []
            for clf in self._create_base_estimators():
                clf.fit(base_X, y)
                layer_estimators.append(clf)
            self.layers_.append(layer_estimators)

        return self

    def predict_proba(self, X):
        if not hasattr(self, 'layers_') or len(self.layers_) == 0:
            raise RuntimeError("必须先调用fit训练模型！")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"期望2D数组，得到{X.ndim}D数组")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"输入特征数 {X.shape[1]} 与训练时 {self.n_features_in_} 不匹配")

        all_probas = []
        for layer in self.layers_:
            for clf in layer:
                probas = clf.predict_proba(X)
                all_probas.append(probas)
        # 对所有分类器的预测取平均
        return np.mean(all_probas, axis=0)

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
        if 'cascade_forest' in params:
            self.cascade_forest = params['cascade_forest']
        return self


def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        print(f"模型已保存至: {filepath}")


def load_object(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --- 主程序 ---
if __name__ == "__main__":
    print("=" * 50)
    print("开始数据预处理和模型训练...")
    print("=" * 50)

    # 1. 加载数据
    data = pd.read_csv('./clean.csv').drop(columns=['company_id'])
    print(f"原始数据形状: {data.shape}")

    # 2. 编码分类变量
    data = pd.get_dummies(data, drop_first=True)
    print(f"编码后数据形状: {data.shape}")

    # 3. 分离特征和标签
    X = data.drop(columns=['target'])
    y = data['target']
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")

    # 4. 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 5. 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 6. 数据平衡
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
    print(f"平衡后数据形状: {X_resampled.shape}")

    # 7. 宽容度更高的特征选择：保留前 80% 的特征 或 至少 50 个
    n_features = X_resampled.shape[1]
    k = max(50, int(0.8 * n_features))  # 更宽容
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    print(f"特征选择后数量: {X_selected.shape[1]}")

    # 保存预处理组件
    save_object(selector, './feature_selector_xgboost_cemmdan.pkl')
    save_object(imputer, './imputer.pkl')
    save_object(scaler, './scaler.pkl')

    # 8. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_resampled, test_size=0.1, random_state=42)

    # 9. 初始化深度森林（不生成新特征）
    base_enhanced_df = EnhancedDeepForest(
        n_estimators=800,
        max_layers=5,  # 可适当减少
        min_delta=0.0005,
        random_state=42
    )
    enhanced_df = CascadeForestWrapper(base_enhanced_df)

    # 10. 构建集成模型
    adaboost = AdaBoostClassifier(n_estimators=700, algorithm='SAMME', random_state=42)
    # 显式设置 StackingClassifier 的 n_jobs=2，避免使用全部核心
    stacking_model = StackingClassifier(
        estimators=[
            ('enhanced_df', enhanced_df),
            ('adaboost', adaboost)
        ],
        final_estimator=GaussianNB(),
        n_jobs=2,  # 修改点：从 -1 改为 2
        cv=3
    )

    # 11. 训练模型
    print("开始训练模型...")
    stacking_model.fit(X_train, y_train)
    print("模型训练完成！")

    # 12. 保存模型
    save_object(stacking_model, './model_pipeline_xgboost_cemmdan.pkl')

    # 13. 评估
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



