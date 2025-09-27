import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
import pickle
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.utils.validation import check_X_y


# --- 自定义深度森林实现 (已移除多粒度扫描) ---
class EnhancedDeepForest(BaseEstimator, ClassifierMixin):
    """
    增强版深度森林实现（已移除多粒度扫描功能）。

    Attributes:
        n_estimators (int): 每个子模型中的树的数量。
        max_layers (int): 级联的最大层数。
        use_scan (bool): 已禁用多粒度扫描（始终为False）。
        min_delta (float): 触发新增层级所需的最小精度差异阈值。
        random_state (int or None): 控制随机种子。

    Methods:
        fit(X, y): 训练模型。
        predict_proba(X): 返回类别概率估计。
        predict(X): 返回预测类别标签。
    """

    def __init__(self,
                 n_estimators=100,
                 max_layers=5,
                 use_scan=False,  # 已固定为False
                 min_delta=0.001,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.use_scan = use_scan  # 始终为False
        self.min_delta = min_delta
        self.random_state = random_state
        self.cascade_layers_ = []  # 存储每层的基分类器组
        self.classes_ = None
        self._fitted_feature_count = None

    def _create_base_estimators(self):
        """创建一组基础分类器"""
        return [
            ('rf', RandomForestClassifier(n_estimators=self.n_estimators,
                                          random_state=self.random_state,
                                          n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=self.n_estimators,
                                        random_state=self.random_state,
                                        n_jobs=-1))
        ]

    def _evaluate_layer_gain(self, prev_X, current_X, y_true):
        """
        评估当前层带来的性能增益
        """
        # 使用简单分类器评估增益
        prev_score = cross_val_score(GaussianNB(), prev_X, y_true, cv=3, scoring='accuracy').mean()
        curr_score = cross_val_score(GaussianNB(), current_X, y_true, cv=3, scoring='accuracy').mean()
        return (curr_score - prev_score) >= self.min_delta, curr_score - prev_score

    def fit(self, X, y):
        """
        训练深度森林模型（已移除多粒度扫描）
        """
        # 1. 强制转换输入为NumPy数组并验证
        X, y = check_X_y(X, y, ensure_min_features=1)
        print(f"深度森林输入数据类型: {type(X)}, 形状: {X.shape}")

        self.classes_ = np.unique(y)
        self._fitted_feature_count = X.shape[1]  # 记录初始特征数

        # 2. 直接使用原始数据（已移除扫描）
        print("已移除多粒度扫描，直接使用原始特征数据")
        current_input = X.copy()

        # 3. 级联层训练
        print(f"开始级联层训练（最大层数: {self.max_layers}）")
        last_gain = -np.inf
        no_improvement = 0
        patience = 2  # 无改善层数阈值

        for layer_idx in range(self.max_layers):
            print(f"\n--- 训练级联层 {layer_idx + 1} ---")
            print(f"当前输入特征数: {current_input.shape[1]}")

            # 训练基分类器
            layer_estimators = {}
            all_preds = []
            for name, clf in self._create_base_estimators():
                clf = type(clf)(**clf.get_params())  # 克隆分类器
                clf.fit(current_input, y)
                layer_estimators[name] = clf
                all_preds.append(clf.predict_proba(current_input))

            # 合并预测结果作为新特征
            new_features = np.hstack(all_preds)
            print(f"生成新特征数: {new_features.shape[1]}")

            # 组合原始特征和新特征
            combined_features = np.hstack([current_input, new_features])
            print(f"组合后特征数: {combined_features.shape[1]}")

            # 评估性能增益
            gain_detected, gain_value = self._evaluate_layer_gain(
                current_input, combined_features, y)
            print(f"性能增益: {gain_value:.4f}, 是否提升: {gain_detected}")

            if gain_detected or layer_idx == 0:  # 第一层必须保留
                self.cascade_layers_.append(layer_estimators)
                current_input = combined_features
                last_gain = gain_value
                no_improvement = 0
            else:
                no_improvement += 1
                print(f"无显著性能提升，已连续 {no_improvement} 层无改善")
                if no_improvement >= patience:
                    print(f"提前终止级联训练（连续{patience}层无提升）")
                    break

        print(f"\n最终级联层数: {len(self.cascade_layers_)} / 最大层数: {self.max_layers}")
        print(f"最终输入特征数: {current_input.shape[1]}")
        return self

    def _ensure_feature_dimension(self, X, expected_features):
        """确保输入特征维度与预期一致"""
        if X.shape[1] < expected_features:
            # 填充零值
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            return np.hstack([X, padding])
        elif X.shape[1] > expected_features:
            # 截断多余特征
            return X[:, :expected_features]
        return X

    def predict_proba(self, X):
        """获取测试集上的类别概率分布"""
        if not hasattr(self, 'cascade_layers_') or len(self.cascade_layers_) == 0:
            raise RuntimeError("必须先调用fit训练模型！")

        # 1. 强制转换输入为NumPy数组并验证
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"期望2D数组，得到{X.ndim}D数组")

        print(f"预测输入数据类型: {type(X)}, 形状: {X.shape}")

        # 2. 确保初始特征维度一致
        if not hasattr(self, '_fitted_feature_count'):
            raise RuntimeError("模型未正确训练，缺少特征计数信息")

        X = self._ensure_feature_dimension(X, self._fitted_feature_count)

        # 3. 级联层前向传播
        for layer_idx, layer_estimators in enumerate(self.cascade_layers_, 1):
            print(f"级联层 {layer_idx} 输入特征数: {X.shape[1]}")
            all_preds = []
            for clf in layer_estimators.values():
                # 确保特征数匹配
                X = self._ensure_feature_dimension(X, clf.n_features_in_)
                all_preds.append(clf.predict_proba(X))
            new_features = np.hstack(all_preds)
            X = np.hstack([X, new_features])

        # 4. 最终预测
        final_preds = []
        for clf in self.cascade_layers_[-1].values():
            final_preds.append(clf.predict_proba(X))
        return np.mean(final_preds, axis=0)

    def predict(self, X):
        """根据最高置信度返回类别标签"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# --- 包装器保持一致 ---
class CascadeForestWrapper(ClassifierMixin):
    def __init__(self, cascade_forest, **kwargs):
        self.cascade_forest = cascade_forest
        self.classes_ = None

    def fit(self, X, y):
        self.cascade_forest.fit(X, y)
        self.classes_ = np.array(np.unique(y))
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
    """保存对象至磁盘"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        print(f"模型已保存至: {filepath}")


def load_object(filepath):
    """加载已保存的模型"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --- 主程序入口 ---
if __name__ == "__main__":
    print("=" * 50)
    print("开始数据预处理和模型训练...")
    print("=" * 50)

    # 1. 加载数据
    print("加载数据中...")
    data = pd.read_csv('./clean.csv').drop(columns=['company_id'])
    print(f"原始数据形状: {data.shape}")

    # 2. 处理分类变量
    print("编码分类变量...")
    data = pd.get_dummies(data, drop_first=True)
    print(f"编码后数据形状: {data.shape}")

    # 3. 分离特征和标签
    X = data.drop(columns=['target'])
    y = data['target']
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")

    # 4. 处理缺失值
    print("处理缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    print(f"缺失值处理后特征形状: {X_imputed.shape}")

    # 5. 特征缩放
    print("缩放特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"缩放后特征形状: {X_scaled.shape}")

    # 6. 数据平衡
    print("应用SMOTEENN平衡数据...")
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
    print(f"平衡后数据形状: {X_resampled.shape}, 标签形状: {y_resampled.shape}")

    # 7. 特征选择
    print("特征选择...")
    selector = SelectFromModel(xgb.XGBClassifier(n_estimators=100, random_state=42))
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    selected_features = selector.get_support(indices=True)
    print(f"选择后特征数: {X_selected.shape[1]}")
    print(f"被选中的特征索引: {selected_features}")

    # 保存特征选择器
    save_object(selector, './feature_selector_xgboost_cemmdan.pkl')
    save_object(imputer, './imputer.pkl')
    save_object(scaler, './scaler.pkl')

    # 8. 数据集划分
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_resampled, test_size=0.1, random_state=42)
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 9. 初始化增强深度森林（已移除多粒度扫描）
    print("初始化增强深度森林模型...")
    base_enhanced_df = EnhancedDeepForest(
        n_estimators=80,
        max_layers=4,
        use_scan=False,  # 已显式设置为False
        min_delta=0.0005,
        random_state=42
    )

    enhanced_df = CascadeForestWrapper(base_enhanced_df)

    # 10. 构建集成模型
    print("构建集成模型...")
    adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=42)
    stacking_model = StackingClassifier(
        estimators=[
            ('enhanced_df', enhanced_df),
            ('adaboost', adaboost)
        ],
        final_estimator=GaussianNB(),
        n_jobs=-1,
        cv=3  # 添加交叉验证折数
    )

    # 11. 训练模型
    print("开始训练模型...")
    stacking_model.fit(X_train, y_train)
    print("模型训练完成！")

    # 12. 修复：现在在fit之后打印模型结构
    print("\n" + "=" * 50)
    print("集成模型结构:")
    print(f"  - 基分类器: {stacking_model.estimators_}")  # 现在可以安全访问
    print(f"  - 最终分类器: {stacking_model.final_estimator_}")
    print("=" * 50)

    # 13. 保存模型
    save_object(stacking_model, './model_pipeline_xgboost_cemmdan.pkl')
    print("模型已保存至: ./model_pipeline_xgboost_cemmdan.pkl")

    # 14. 评估模型
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
