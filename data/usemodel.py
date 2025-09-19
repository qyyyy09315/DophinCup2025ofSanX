import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

# 设置随机种子和并行计算
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "32"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ------------------- 阈值搜索函数（优化 G-Mean）-------------------
def find_optimal_threshold(y_true, y_proba, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_threshold = 0.5  # 修复拼写错误
    best_gmean = 0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        gmean = np.sqrt(rec * prec) if rec > 0 and prec > 0 else 0
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = th  # 修复：之前变量名拼错
    return best_threshold, best_gmean


# ------------------- 类别特异性特征重要性（近似）-------------------
def compute_class_specific_feature_importance(X, y, model, feature_names=None):
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_importances_"):
        return None, None

    proba = model.predict_proba(X)  # 保留完整二维概率
    importances = model.feature_importances_

    pos_idx = y == 1
    neg_idx = y == 0
    if pos_idx.sum() == 0 or neg_idx.sum() == 0:
        return None, None

    # 使用正类概率加权正样本，负类概率（1 - proba[:,1]）加权负样本
    pos_weighted = np.mean(proba[pos_idx, 1][:, np.newaxis] * importances, axis=0)  # 修复 axis 拼写 + 维度
    neg_weighted = np.mean((1 - proba[neg_idx, 1])[:, np.newaxis] * importances, axis=0)  # 修复 axis 拼写 + 维度

    pos_weighted /= (np.sum(pos_weighted) + 1e-8)
    neg_weighted /= (np.sum(neg_weighted) + 1e-8)

    if feature_names is not None:
        pos_df = pd.DataFrame({'feature': feature_names, 'importance': pos_weighted}).sort_values('importance',
                                                                                                  ascending=False)
        neg_df = pd.DataFrame({'feature': feature_names, 'importance': neg_weighted}).sort_values('importance',
                                                                                                  ascending=False)
        return pos_df, neg_df
    return pos_weighted, neg_weighted


# ------------------- 替换模块：使用随机森林进行特征选择 -------------------
def random_forest_feature_selection(X_train, y_train, n_features_to_select=None):
    """
    使用随机森林选择特征
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param n_features_to_select: 最终保留的特征数量，默认为总特征数的 30%
    :return: 选中的特征索引列表
    """
    n_features = X_train.shape[1]
    if n_features_to_select is None:
        n_features_to_select = max(1, int(0.3 * n_features))  # 默认保留30%

    print(f"[Random Forest] 总特征数: {n_features}, 目标保留: {n_features_to_select}")

    # 配置随机森林参数以最大化CPU利用率
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,  # 使用所有可用CPU核心
        verbose=0  # 减少输出干扰
    )

    # 训练随机森林
    print("[Random Forest] 训练随机森林模型进行特征选择...")
    rf.fit(X_train, y_train)

    # 基于特征重要性选择特征
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]  # 修复 args极ort -> argsort

    # 选择最重要的特征
    selected_indices = sorted_indices[:n_features_to_select].tolist()

    print(f"[Random Forest] 最终选择特征数量: {len(selected_indices)}")
    print(f"[Random Forest] 特征重要性范围: {importances.min():.6f} - {importances.max():.6f}")

    return selected_indices


# ------------------- 自定义模型类（修复PicklingError）-------------------
class RandomForestFeatureSelectionModel:
    def __init__(self, imputer, non_constant_mask, selected_indices, scaler, voting_classifier, threshold):
        self.imputer = imputer
        self.non_constant_mask = non_constant_mask
        self.selected_indices = selected_indices
        self.scaler = scaler
        self.voting_classifier = voting_classifier
        self.threshold = threshold

    def predict_proba(self, X_raw):
        X_imp = self.imputer.transform(X_raw)
        X_nonconst = X_imp[:, self.non_constant_mask]
        if len(self.selected_indices) > 0:
            X_selected = X_nonconst[:, self.selected_indices]
        else:
            X_selected = X_nonconst
        X_scaled = self.scaler.transform(X_selected)
        return self.voting_classifier.predict_proba(X_scaled)

    def predict(self, X_raw):
        proba = self.predict_proba(X_raw)[:, 1]
        return (proba >= self.threshold).astype(int)

# 以上填写对象的定义
test_data = pd.read_csv(r'testClean.csv')

# 2. 加载完整的级联模型（已包含所有预处理步骤）
cascaded_model = joblib.load('random_forest_feature_model.pkl')

# 3. 应用预处理和模型预测
# 删除 'company_id' 列（如果存在）
X_test = test_data.drop(columns=['company_id'], errors='ignore')

# 直接使用级联模型进行预测
y_proba = cascaded_model.predict_proba(X_test)[:, 1]  # 类别1的概率
y_pred = cascaded_model.predict(X_test)  # 类别预测

# 4. 创建结果数据框
results_df = pd.DataFrame({
    'uuid': test_data['company_id'],  # 保留原始company_id
    'proba': y_proba,
    'prediction': y_pred
})

# 5. 保存结果
results_df.to_csv(r'C:\Users\YKSHb\Desktop\submit_template.csv', index=False)

print("预测结果已保存到 submit_template.csv")
