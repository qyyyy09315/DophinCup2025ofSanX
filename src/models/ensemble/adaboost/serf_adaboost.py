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


# ------------------- 主流程 -------------------
def main():
    print("开始执行随机森林特征选择 + 集成分类流程...")

    # 1. 数据加载
    data = pd.read_csv("clean.csv")
    if "company_id" in data.columns:
        data = data.drop(columns=["company_id"])
    if "target" not in data.columns:
        raise KeyError("数据中找不到 'target' 列，请检查 clean.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    # 2. 划分数据集
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=SEED,
                                                      stratify=y_train_all)

    # 3. 缺失值填补
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # ✅ 样本重采样（仅训练集）
    print("[Resampling] 开始处理样本不平衡...")
    sampler = SMOTEENN(random_state=SEED)
    X_train_imp, y_train = sampler.fit_resample(X_train_imp, y_train)
    print(f"重采样后训练集分布: {np.bincount(y_train)}")

    # 4. 删除常量特征
    std_devs = np.std(X_train_imp, axis=0)
    non_constant_mask = std_devs > 1e-8  # 修复 1极e-8 -> 1e-8
    X_train_nonconst = X_train_imp[:, non_constant_mask]
    X_val_nonconst = X_val_imp[:, non_constant_mask]
    X_test_nonconst = X_test_imp[:, non_constant_mask]

    print(f"删除常量特征后维度: {X_train_nonconst.shape[1]}")

    # 5. ✅ 执行随机森林特征选择
    print("\n[Random Forest] 开始执行特征选择...")
    selected_indices = random_forest_feature_selection(X_train_nonconst, y_train, n_features_to_select=None)
    print(f"[Random Forest] 最终选择特征数量: {len(selected_indices)}")

    if len(selected_indices) == 0:
        print("⚠未选择任何特征，使用全部非恒定特征")
        selected_indices = list(range(X_train_nonconst.shape[1]))

    # 选择特征
    X_train_selected = X_train_nonconst[:, selected_indices]
    X_val_selected = X_val_nonconst[:, selected_indices]
    X_test_selected = X_test_nonconst[:, selected_indices]

    # 6. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # 7. 定义分类器参数
    adaboost_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'random_state': SEED
    }

    xgboost_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'n_jobs': -1,  # 使用所有CPU核心
        'eval_metric': 'logloss'
    }

    # 8. 训练基分类器
    print("\n[Training] 训练 AdaBoost...")
    adaboost = AdaBoostClassifier(**adaboost_params)
    adaboost.fit(X_train_scaled, y_train)

    print("[Training] 训练 XGBoost...")
    xgboost = XGBClassifier(**xgboost_params)
    xgboost.fit(X_train_scaled, y_train)

    # 9. 构建投票分类器
    voting_classifier = VotingClassifier(
        estimators=[
            ('adaboost', adaboost),
            ('xgboost', xgboost)
        ],
        voting='soft'
    )
    voting_classifier.fit(X_train_scaled, y_train)

    # 10. 阈值优化
    y_proba_val = voting_classifier.predict_proba(X_val_scaled)[:, 1]
    best_threshold, best_gmean = find_optimal_threshold(y_val, y_proba_val)
    print(f"\n[Threshold] 最佳阈值: {best_threshold:.4f}, G-Mean: {best_gmean:.6f}")

    # 11. 测试集评估
    y_proba_test = voting_classifier.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)  # 修复 y_pred极_test -> y_pred_test

    recall = recall_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    final_score = 100 * (0.3 * recall + 0.5 * auc + 0.2 * precision)

    print("=" * 50)
    print("最终评估结果：")
    print(f"Recall:    {recall:.6f}")
    print(f"AUC:       {auc:.6f}")  # 修复 A极UC -> AUC
    print(f"Precision: {precision:.6f}")
    print(f"F1-Score:  {f1:.6f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"使用阈值: {best_threshold:.4f}")
    print("=" * 50)

    # 12. 类别特异性重要性分析
    feature_names = [f"Feature_{i}" for i in range(X_train_scaled.shape[1])]

    for name, model in [("AdaBoost", adaboost), ("XGBoost", xgboost)]:
        print(f"\n[{name}] 类别特异性特征重要性:")
        pos_imp, neg_imp = compute_class_specific_feature_importance(X_val_scaled, y_val, model, feature_names)
        if pos_imp is not None:
            print("→ 正样本 Top 5:")
            print(pos_imp.head(5))
            print("→ 负样本 Top 5:")
            print(neg_imp.head(5))

    # 13. ✅ 全量训练（使用训练集+验证集进行最终训练）
    print("\n[Full Training] 开始全量训练...")

    # 合并训练集和验证集
    X_train_full = np.vstack([X_train_scaled, X_val_scaled])
    y_train_full = np.concatenate([y_train, y_val])

    # 重新训练模型
    print("[Full Training] 训练 AdaBoost...")
    adaboost_full = AdaBoostClassifier(**adaboost_params)
    adaboost_full.fit(X_train_full, y_train_full)

    print("[Full Training] 训练 XGBoost...")
    xgboost_full = XGBClassifier(**xgboost_params)
    xgboost_full.fit(X_train_full, y_train_full)

    # 构建全量投票分类器
    voting_classifier_full = VotingClassifier(
        estimators=[
            ('adaboost', adaboost_full),
            ('xgboost', xgboost_full)
        ],
        voting='soft'
    )
    voting_classifier_full.fit(X_train_full, y_train_full)

    # 使用测试集评估全量模型
    y_proba_test_full = voting_classifier_full.predict_proba(X_test_scaled)[:, 1]
    y_pred_test_full = (y_proba_test_full >= best_threshold).astype(int)

    recall_full = recall_score(y_test, y_pred_test_full)
    auc_full = roc_auc_score(y_test, y_proba_test_full)
    precision_full = precision_score(y_test, y_pred_test_full)
    f1_full = f1_score(y_test, y_pred_test_full)
    final_score_full = 100 * (0.3 * recall_full + 0.5 * auc_full + 0.2 * precision_full)

    print("=" * 50)
    print("全量训练后测试集评估结果：")
    print(f"Recall:    {recall_full:.6f}")
    print(f"AUC:       {auc_full:.6f}")
    print(f"Precision: {precision_full:.6f}")
    print(f"F1-Score:  {f1_full:.6f}")
    print(f"Final Score: {final_score_full:.4f}")
    print("=" * 50)

    # 14. 保存模型
    model = RandomForestFeatureSelectionModel(
        imputer=imputer,
        non_constant_mask=non_constant_mask,
        selected_indices=selected_indices,
        scaler=scaler,
        voting_classifier=voting_classifier_full,  # 使用全量训练的模型
        threshold=best_threshold
    )

    joblib.dump(model, "random_forest_feature_model.pkl")
    print("✅ 模型已保存为 random_forest_feature_model.pkl")


if __name__ == "__main__":
    main()