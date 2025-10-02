import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # 导入 ExtraTreesClassifier
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 引入进度条库

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """
    修改版cascade_predict，适用于单个基模型。
    修复了当meta_model为None时的错误处理
    """
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        probas_base = base_model.predict(X).astype(float)

    preds = (probas_base >= threshold).astype(int)

    # 简化的不确定性判断：这里我们假设当概率接近0.5时不确定
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)  # 可调整阈值

    # 只有当meta_model不为None且存在不确定样本时才使用meta_model
    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


def f1_2_threshold(y_true, probas):
    """
    基于 F1.2 分数寻找最优阈值。
    F1.2 更重视召回率 (Recall)。
    """
    thresholds = np.linspace(0, 1, 101)
    best_t, best_f1_2 = 0.5, -1
    beta = 1.2  # F1.2 的 beta 值

    for t in thresholds:
        preds = (probas >= t).astype(int)
        # 计算 F1.2 分数
        # 注意：fbeta_score 在某些极端情况下（如所有预测为一类）可能返回 0.0
        # 我们接受这种行为，因为它反映了模型在该阈值下的性能不佳
        try:
            f1_2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        except ValueError:
            # 如果 y_true 或 preds 只有一个类别，fbeta_score 会报错
            # 在这种情况下，我们将其视为 F1.2 = 0
            f1_2 = 0.0

        if f1_2 > best_f1_2:
            best_f1_2, best_t = f1_2, t
    return best_t


def custom_resample(X, y, pos_ratio=1.4, neg_ratio=0.6, random_state=None):
    """
    根据指定的比例对数据进行重采样。
    pos_ratio: 正类相对于原始正类数量的倍数 (例如 1.4 表示变为140%)
    neg_ratio: 负类相对于原始负类数量的倍数 (例如 0.6 表示变为60%)
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError("仅支持二分类问题")
    neg_class, pos_class = unique_classes[0], unique_classes[1]
    neg_count, pos_count = class_counts[0], class_counts[1]

    print(f"原始数据分布: 负类({neg_class})={neg_count}, 正类({pos_class})={pos_count}")

    neg_indices = np.where(y == neg_class)[0]
    pos_indices = np.where(y == pos_class)[0]

    target_neg_count = int(neg_count * neg_ratio)
    target_pos_count = int(pos_count * pos_ratio)

    print(f"目标采样后分布: 负类={target_neg_count}, 正类={target_pos_count}")

    # 下采样负类
    if target_neg_count < neg_count:
        sampled_neg_indices = np.random.choice(neg_indices, size=target_neg_count, replace=False)
    else:  # 上采样负类
        sampled_neg_indices = np.random.choice(neg_indices, size=target_neg_count, replace=True)

    # 上采样正类
    if target_pos_count < pos_count:
        sampled_pos_indices = np.random.choice(pos_indices, size=target_pos_count, replace=False)
    else:  # 上采样正类
        sampled_pos_indices = np.random.choice(pos_indices, size=target_pos_count, replace=True)

    # 合并并打乱
    combined_indices = np.concatenate([sampled_neg_indices, sampled_pos_indices])
    np.random.shuffle(combined_indices)

    X_resampled = X[combined_indices]
    y_resampled = y[combined_indices]

    resampled_neg_count = np.sum(y_resampled == neg_class)
    resampled_pos_count = np.sum(y_resampled == pos_class)
    print(f"重采样后实际分布: 负类={resampled_neg_count}, 正类={resampled_pos_count}")

    return X_resampled, y_resampled


if __name__ == "__main__":

    print("=" * 60)
    print(
        "开始训练：Variance Filter -> Mutual Info Filter -> Balanced Random Forest (Select Features) -> Custom Resample (Pos 1.6x, Neg 0.6x) -> XGBoost -> Cascade Logistic Regression")
    print("-> 已将 DNN 特征加权替换为 ExtraTreesClassifier 特征重要性")
    print("-> 新增按特征重要性排序后选取 Top 90% 的特征")
    print("-> 阈值调优方法已修改为 F1.2")
    print("-> 已移除 SMOTEENN，使用自定义采样方法")
    print("-> 关键修改: 级联修正器由 GaussianNB 改为带 L2 正则化的 LogisticRegression")
    print("-> 新增关键修改: 在BRF前加入基于互信息(Mutual Information)的特征过滤")
    print("=" * 60)

    # ----- 配置 -----
    data_path = "./clean.csv"
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 创建示例数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到数据文件 '{data_path}'。正在创建示例数据...")
        # 示例数据生成代码（简化版）
        sample_n = 1000
        sample_features = 20
        # 添加一些零方差特征
        zero_var_features = 2
        np.random.seed(42)
        sample_X = np.random.randn(sample_n, sample_features - zero_var_features)
        zero_var_data = np.full((sample_n, zero_var_features), 5.0)  # 零方差列
        sample_X = np.hstack([sample_X, zero_var_data])

        # 创建一些有意义的相关性
        sample_y = ((sample_X[:, 0] + sample_X[:, 1] - sample_X[:, 2]) > 0).astype(int)
        feature_names = [f"f{i}" for i in range(sample_features - zero_var_features)] + [f"zero_var_{i}" for i in
                                                                                         range(zero_var_features)]
        sample_df = pd.DataFrame(sample_X, columns=feature_names)
        sample_df['target'] = sample_y
        sample_df.to_csv(data_path, index=False)
        print(f"...示例数据已保存至 '{data_path}'")

    test_size = 0.10
    random_state = 42
    variance_threshold_value = 0.0  # 设置方差阈值
    mi_percentile_to_keep = 0.85  # 关键新增：保留前85%互信息得分的特征
    top_percentile_to_select = 0.9  # 保留前90%重要的特征
    pos_resample_ratio = 1.6  # 正类采样比例
    neg_resample_ratio = 0.6  # 负类采样比例
    num_features_to_select_brf = 90  # BRF选择的特征数

    # XGBoost 超参
    xgb_params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # 用于早停
        'random_state': random_state,
        'n_jobs': -1,
        # 启用 GPU (如果可用且配置了 GPU 支持的 XGBoost)
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
    }

    # ----- 1. 读取并预处理数据 -----
    data = pd.read_csv(data_path)
    # 删除 company_id 列（如果存在）
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # 进行独热编码
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("数据中未找到 'target' 列")

    X_all = data.drop(columns=["target"]).values
    initial_feature_names = data.drop(columns=["target"]).columns.tolist()
    y_all = data["target"].values
    print(f"加载数据: X={X_all.shape}, y={y_all.shape}, positive={y_all.sum()}, negative={(y_all == 0).sum()}")

    # ---- 2. 特征过滤：移除低方差特征 ----
    print(f"应用方差过滤器 (阈值={variance_threshold_value}) ...")
    selector_variance = VarianceThreshold(threshold=variance_threshold_value)
    X_var_filtered = selector_variance.fit_transform(X_all)
    selected_feature_indices_variance = selector_variance.get_support(indices=True)
    selected_feature_names_variance = [initial_feature_names[i] for i in selected_feature_indices_variance]
    print(
        f"方差过滤后特征数: {X_var_filtered.shape[1]} (移除了 {len(initial_feature_names) - len(selected_feature_names_variance)} 个特征)")

    # ---- 3. 数据清洗与标准化 ----
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_var_filtered)
    scaler = StandardScaler()
    X_scaled_temp = scaler.fit_transform(X_imputed)  # 临时标准化用于MI计算

    # ---- 关键新增步骤 4: 互信息特征过滤 ----
    print(f"计算互信息并保留 Top {mi_percentile_to_keep * 100}% 的特征...")
    mi_scores = mutual_info_classif(X_scaled_temp, y_all, random_state=random_state)
    mi_threshold_index = int(len(mi_scores) * mi_percentile_to_keep)
    sorted_mi_indices = np.argsort(-mi_scores)  # 降序排列
    selected_mi_indices = sorted_mi_indices[:mi_threshold_index]

    X_mi_filtered = X_scaled_temp[:, selected_mi_indices]
    selected_feature_names_mi = [selected_feature_names_variance[i] for i in selected_mi_indices]
    print(
        f"互信息过滤后特征数: {X_mi_filtered.shape[1]} (移除了 {len(selected_feature_names_variance) - len(selected_feature_names_mi)} 个特征)")

    # 更新后续使用的变量
    current_X = X_mi_filtered
    current_feature_names = selected_feature_names_mi

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")
    save_object(selector_variance, "./variance_selector.pkl")  # 保存方差过滤器

    # ---- 5. 平衡随机森林特征选择 ----
    print(f"使用平衡随机森林进行特征选择，最多选择 {num_features_to_select_brf} 个特征...")
    # 使用带类权重的随机森林处理不平衡
    brf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1,
                                 class_weight='balanced')
    # SelectFromModel 使用默认阈值（通常是特征重要性的中位数），但我们指定了 max_features
    brf_selector = SelectFromModel(brf, max_features=num_features_to_select_brf, threshold=-np.inf)
    X_brf_selected = brf_selector.fit_transform(current_X, y_all)  # 使用 MI 过滤后的数据
    selected_feature_indices_brf = brf_selector.get_support(indices=True)
    selected_feature_names_brf = [current_feature_names[i] for i in selected_feature_indices_brf]  # 使用 MI 过滤后的特征名
    print(f"平衡随机森林特征选择完成，剩余特征数: {X_brf_selected.shape[1]}")

    save_object(brf_selector, "./brf_feature_selector.pkl")

    # ----- 6. Tree-based 特征权重 (替代DNN) -----
    print("使用 ExtraTreesClassifier 训练获取特征权重 ...")
    # 关键修改: 替换为 ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=300,
                              class_weight='balanced',
                              random_state=random_state,
                              n_jobs=-1)
    et.fit(X_brf_selected, y_all)

    # 获取特征重要性
    feature_importance = et.feature_importances_
    ranked_idx_by_importance = np.argsort(-feature_importance)

    # 关键修改: 根据特征重要性排序选择Top百分比的特征
    num_top_features = int(top_percentile_to_select * len(feature_importance))
    if num_top_features <= 0:
        num_top_features = 1
    selected_top_feature_indices = ranked_idx_by_importance[:num_top_features]

    X_selected = X_brf_selected[:, selected_top_feature_indices]
    selected_feature_names_final = [selected_feature_names_brf[i] for i in selected_top_feature_indices]
    print(
        f"ExtraTrees 特征加权完成，并选择了 Top {top_percentile_to_select * 100}% ({num_top_features}/{len(feature_importance)}) 的特征.")

    save_object(ranked_idx_by_importance, "./tree_feature_ranking.pkl")  # 保存排名

    # ----- 7. 划分训练/验证集 -----
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"训练集: {X_train_full.shape}, 验证集: {X_holdout.shape}")

    # ----- 8. 自定义重采样 (替换 SMOTEENN) -----
    print(f"正在进行自定义重采样 (正类x{pos_resample_ratio}, 负类x{neg_resample_ratio}) ...")
    X_resampled, y_resampled = custom_resample(X_train_full, y_train_full,
                                               pos_ratio=pos_resample_ratio,
                                               neg_ratio=neg_resample_ratio,
                                               random_state=random_state)

    # ----- 9. 训练基模型 (改为 XGBoost) -----
    print("训练 XGBoost 基模型 ...")
    # 准备 XGBoost 数据集以便使用早停功能（可选）
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    # 训练 XGBoost 模型
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,  # 可选：启用早停
        verbose_eval=False,  # 设置为True可以看到训练过程
        evals_result=evals_result
    )


    # 为了兼容后续的 predict_proba 接口，我们可以包装一下
    class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)

            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost.pkl")
    print("已训练基模型：XGBoost")

    # ----- 10. 训练级联修正器 -----
    print("识别基模型误分类样本，训练级联修正器 (使用带 L2 正则化的 Logistic Regression)...")
    # 使用新的预测方法，传入None作为meta_model
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:  # 确保困难样本集中至少有两个类别

        meta_clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=random_state,
                                      max_iter=1000)  # 使用 L2 正则化
        meta_clf.fit(X_hard, y_hard)
        print(f"级联修正器 (Logistic Regression) 训练完成，困难样本数={len(X_hard)}")
    else:
        # 如果没有误分类样本或困难样本只属于一个类别，则使用全部重采样后的数据训练
        # meta_clf = GaussianNB()
        meta_clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=random_state,
                                      max_iter=1000)  # 使用 L2 正则化
        meta_clf.fit(X_resampled, y_resampled)
        print("未发现足够的误分类样本，使用全部重采样数据训练修正器 (Logistic Regression)")

    # ----- 11. 阈值选择（修改为 F1.2） -----
    # 使用新的预测函数
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    # 关键修改: 使用 F1.2 阈值选择函数
    best_thresh = f1_2_threshold(y_holdout, holdout_probas)

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # 计算最终的 F1.2 分数
    f1_2_final = fbeta_score(y_holdout, y_pred_holdout, beta=1.2, zero_division=0)
    # 自定义评分公式
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"最佳阈值 (基于 F1.2)={best_thresh:.4f}, F1={f1:.4f}, F1.2={f1_2_final:.4f}")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, FinalScore={final_score:.5f}")

    # ----- 12. 保存完整模型 -----
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "variance_selector": selector_variance,  # 保存方差过滤器
        "brf_feature_selector": brf_selector,  # 保存BRF特征选择器
        "tree_feature_ranking": ranked_idx_by_importance,  # 保存原始排名索引
        "selected_feature_names": selected_feature_names_final,  # 保存最终选择的特征名称
        "base_models": wrapped_xgb_model,  # 单一模型
        "base_types": ["XGBoost"],  # 类型列表
        "meta_nb": meta_clf,  # 键名保持不变，但内容已经是 LogisticRegression
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model.pkl")
    print("已保存完整模型 ./model.pkl")
    print("=" * 60)




