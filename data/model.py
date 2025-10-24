# -*- coding: utf-8 -*-
"""简单模型对比测试，分别测试XGBoost、AdaBoost、CatBoost和LGBM"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, fbeta_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb

# 尝试导入CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    print("警告: 未找到 catboost。将跳过CatBoost测试。")
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)


def save_object(obj, filepath):
    """将Python对象序列化保存到文件"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"已保存对象至: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("简单模型对比测试启动:")
    print("- 测试模型: XGBoost, AdaBoost, LGBM")
    print("- 如果安装了CatBoost则也测试CatBoost")
    print("- 数据预处理: 方差筛选 -> 缺失值填充 -> 标准化")
    print("- 模型评估: AUC, Accuracy, Precision, Recall, F1-Score")
    print("=" * 60)

    # 配置区域
    DATA_PATH = "./clean.csv"
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    VARIANCE_THRESHOLD_VALUE = 0.0

    # 模型参数配置
    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.3,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': RANDOM_STATE
    }

    ADA_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 1.0,
        'random_state': RANDOM_STATE
    }

    LGBM_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    CAT_PARAMS = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'random_strength': 0.1,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_state': RANDOM_STATE,
        'verbose': False
    }

    # 第一步：加载/创建数据集
    if not os.path.exists(DATA_PATH):
        print(f"未找到数据文件 '{DATA_PATH}'。正在创建示例数据集...")
        N_SAMPLES = 1000
        N_FEATURES_TOTAL = 20
        ZERO_VAR_FEATS = 2

        np.random.seed(RANDOM_STATE)
        SAMPLE_X_BASE = np.random.randn(N_SAMPLES, N_FEATURES_TOTAL - ZERO_VAR_FEATS)
        ZEROS_DATA_BLOCK = np.full((N_SAMPLES, ZERO_VAR_FEATS), 5.0)
        SAMPLE_X_FULL = np.hstack([SAMPLE_X_BASE, ZEROS_DATA_BLOCK])

        TARGET_LABELS_RAW = ((SAMPLE_X_FULL[:, 0] + SAMPLE_X_FULL[:, 1] - SAMPLE_X_FULL[:, 2]) > 0).astype(int)
        POSITIVE_INDICES_ALL = np.where(TARGET_LABELS_RAW == 1)[0]

        REMOVE_COUNT_POSITIVES = int(0.8 * len(POSITIVE_INDICES_ALL))
        if REMOVE_COUNT_POSITIVES > 0:
            INDICES_TO_REMOVE = np.random.choice(POSITIVE_INDICES_ALL, size=REMOVE_COUNT_POSITIVES, replace=False)
            SAMPLE_X_FINAL = np.delete(SAMPLE_X_FULL, INDICES_TO_REMOVE, axis=0)
            SAMPLE_Y_CLEANED = np.delete(TARGET_LABELS_RAW, INDICES_TO_REMOVE, axis=0)
        else:
            SAMPLE_X_FINAL = SAMPLE_X_FULL
            SAMPLE_Y_CLEANED = TARGET_LABELS_RAW

        FEATURE_NAMES_LIST = [f"f{i}" for i in range(N_FEATURES_TOTAL - ZERO_VAR_FEATS)] \
                             + [f"zero_var_{j}" for j in range(ZERO_VAR_FEATS)]

        EXAMPLE_DF = pd.DataFrame(SAMPLE_X_FINAL, columns=FEATURE_NAMES_LIST)
        EXAMPLE_DF['target'] = SAMPLE_Y_CLEANED
        EXAMPLE_DF.to_csv(DATA_PATH, index=False)
        print(f"...示例不平衡数据集已保存到 '{DATA_PATH}'")

    df_raw = pd.read_csv(DATA_PATH)
    if 'company_id' in df_raw.columns:
        df_cleaned = df_raw.drop(columns=["company_id"])
    else:
        df_cleaned = df_raw.copy()

    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    assert "target" in df_encoded.columns, "缺少 'target' 列."

    FEATURES_MATRIX_ALL = df_encoded.drop(columns=["target"]).values
    INITIAL_FEATURE_NAMES = df_encoded.drop(columns=["target"]).columns.tolist()
    LABEL_VECTOR_ALL = df_encoded["target"].values

    TOTAL_SAMPLES_BEFORE_PREPROCESSING = FEATURES_MATRIX_ALL.shape[0]
    POSITIVE_CLASS_COUNT_INITIAL = LABEL_VECTOR_ALL.sum()
    NEGATIVE_CLASS_COUNT_INITIAL = (LABEL_VECTOR_ALL == 0).sum()
    print(f"加载的数据形状: 特征({FEATURES_MATRIX_ALL.shape}), 标签({LABEL_VECTOR_ALL.shape}).")
    print(f"预处理前的类别分布: 正类={POSITIVE_CLASS_COUNT_INITIAL}, 负类={NEGATIVE_CLASS_COUNT_INITIAL}")

    # 第二步：方差筛选
    print(f"应用 VarianceFilter，阈值={VARIANCE_THRESHOLD_VALUE} ...")
    selector_variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD_VALUE)
    FEATURES_AFTER_VARIANCE_FILTERING = selector_variance_filter.fit_transform(FEATURES_MATRIX_ALL)
    SELECTED_FEATURE_IDX_POST_VARIANCE = selector_variance_filter.get_support(indices=True)
    SELECTED_FEATURE_NAMES_POST_VARIANCE = [INITIAL_FEATURE_NAMES[i] for i in SELECTED_FEATURE_IDX_POST_VARIANCE]
    print(
        f"经过 VarianceFilter 后: 保留了 {FEATURES_AFTER_VARIANCE_FILTERING.shape[1]} 个特征 (移除了 {len(INITIAL_FEATURE_NAMES) - len(SELECTED_FEATURE_NAMES_POST_VARIANCE)})")

    # 插补和标准化
    print("对特征进行插补和标准化...")
    knn_imputer_instance = KNNImputer(n_neighbors=5)
    FEATURES_IMPUTED_NO_MISSING = knn_imputer_instance.fit_transform(FEATURES_AFTER_VARIANCE_FILTERING)

    standard_scaler_instance = StandardScaler()
    FEATURES_SCALED_STANDARDIZED = standard_scaler_instance.fit_transform(FEATURES_IMPUTED_NO_MISSING)

    save_object(knn_imputer_instance, "./imputer.pkl")
    save_object(standard_scaler_instance, "./scaler.pkl")
    save_object(selector_variance_filter, "./variance_selector.pkl")

    # 分割数据集
    TRAIN_INPUT_SPLIT, HOLDOUT_EVAL_SPLIT, TRAIN_TARGET_SPLIT, HOLDOUT_TARGET_SPLIT = train_test_split(
        FEATURES_SCALED_STANDARDIZED, LABEL_VECTOR_ALL, test_size=TEST_SIZE, stratify=LABEL_VECTOR_ALL,
        random_state=RANDOM_STATE
    )
    print(f"分割为训练集 ({TRAIN_INPUT_SPLIT.shape}) 和留出验证集 ({HOLDOUT_EVAL_SPLIT.shape})。")

    # 定义模型字典
    models = {
        'XGBoost': xgb.XGBClassifier(**XGB_PARAMS),
        'AdaBoost': AdaBoostClassifier(**ADA_PARAMS),
        'LGBM': lgb.LGBMClassifier(**LGBM_PARAMS)
    }

    # 如果可用，添加CatBoost
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(**CAT_PARAMS)

    # 存储结果
    results = {}

    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n训练 {model_name} 模型...")

        # 训练模型
        model.fit(TRAIN_INPUT_SPLIT, TRAIN_TARGET_SPLIT)

        # 预测
        y_pred_proba = model.predict_proba(HOLDOUT_EVAL_SPLIT)[:, 1]
        y_pred = model.predict(HOLDOUT_EVAL_SPLIT)

        # 计算指标
        auc = roc_auc_score(HOLDOUT_TARGET_SPLIT, y_pred_proba)
        accuracy = accuracy_score(HOLDOUT_TARGET_SPLIT, y_pred)
        precision = precision_score(HOLDOUT_TARGET_SPLIT, y_pred, zero_division=0)
        recall = recall_score(HOLDOUT_TARGET_SPLIT, y_pred, zero_division=0)
        f1 = fbeta_score(HOLDOUT_TARGET_SPLIT, y_pred, beta=1.0, zero_division=0)

        results[model_name] = {
            'AUC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

        print(f"{model_name} 结果:")
        print(f"  AUC = {auc:.5f}")
        print(f"  Accuracy = {accuracy:.5f}")
        print(f"  Precision = {precision:.5f}")
        print(f"  Recall = {recall:.5f}")
        print(f"  F1-Score = {f1:.5f}")

        # 保存模型
        save_object(model, f"./{model_name.lower()}_model.pkl")

    # 汇总结果
    print("\n" + "=" * 60)
    print("模型对比汇总:")
    print("-" * 60)
    print(f"{'模型':<12} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<12} {metrics['AUC']:<10.5f} {metrics['Accuracy']:<10.5f} "
              f"{metrics['Precision']:<10.5f} {metrics['Recall']:<10.5f} {metrics['F1-Score']:<10.5f}")

    print("=" * 60)

    # 找出最佳模型
    best_auc_model = max(results.keys(), key=lambda x: results[x]['AUC'])
    print(f"\nAUC 最佳模型: {best_auc_model} (AUC: {results[best_auc_model]['AUC']:.5f})")

    best_f1_model = max(results.keys(), key=lambda x: results[x]['F1-Score'])
    print(f"F1-Score 最佳模型: {best_f1_model} (F1-Score: {results[best_f1_model]['F1-Score']:.5f})")

    print("\n所有模型已训练并评估完成。")
    print("=" * 60)