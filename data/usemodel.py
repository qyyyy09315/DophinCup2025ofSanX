# -*- coding: utf-8 -*-
"""加载训练好的模型并应用于测试数据，输出预测结果到桌面。"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier


class WrappedXGBPredictiveModel:
    def __init__(self, booster_obj_ref):
        self.booster_internal_reference = booster_obj_ref

    def predict_proba(self, input_features_array_like):
        dmatrix_format_input = xgb.DMatrix(input_features_array_like)
        probabilities_positives_only = self.booster_internal_reference.predict(dmatrix_format_input)
        return np.column_stack([1 - probabilities_positives_only, probabilities_positives_only])

    def predict(self, input_features_array_like):
        dmatrix_format_input = xgb.DMatrix(input_features_array_like)
        predicted_probabilities = self.booster_internal_reference.predict(dmatrix_format_input)
        return (predicted_probabilities > 0.5).astype(int)

def load_object(filepath):
    """从文件中反序列化加载Python对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已从 {filepath} 加载对象")
    return obj


class TargetEncoder:
    """简单的均值目标编码器 (推理时使用)"""

    def __init__(self):
        self.mapping_ = {}

    def transform(self, X):
        X = np.asarray(X).flatten()
        # 对于未见过的类别，使用全局均值
        default_val = np.mean(list(self.mapping_.values())) if self.mapping_ else 0
        return np.array([self.mapping_.get(x, default_val) for x in X]).reshape(-1, 1)


# === 新增函数：分箱离散化 (推理时使用) ===
def bin_features(X, discretizer):
    """使用已训练的分箱器对特征进行分箱离散化"""
    if X.size == 0:
        return X
    X_binned = discretizer.transform(X)
    return X_binned


# === 新增函数：分级特征处理 (推理时使用) ===
# 注意：推理时使用的 FEATURE_HIERARCHY 需要与训练时完全一致
FEATURE_HIERARCHY = {
    "core": ['f0', 'f1', 'f2'],  # 示例核心特征
    "secondary": [r'f[3-5]'],  # 示例次级特征
    "contextual": [r'f[6-9]']  # 示例上下文特征
}
import re  # 确保导入正则表达式库


def hierarchical_processing_inference(X, feature_names, encoders_dict, discretizers_dict):
    """
    推理时的分级特征处理
    核心财务特征：保留原始数值
    次级特征：使用训练时保存的分箱器分箱
    上下文特征：使用训练时保存的目标编码器编码
    """
    print("正在执行推理时的分级特征处理...")
    core_idx = [i for i, n in enumerate(feature_names)
                if any(n == p for p in FEATURE_HIERARCHY['core'])]

    sec_idx = [i for i, n in enumerate(feature_names)
               if any(re.match(p, n) for p in FEATURE_HIERARCHY['secondary'])]

    ctx_idx = [i for i, n in enumerate(feature_names)
               if any(re.match(p, n) for p in FEATURE_HIERARCHY['contextual'])]

    processed_parts = []
    processed_names = []

    # 核心特征：直接保留
    if core_idx:
        processed_parts.append(X[:, core_idx])
        processed_names.extend([feature_names[i] for i in core_idx])
        print(f"  - 核心特征 ({len(core_idx)} 个): 已保留原始数值")

    # 次级特征：分箱 (使用保存的分箱器)
    if sec_idx:
        X_sec_list = []
        for i in sec_idx:
            disc = discretizers_dict.get(feature_names[i], None)
            if disc is not None:
                X_sec_binned = bin_features(X[:, i].reshape(-1, 1), disc)
                X_sec_list.append(X_sec_binned.flatten())
            else:
                print(f"  - 警告: 未找到特征 '{feature_names[i]}' 的分箱器，将保留原始数值。")
                X_sec_list.append(X[:, i])
        if X_sec_list:
            X_sec_final = np.column_stack(X_sec_list)
            processed_parts.append(X_sec_final)
            processed_names.extend([f"{feature_names[i]}_binned" for i in sec_idx])
            print(f"  - 次级特征 ({len(sec_idx)} 个): 已进行分箱离散化")
        else:
            print("  - 次级特征处理失败。")

    # 上下文特征：目标编码 (使用保存的编码器)
    if ctx_idx:
        X_ctx_list = []
        ctx_names_out = []
        for i in ctx_idx:
            te = encoders_dict.get(feature_names[i], None)
            if te is not None:
                # 假设上下文特征是字符串或可以转换为字符串的类别
                feat_vals = X[:, i].astype(str)
                X_ctx_enc = te.transform(feat_vals)
                X_ctx_list.append(X_ctx_enc.flatten())
                ctx_names_out.append(f"{feature_names[i]}_target_encoded")
            else:
                print(f"  - 警告: 未找到特征 '{feature_names[i]}' 的目标编码器，将保留原始数值。")
                X_ctx_list.append(X[:, i])
        if X_ctx_list:
            X_ctx_final = np.column_stack(X_ctx_list)
            processed_parts.append(X_ctx_final)
            processed_names.extend(ctx_names_out)
            print(f"  - 上下文特征 ({len(ctx_idx)} 个): 已进行目标编码")
        else:
            print("  - 上下文特征处理失败。")

    if processed_parts:
        X_processed = np.hstack(processed_parts)
        print(f"推理分级处理完成，输出特征数: {X_processed.shape[1]}")
        return X_processed, processed_names
    else:
        print("未找到匹配任何分级的特征，返回原始数据。")
        return X, feature_names


class ModelPipelineWrapper:
    """包装加载的模型字典，提供统一的预测接口"""

    def __init__(self, model_artifact_dict):
        self.artifact_dict = model_artifact_dict
        # 从字典中提取各个组件
        self.imputer = self.artifact_dict["imputer"]
        self.scaler = self.artifact_dict["scaler"]
        self.variance_selector = self.artifact_dict["variance_selector"]
        self.selected_feature_names = self.artifact_dict["selected_feature_names"]
        self.base_model = self.artifact_dict["base_models"]
        self.meta_model = self.artifact_dict["meta_nb"]
        self.threshold = self.artifact_dict["threshold"]
        self.use_hierarchical_processing = self.artifact_dict.get("use_hierarchical_processing", False)
        # 注意：训练时未保存 encoders 和 discretizers，推理时无法直接使用。
        # 这是一个简化示例。在完整实现中，应在训练时保存这些对象。
        self.encoders = {}  # 在此示例中为空
        self.discretizers = {}  # 在此示例中为空

    def _preprocess(self, X_df):
        """内部预处理函数"""
        print("开始预处理测试数据...")
        # 1. 确保列顺序与训练时一致（如果需要）
        #    这里假设输入DataFrame的列已经与训练时选择的特征一致
        #    实际应用中可能需要更复杂的映射和处理

        # 2. 独热编码 (如果训练时有)
        #    这里假设测试数据已经经过了与训练数据相同的预处理步骤
        #    例如，分类变量已经被正确编码或在训练时被丢弃
        #    为简化，我们假设输入已经是数值型且列名匹配
        feature_names_initial = X_df.columns.tolist()
        X_numeric = X_df.values

        # 3. 方差筛选
        print("应用 VarianceFilter...")
        X_after_variance = self.variance_selector.transform(X_numeric)
        selected_feature_idx_post_variance = self.variance_selector.get_support(indices=True)
        selected_feature_names_post_variance = [feature_names_initial[i] for i in selected_feature_idx_post_variance]
        print(
            f"经过 VarianceFilter 后: 保留了 {X_after_variance.shape[1]} 个特征 (移除了 {len(feature_names_initial) - len(selected_feature_names_post_variance)})")

        # 4. 分级处理 (如果启用)
        if self.use_hierarchical_processing:
            # 注意：此示例中 encoders 和 discretizers 为空，因此实际效果是保留原始特征或使用默认处理
            # 在完整实现中，需要加载训练时保存的这些对象
            try:
                X_hierarchical_processed, names_hierarchical_processed = hierarchical_processing_inference(
                    X_after_variance, selected_feature_names_post_variance, self.encoders, self.discretizers
                )
                X_after_hierarchical = X_hierarchical_processed
                feature_names_after_hierarchical = names_hierarchical_processed
                print(f"分级处理后特征数: {len(feature_names_after_hierarchical)}")
            except Exception as e:
                print(f"推理时分级处理失败: {e}。回退到方差筛选后的特征。")
                X_after_hierarchical = X_after_variance
                feature_names_after_hierarchical = selected_feature_names_post_variance
        else:
            X_after_hierarchical = X_after_variance
            feature_names_after_hierarchical = selected_feature_names_post_variance

        # 5. 插补和标准化
        print("进行插补和标准化...")
        X_imputed = self.imputer.transform(X_after_hierarchical)
        X_scaled = self.scaler.transform(X_imputed)

        # 6. 特征对齐 (确保特征顺序和数量与模型训练时一致)
        #    假设 self.selected_feature_names 已经包含了所有需要的特征（包括可能的拓扑特征）
        #    并且经过上述步骤后，X_scaled 的列已经与之对应。
        #    如果存在不匹配（例如，训练时用了RFA但这里没有），则需要更复杂的映射逻辑。
        #    此示例假设处理后的特征与模型期望的特征完全匹配。

        print("预处理完成。")
        return X_scaled

    def predict_proba(self, X_df):
        """获取预测概率"""
        X_processed = self._preprocess(X_df)
        # 假设 base_model 是 WrappedXGBPredictiveModel 或类似接口
        return self.base_model.predict_proba(X_processed)

    def predict(self, X_df):
        """获取最终预测结果"""
        X_processed = self._preprocess(X_df)
        # 假设 base_model 和 meta_model 有 predict_proba 或 predict 方法
        # 这里简化实现，直接使用基础模型的概率和阈值
        # 级联预测逻辑可以更复杂，但需要保存 meta_model 的输入特征对齐信息
        probas = self.base_model.predict_proba(X_processed)[:, 1]
        preds = (probas >= self.threshold).astype(int)
        return preds


if __name__ == "__main__":
    # -----------------------------
    # 1. 加载测试数据
    # -----------------------------
    test_data_path = "testClean.csv"
    print(f"正在加载测试数据: {test_data_path}")
    if not os.path.exists(test_data_path):
        # 创建示例测试数据用于演示目的，实际运行时请替换为真实数据路径
        print(f"警告: 未找到测试数据文件 '{test_data_path}'。正在创建示例测试数据...")
        np.random.seed(100)  # 固定种子以便示例一致
        sample_n = 200
        # 模拟训练脚本中生成的数据结构
        # 训练时有 f0-f(N-zero_var_features-1), zero_var_0, zero_var_1
        # 示例用了20个特征，其中2个是零方差的
        feature_cols_simulation = [f'f{i}' for i in range(18)] + ['zero_var_0', 'zero_var_1']
        sample_test_data_dict = {
            col: np.random.randn(sample_n) for col in feature_cols_simulation if not col.startswith('zero_var')
        }
        # 添加零方差特征
        sample_test_data_dict['zero_var_0'] = [5.0] * sample_n
        sample_test_data_dict['zero_var_1'] = [5.0] * sample_n
        # 添加 company_id 列
        sample_test_data_dict['company_id'] = [f"COMP_{i}" for i in range(sample_n)]

        sample_test_data = pd.DataFrame(sample_test_data_dict)

        sample_test_data.to_csv(test_data_path, index=False)
        print(f"已生成示例测试数据并保存至 {test_data_path}")

    test_data = pd.read_csv(test_data_path)
    print(
        f"测试数据加载成功，样本数: {test_data.shape[0]}, 特征数: {test_data.shape[1] - 1 if 'company_id' in test_data.columns else test_data.shape[1]}")

    # -----------------------------
    # 2. 加载模型字典并包装
    # -----------------------------
    model_path = "./model.pkl"  # 使用训练脚本最后保存的模型文件
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。请确保模型已训练并保存。")
        exit(1)

    print(f"正在加载模型字典: {model_path}")
    loaded_model_dict = load_object(model_path)

    # 包装模型字典以便使用
    loaded_pipeline = ModelPipelineWrapper(loaded_model_dict)

    # -----------------------------
    # 3. 读取阈值
    # -----------------------------
    threshold = loaded_pipeline.threshold
    print(f"从模型对象读取阈值: {threshold:.4f}")

    # -----------------------------
    # 4. 测试数据处理与预测
    # -----------------------------
    # 准备特征数据 (保留列名为DataFrame)
    feature_columns = [col for col in test_data.columns if col != "company_id"]
    X_test_raw_df = test_data[feature_columns]  # 保留为 DataFrame
    print(f"用于预测的原始特征数: {X_test_raw_df.shape[1]}")

    print("正在进行预测...")
    # 获取概率 (注意 predict_proba 返回的是二维数组 [[neg_prob, pos_prob], ...])
    try:
        y_proba_full = loaded_pipeline.predict_proba(X_test_raw_df)
        y_proba = y_proba_full[:, 1]  # 获取正类概率
        print(f"预测完成，共 {len(y_proba)} 个概率值。")
    except Exception as e:
        print(f"预测概率时发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    print(f"应用分类阈值: {threshold:.4f}")
    # 获取最终预测结果 (0 或 1)
    try:
        y_pred = loaded_pipeline.predict(X_test_raw_df)
        print("分类完成。")
    except Exception as e:
        print(f"分类时发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # -----------------------------
    # 5. 结果输出到桌面
    # -----------------------------
    if "company_id" in test_data.columns:
        uuid_column = test_data["company_id"]
    else:
        print("警告: 测试数据中未找到 'company_id' 列，将使用行索引作为 uuid。")
        uuid_column = test_data.index.map(str)  # 确保索引是字符串类型

    results_df = pd.DataFrame({
        "uuid": uuid_column,
        "proba": y_proba,
        "prediction": y_pred
    })
    print("结果数据框创建成功。")

    # 构造桌面路径 (跨平台基本兼容的方式)
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop_path):
        desktop_path = "."  # 如果桌面路径不存在，则保存到当前目录

    output_filename = "submit_template.csv"
    output_path = os.path.join(desktop_path, output_filename)

    print(f"正在保存结果到: {output_path}")
    try:
        results_df.to_csv(output_path, index=False)
        print(f"预测完成，阈值 {threshold:.4f}，结果已保存到 {output_path}")
    except Exception as e:
        print(f"保存结果时发生错误: {e}")




