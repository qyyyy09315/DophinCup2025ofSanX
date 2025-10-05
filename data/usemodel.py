# -*- coding: utf-8 -*-
"""应用已训练的增强版机器学习管道进行预测。"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

# --- 注意：预测阶段不主动导入 gtda ---
# 拓扑特征应在训练时完全计算并包含在最终特征集中，
# 或者其计算对象应一并保存。在此阶段尝试重建会导致不一致。
TOPOLOGY_AVAILABLE = False


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
    """从文件反序列化加载Python对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已从 {filepath} 加载对象")
    return obj


class ModelPipelineWrapper:
    """
    包装整个模型流水线的对象，提供predict和predict_proba接口。
    """

    def __init__(self, model_artifact_dict):
        """
        初始化包装器。

        :param model_artifact_dict: 由训练脚本保存的包含所有组件的字典。
        """
        self.artifact = model_artifact_dict
        # 提取各个组件
        self.imputer = self.artifact["imputer"]
        self.scaler = self.artifact["scaler"]
        self.variance_selector = self.artifact["variance_selector"]

        # 训练时最终选定的特征名称列表
        self.selected_feature_names = self.artifact["selected_feature_names"]

        # 基础模型和元模型
        self.base_model = self.artifact["base_models"]
        self.meta_model = self.artifact["meta_nb"]

        # 决策阈值
        self.threshold = self.artifact["threshold"]

        # 拓扑相关配置 (记录状态，但预测时不执行)
        self.add_topological_features = self.artifact.get("add_topological_features", False)
        if self.add_topological_features and not TOPOLOGY_AVAILABLE:
            print("警告: 模型配置使用了拓扑特征，但预测环境中未启用或缺少 gtda 库。"
                  "预测将基于不含拓扑特征的数据进行，请确认这与训练方式一致。")

        # RFA相关 (用于理解训练过程，但不影响预测核心逻辑)
        self.use_rfa = self.artifact.get("use_rfa", True)

    def _preprocess(self, X_df):
        """
        对输入的DataFrame进行预处理，使其与训练时的格式一致。
        包括：独热编码、方差过滤、插补、标准化、特征对齐。
        """
        print("正在进行预测数据预处理...")

        # 1. 独热编码 (模拟训练时的行为)
        #    假设 X_df 是原始特征，可能需要 OHE。
        #    注意：必须与训练时使用的 get_dummies 参数保持一致。
        X_encoded_df = pd.get_dummies(X_df, drop_first=True)
        print(f"独热编码后特征数: {X_encoded_df.shape[1]}")

        # 2. 获取训练时初始特征名（用于后续步骤参考）
        #    这一步是为了知道哪些特征应该参与 variance_filtering。
        #    实际项目中，这部分信息最好也保存下来。
        initial_feature_names = self._get_initial_feature_names()

        # 3. 确保特征列与训练时经过VarianceFilter后的列对齐
        #    a. 找出训练时经过 VarianceThreshold 后保留的特征名
        selected_feature_names_post_variance_from_training = [
            name for i, name in enumerate(initial_feature_names)
            if i in self.variance_selector.get_support(indices=True)
        ]
        print(f"训练时方差过滤后应保留特征数: {len(selected_feature_names_post_variance_from_training)}")

        #    b. 对齐预测数据集中的特征
        #       缺失的特征用0填充（或根据业务逻辑）
        #       多余的特征会被丢弃
        for feat_name in selected_feature_names_post_variance_from_training:
            if feat_name not in X_encoded_df.columns:
                print(f"警告: 特征 '{feat_name}' 在预测数据中缺失，将以0填充。")
                X_encoded_df[feat_name] = 0

                #    c. 只保留经过方差筛选的特征 (按训练时的顺序)
        try:
            X_aligned_variance_filtered = X_encoded_df[selected_feature_names_post_variance_from_training].values
        except KeyError as e:
            print(f"错误: 预测数据无法正确对齐方差过滤后的特征。缺失特征: {e}")
            raise

        current_feature_names_after_variance = selected_feature_names_post_variance_from_training.copy()
        print(f"方差过滤后特征矩阵形状: {X_aligned_variance_filtered.shape}")

        # 4. 插补 (使用训练时fit的imputer)
        X_imputed = self.imputer.transform(X_aligned_variance_filtered)
        print(f"插补后特征矩阵形状: {X_imputed.shape}")

        # 5. 标准化 (使用训练时fit的scaler)
        X_scaled = self.scaler.transform(X_imputed)
        print(f"标准化后特征矩阵形状: {X_scaled.shape}")

        # 6. 最终特征对齐 (确保列名和顺序与模型期望的一致)
        #    此时 X_scaled 的列对应于 current_feature_names_after_variance

        # 查找最终选中特征在当前处理后特征中的索引
        final_selected_names = self.selected_feature_names
        indices_of_final_selected_features = []
        missing_features_in_final_set = []

        for name in final_selected_names:
            try:
                idx = current_feature_names_after_variance.index(name)
                indices_of_final_selected_features.append(idx)
            except ValueError:
                # 如果找不到，可能是交互项或拓扑特征命名存在差异
                # 但由于我们在预测时不再生成这些，它们必须存在于 current_feature_names_after_variance 中
                # 否则说明训练和预测的数据预处理不匹配
                missing_features_in_final_set.append(name)

        if missing_features_in_final_set:
            error_msg = (f"严重错误: 以下特征在预处理后的特征集中未找到，但却是模型所必需的: "
                         f"{missing_features_in_final_set}. "
                         f"请检查预测数据的特征工程是否与训练时完全一致。")
            print(error_msg)
            raise ValueError(error_msg)

        if not indices_of_final_selected_features:
            raise ValueError("没有找到任何训练时选定的特征。")

        # 选择最终进入模型的特征
        X_final_preprocessed = X_scaled[:, indices_of_final_selected_features]
        print(f"最终预处理后特征矩阵形状: {X_final_preprocessed.shape}, 特征数: {len(final_selected_names)}")

        return X_final_preprocessed

    def _get_initial_feature_names(self):
        """
        辅助函数：重建训练开始时的特征名称列表。
        ***注意***: 这个函数需要与训练脚本严格同步！
        因为它没有被保存到 pkl 中，所以这是一个脆弱的设计点。
        推荐做法是在训练时也将此列表存入 model.pkl 字典。
        """
        # 这些参数需要与训练脚本保持一致才能正确工作
        N_FEATURES_TOTAL = 20  # 示例值，需根据实际情况调整
        ZERO_VAR_FEATS = 2  # 示例值，需根据实际情况调整
        FEATURE_NAMES_LIST = [f"f{i}" for i in range(N_FEATURES_TOTAL - ZERO_VAR_FEATS)] \
                             + [f"zero_var_{j}" for j in range(ZERO_VAR_FEATS)]
        return FEATURE_NAMES_LIST

    def predict_proba(self, X_df):
        """
        对输入数据进行预测，返回各类别的概率。

        :param X_df: pandas.DataFrame, 待预测的数据，列名为特征名。
        :return: numpy.ndarray, shape (n_samples, 2), 第一列为负类概率，第二列为正类概率。
        """
        # 预处理
        X_processed = self._preprocess(X_df)

        # 使用基础模型预测概率
        try:
            probas_base = self.base_model.predict_proba(X_processed)  # 返回 [neg_prob, pos_prob]
        except AttributeError:
            # 如果 base_model 没有 predict_proba，尝试用 predict 并假设输出是概率
            pred_or_score = self.base_model.predict(X_processed)
            if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
                # 假设是一维概率数组
                clipped_probs = np.clip(pred_or_score.flatten(), 0, 1)
                probas_base = np.column_stack([1 - clipped_probs, clipped_probs])
            else:
                # 假设是二维概率数组
                probas_base = pred_or_score

        return probas_base  # 已经是正确的 [neg_prob, pos_prob] 形式

    def predict(self, X_df):
        """
        对输入数据进行预测，返回类别标签 (0 或 1)。

        :param X_df: pandas.DataFrame, 待预测的数据，列名为特征名。
        :return: numpy.ndarray, shape (n_samples,), 类别标签。
        """
        # 预处理
        X_processed = self._preprocess(X_df)

        # 调用级联预测逻辑
        preds, _ = self._cascade_predict(X_processed)
        return preds

    def _cascade_predict(self, X):
        """内部方法：执行级联预测逻辑"""
        try:
            probas_base_full = self.base_model.predict_proba(X)  # [neg_prob, pos_prob]
            probas_base = probas_base_full[:, 1]  # 获取正类概率
        except AttributeError:
            pred_or_score = self.base_model.predict(X)
            if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
                probas_base = np.clip(pred_or_score.flatten(), 0, 1)
            else:
                probas_base = pred_or_score[:, 1]

        # 应用最优阈值进行初步分类
        preds = (probas_base >= self.threshold).astype(int)

        # 确定不确定性样本
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        # 如果有不确定样本且元模型可用，则使用元模型重新预测
        if self.meta_model is not None and uncertain_mask.sum() > 0:
            print(f"发现 {uncertain_mask.sum()} 个不确定性样本，交由元模型处理...")

            if hasattr(self.meta_model, 'predict_proba'):
                try:
                    meta_probas_full = self.meta_model.predict_proba(X[uncertain_mask])
                    meta_preds = (meta_probas_full[:, 1] >= 0.5).astype(int)  # 使用元模型的概率做判断
                except Exception as e:
                    print(f"使用元模型 predict_proba 出错 ({e}), 尝试直接 predict...")
                    meta_preds = self.meta_model.predict(X[uncertain_mask])
            else:
                meta_preds = self.meta_model.predict(X[uncertain_mask])

            preds[uncertain_mask] = meta_preds.astype(int)

        return preds, probas_base  # 返回标签和基础模型的正类概率


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




