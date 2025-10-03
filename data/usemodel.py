import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


class WrappedXGBModel:
    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        dmatrix = xgb.DMatrix(X)
        probs = self.booster.predict(dmatrix)
        # For binary classification, Booster returns probabilities for class 1.
        # We need to stack them with 1-probs for class 0 to form [prob_0, prob_1].
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        probs = self.booster.predict(dmatrix)
        return (probs > 0.5).astype(int)

def load_object(filepath):
    """加载通过pickle保存的对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已加载: {filepath}")
    return obj


class ModelPipelineWrapper:
    """
    封装整个模型流水线，包括预处理和预测步骤。
    """

    def __init__(self, model_dict):
        """
        初始化包装器。

        :param model_dict: 包含所有模型组件的字典。
        """
        self.model_dict = model_dict
        # --- 预处理器 ---
        self.imputer = model_dict["imputer"]
        self.scaler = model_dict["scaler"]
        self.variance_selector = model_dict["variance_selector"]

        # --- 特征信息 ---
        self.selected_feature_names = model_dict["selected_feature_names"]
        self.tree_feature_ranking = model_dict["tree_feature_ranking"]  # 保留但本次未使用

        # --- 模型 ---
        self.base_models = model_dict["base_models"]
        self.meta_nb = model_dict["meta_nb"]

        # --- 配置 ---
        self.threshold = model_dict["threshold"]
        self.use_rfa = model_dict.get("use_rfa", True)  # 默认True if missing
        self.pool_of_meta_models_metadata = model_dict.get("pool_of_meta_models_metadata", [])

    def _preprocess(self, X_raw_df):
        """
        对原始特征DataFrame进行预处理。
        注意：此函数假定输入X_raw_df已经去掉了非特征列（如company_id），
             并且其列名对应于训练时的初始特征名。
        """
        # 1. 方差过滤 (使用训练时保存的selector)
        X_var_filtered = self.variance_selector.transform(X_raw_df)
        # 获取方差过滤后对应的特征名索引
        selected_indices_variance = self.variance_selector.get_support(indices=True)
        # 假设 X_raw_df 的列名是初始特征名
        feature_names_after_variance = X_raw_df.columns[selected_indices_variance]

        # 2. KNN 填充缺失值
        X_imputed = self.imputer.transform(X_var_filtered)

        # 3. 标准化
        X_scaled = self.scaler.transform(X_imputed)

        # 4. 应用 RFE 或 RFA 选择的特征
        # 我们需要找到在缩放后的特征中，哪些是我们最终选择的特征
        # selected_feature_names 是最终选择的特征名列表
        # 我们需要从 feature_names_after_variance 中找到它们的索引

        # 创建一个映射：当前特征名 -> 当前方差过滤后特征的索引
        current_feature_to_index = {name: i for i, name in enumerate(feature_names_after_variance)}

        # 找到最终选中的特征在当前特征集中的索引
        final_selected_indices_in_current = [
            current_feature_to_index[name]
            for name in self.selected_feature_names
            if name in current_feature_to_index
        ]

        # 检查是否所有选中的特征都在当前特征集中
        if len(final_selected_indices_in_current) != len(self.selected_feature_names):
            missing_features = set(self.selected_feature_names) - set(feature_names_after_variance)
            raise ValueError(f"测试数据缺少必要的特征: {missing_features}")

        # 应用特征选择
        X_final = X_scaled[:, final_selected_indices_in_current]

        return X_final

    def predict_proba(self, X_raw):
        """
        对原始特征矩阵进行预测，返回正类概率。

        :param X_raw: 原始特征矩阵 (numpy array or DataFrame)。
                      如果是numpy array, 列顺序必须与训练时完全一致。
                      推荐使用DataFrame以利用列名匹配。
        :return: 正类概率数组 (numpy array)。
        """
        # 确保输入是 DataFrame 以便正确处理列名
        if not isinstance(X_raw, pd.DataFrame):
            # 如果不是DataFrame，我们无法安全地进行特征对齐，这里给出警告
            # 并假设列顺序与训练时完全一致。这很脆弱，最好提供DataFrame。
            print("警告: 输入不是 pandas DataFrame。强烈建议提供带有正确列名的 DataFrame 以确保特征对齐。")
            # 为了兼容性，我们可以尝试继续，但这有风险。
            # 我们需要知道训练时的初始特征名来进行正确的预处理。
            # 这里我们做一个简化假设：X_raw 的列数等于训练时经过方差过滤前的特征数
            # 并且顺序完全一致。这是一个强假设，可能导致错误。

            # 一种更健壮的方法是在 model_dict 中保存初始特征名列表
            # 但我们没有这样做。因此，如果输入是 ndarray，
            # 最好要求用户提供一个带有正确列名的 DataFrame。
            # 这里我们临时创建一个假的DataFrame，但这不是一个好的实践。
            # 假设我们知道初始特征数量（这是不对的，因为我们不知道确切的名字）
            # 因此，最安全的做法是强制要求输入为 DataFrame

            # 抛出异常或要求输入为DataFrame
            raise ValueError("输入 `X_raw` 必须是一个 pandas DataFrame，其中包含与训练数据一致的列名。")

        # 如果是 DataFrame，则复制一份避免修改原数据，并进行预处理
        X_processed = self._preprocess(X_raw.copy())

        # --- 级联预测逻辑 ---
        # 1. 基模型预测
        try:
            probas_base = self.base_models.predict_proba(X_processed)[:, 1]
        except Exception as e:
            print(f"基模型 predict_proba 失败: {e}")
            # Fallback: 使用 predict 并假设输出可以直接解释为概率 (不推荐)
            pred_or_score = self.base_models.predict(X_processed)
            if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
                probas_base = np.clip(pred_or_score.flatten(), 0, 1)
            else:
                probas_base = pred_or_score[:, 1]

        preds_initial = (probas_base >= 0.5).astype(int)
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        # 2. 如果有不确定样本且元模型存在，则用元模型修正
        if self.meta_nb is not None and uncertain_mask.sum() > 0:
            # Check if meta_model has predict_proba method for consistency
            if hasattr(self.meta_nb, 'predict_proba'):
                try:
                    meta_probas = self.meta_nb.predict_proba(X_processed[uncertain_mask])[:, 1]
                    # 不直接替换概率，而是可以替换预测结果，或者根据需求融合概率
                    # 这里我们按照原始cascade_predict_single_model的逻辑，
                    # 只更新那些被判定为“不确定”的样本的预测标签
                    # 但返回的仍然是基模型的概率
                    # 如果需要返回融合后的概率，逻辑会更复杂
                    # 按照原始意图，只修正预测，不改变返回的概率？
                    # 但是 cascade_predict_single_model 返回了修正后的 preds 和 base probas
                    # 为了让 predict 方法工作，我们需要存储修正后的预测
                    # 但 predict_proba 通常只返回主模型概率
                    # 这里保持 predict_proba 返回基模型概率不变

                except Exception as e:
                    print(f"元模型 predict_proba 失败: {e}")
                    meta_preds = self.meta_nb.predict(X_processed[uncertain_mask])
            else:
                meta_preds = self.meta_nb.predict(X_processed[uncertain_mask])

            # 更新初始预测（仅用于内部状态或predict方法）
            # 注意：predict_proba本身不修改其返回的概率值
            # 但如果需要融合概率，需要在这里实现不同的逻辑
            # 目前按惯例，predict_proba返回基模型概率
            # 而predict方法结合阈值和可能的级联修正来决定最终类别

            # 存储修正后的预测供 predict 方法使用
            # 我们不能直接修改 preds_initial 因为 predict_proba 应该是无状态的
            # 最好的方式是在 predict 方法中重新执行这部分逻辑
            pass  # 在 predict 方法中处理

        return np.column_stack([1 - probas_base, probas_base])  # 返回 [prob_0, prob_1]

    def predict(self, X_raw):
        """
        对原始特征矩阵进行预测，返回类别标签。

        :param X_raw: 原始特征矩阵 (numpy array or DataFrame)。
        :return: 类别标签数组 (numpy array)。
        """
        # 确保输入是 DataFrame 以便正确处理列名
        if not isinstance(X_raw, pd.DataFrame):
            raise ValueError("输入 `X_raw` 必须是一个 pandas DataFrame，其中包含与训练数据一致的列名。")

        # 预处理
        X_processed = self._preprocess(X_raw.copy())

        # --- 级联预测逻辑 ---
        # 1. 基模型预测
        try:
            probas_base = self.base_models.predict_proba(X_processed)[:, 1]
        except Exception as e:
            print(f"基模型 predict_proba 失败: {e}")
            pred_or_score = self.base_models.predict(X_processed)
            if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
                probas_base = np.clip(pred_or_score.flatten(), 0, 1)
            else:
                probas_base = pred_or_score[:, 1]

        preds = (probas_base >= self.threshold).astype(int)  # 使用加载的阈值
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        # 2. 如果有不确定样本且元模型存在，则用元模型修正
        if self.meta_nb is not None and uncertain_mask.sum() > 0:
            if hasattr(self.meta_nb, 'predict_proba'):
                try:
                    meta_probas = self.meta_nb.predict_proba(X_processed[uncertain_mask])[:, 1]
                    meta_preds = (meta_probas >= 0.5).astype(int)  # 元模型内部使用0.5阈值?
                except Exception as e:
                    print(f"元模型 predict_proba 失败: {e}")
                    meta_preds = self.meta_nb.predict(X_processed[uncertain_mask])
            else:
                meta_preds = self.meta_nb.predict(X_processed[uncertain_mask])

            preds[uncertain_mask] = meta_preds.astype(int)

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
        exit(1)

    print(f"应用分类阈值: {threshold:.4f}")
    # 获取最终预测结果 (0 或 1)
    try:
        y_pred = loaded_pipeline.predict(X_test_raw_df)
        print("分类完成。")
    except Exception as e:
        print(f"分类时发生错误: {e}")
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
