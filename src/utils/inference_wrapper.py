# -*- coding: utf-8 -*-
"""加载训练好的模型并应用于测试数据，输出预测结果到桌面。"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SimpleDNN(nn.Module):
    """一个简单的深度神经网络分类器"""

    def __init__(self, input_dim, hidden_layers=[128, 64], dropout_rate=0.3):
        super(SimpleDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # 输出 logits
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class WrappedDNNPredictiveModel:
    """包装 PyTorch DNN 模型以符合 scikit-learn 接口"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor).squeeze()
            probs_positive = torch.sigmoid(logits).cpu().numpy()
            # 返回 [prob_negative, prob_positive]
            return np.column_stack([1 - probs_positive, probs_positive])

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor).squeeze()
            predictions = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            return predictions

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


# -*- coding: utf-8 -*-
"""用于加载训练好的增强版机器学习管道并对新数据进行预测的脚本。"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# 尝试导入 giotto-tda 用于拓扑数据分析 (如果训练时使用了)
try:
    from gtda.homology import VietorisRipsPersistence

    TOPOLOGY_AVAILABLE = True
except ImportError:
    print("警告: 未找到 gtda (giotto-tda)。如果模型使用了拓扑特征，预测可能失败。")
    TOPOLOGY_AVAILABLE = False


def load_object(filepath):
    """从文件反序列化加载Python对象"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"已从 {filepath} 加载对象。")
    return obj


class ModelPipelineWrapper:
    """
    包装加载的模型字典，提供统一的 predict 和 predict_proba 接口。
    """

    def __init__(self, model_dict):
        self.model_dict = model_dict
        # 从字典中提取组件
        self.imputer = self.model_dict['imputer']
        self.scaler = self.model_dict['scaler']
        self.variance_selector = self.model_dict['variance_selector']
        self.tree_feature_ranking = self.model_dict['tree_feature_ranking']
        self.selected_feature_names = self.model_dict['selected_feature_names']
        self.base_models = self.model_dict['base_models']
        self.base_types = self.model_dict['base_types']
        self.meta_nb = self.model_dict['meta_nb']
        self.threshold = self.model_dict['threshold']

        # 可选的高级配置
        self.use_rfa = self.model_dict.get('use_rfa', False)
        self.add_topological_features = self.model_dict.get('add_topological_features', False)
        self.topo_homology_dims = self.model_dict.get('topo_homology_dims', [0, 1])
        self.use_adaptive_scales = self.model_dict.get('use_adaptive_scales', False)
        self.use_hierarchical_processing = self.model_dict.get('use_hierarchical_processing', False)

        # 如果模型使用了拓扑特征，则初始化持久同调计算器
        if self.add_topological_features and TOPOLOGY_AVAILABLE:
            self.persistence = VietorisRipsPersistence(
                metric="euclidean",
                homology_dimensions=self.topo_homology_dims,
                collapse_edges=True,
                max_edge_length=np.inf,
                infinity_values=None,
                n_jobs=1
            )
        elif self.add_topological_features and not TOPOLOGY_AVAILABLE:
            print("错误: 模型需要拓扑特征，但运行时环境中未安装 'giotto-tda'。")
            raise RuntimeError("Missing required library 'giotto-tda' for topological features.")

    def _extract_topological_features(self, X_sample):
        """内部方法：为单个样本提取拓扑特征（简化版）"""
        # 注意：这是一个非常低效的实现，仅为兼容性。
        # 实际应用中应考虑批量处理或近似方法。
        try:
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)

            # gtda 要求输入是三维数组 (n_samples, n_points, n_dimensions)
            X_diagram_input = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1])

            diagrams = self.persistence.fit_transform(X_diagram_input)
            diagram = diagrams[0]

            topo_features = []
            for dim in self.topo_homology_dims:
                mask = (diagram[:, 2] == dim)
                births_deaths_dim = diagram[mask][:, :2]

                if len(births_deaths_dim) > 0:
                    lifetimes = births_deaths_dim[:, 1] - births_deaths_dim[:, 0]
                    topo_features.extend([
                        np.sum(mask),
                        np.max(lifetimes) if len(lifetimes) > 0 else 0,
                        np.mean(lifetimes) if len(lifetimes) > 0 else 0,
                        np.std(lifetimes) if len(lifetimes) > 1 else 0,
                    ])
                else:
                    topo_features.extend([0, 0, 0, 0])

            max_death_global = np.max(diagram[:, 1]) if diagram.size > 0 else 0
            topo_features.append(max_death_global)

            return np.array(topo_features)
        except Exception as e:
            print(f"警告: 计算样本的拓扑特征时出错: {e}")
            expected_size = len(self.topo_homology_dims) * 4 + 1
            return np.zeros(expected_size)

    def predict_proba(self, X_raw_df):
        """
        对原始特征DataFrame进行完整的预处理和预测，返回正类概率。
        """
        print("开始预测流程...")
        # 1. 方差筛选
        print("1. 应用方差筛选...")
        X_var_filtered = self.variance_selector.transform(X_raw_df)
        current_feature_names = [X_raw_df.columns[i] for i in self.variance_selector.get_support(indices=True)]

        # 2. 分级处理 (如果训练时启用了)
        # 注意：此处假设测试数据不需要标签y进行处理，因此目标编码部分可能不适用或需要预先fit的编码器。
        # 为了简化，我们在这里仅保留核心和次级特征的处理逻辑。
        if self.use_hierarchical_processing:
            print("2. 应用分级特征处理 (简化版)...")
            # 这是一个简化的版本，实际应用中可能需要更复杂的逻辑来匹配训练时的行为。
            # 例如，保存和加载目标编码器。
            # 此处我们假设只对核心特征做处理，其他保持不变或按规则变换。
            # 由于缺乏训练时保存的编码器，上下文特征的目标编码在此被跳过或以默认方式处理。
            # 一种保守的做法是直接使用方差筛选后的特征作为下一步输入。
            # 更精确的方法需要重构hierarchical_processing函数使其能接受预训练的转换器。
            # 为保持一致性，这里我们暂时沿用方差筛选后的结果。
            # TODO: 如果需要完全复现训练时的分级处理，需重构并保存相关转换器。

        # 3. 插补缺失值
        print("3. 插补缺失值...")
        X_imputed = self.imputer.transform(X_var_filtered)

        # 4. 标准化
        print("4. 特征标准化...")
        X_scaled = self.scaler.transform(X_imputed)

        # 5. 添加拓扑特征 (如果训练时启用了)
        if self.add_topological_features:
            print("5. 添加拓扑特征...")
            topo_features_list = []
            for i, sample in enumerate(X_scaled):
                if (i + 1) % 100 == 0 or i == len(X_scaled) - 1:
                    print(f"   处理第 {i + 1}/{len(X_scaled)} 个样本的拓扑特征...")
                tf = self._extract_topological_features(sample)
                topo_features_list.append(tf)

            X_topo = np.array(topo_features_list)
            X_final = np.hstack([X_scaled, X_topo])
        else:
            X_final = X_scaled

        # 6. 级联预测
        print("6. 执行级联预测...")
        # 假设cascade_predict_single_model的逻辑在这里展开
        # base_model 预测
        try:
            # 尝试直接调用模型的predict_proba
            probas_base = self.base_models.predict_proba(X_final)[:, 1]
        except AttributeError:
            # 如果没有predict_proba，则尝试predict并假设其输出可以直接解释为概率
            # 或者是logits/scores
            pred_or_score = self.base_models.predict(X_final)
            if len(pred_or_score.shape) == 1 or pred_or_score.shape[1] == 1:
                # 假设是概率或可以clip到[0,1]的分数
                probas_base = np.clip(pred_or_score.flatten(), 0, 1)
            else:
                # 多列输出，取第二列(正类)
                probas_base = pred_or_score[:, 1]

        # meta_model 纠正 (简化逻辑，原逻辑涉及不确定性阈值)
        # 这里我们简化为直接应用meta_model（如果存在且有意义）
        # 原始代码中的不确定区域逻辑较为复杂，在此脚本中省略具体实现，
        # 默认认为base_model已经足够好，或者meta_model在整个输入上重新预测。
        # 一个合理的做法可能是将base_model不确定的样本交给meta_model,
        # 但需要知道如何定义“不确定”。这里我们采用一种替代方案：
        # 总是结合两个模型的预测（例如平均概率），但这改变了原始设计意图。
        # 为忠实于原始pipeline的设计，我们应该重现实现cascade_predict_single_model的逻辑。
        # 但由于它不是类方法，我们在此进行模拟。

        # --- 模拟 cascade_predict_single_model ---
        threshold = 0.5  # 内部决策阈值，不同于最终分类阈值
        preds = (probas_base >= threshold).astype(int)
        uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)

        if self.meta_nb is not None and uncertain_mask.sum() > 0:
            if hasattr(self.meta_nb, 'predict_proba'):
                try:
                    meta_probas = self.meta_nb.predict_proba(X_final[uncertain_mask])[:, 1]
                    # 这里可以替换原概率或综合判断，简单起见直接替换
                    probas_base[uncertain_mask] = meta_probas
                except:
                    meta_preds = self.meta_nb.predict(X_final[uncertain_mask])
                    # 如果meta模型也没有predict_proba，则用其分类结果调整
                    # 一种方式是强行设定概率为接近0或1的值
                    probas_base[uncertain_mask] = np.where(meta_preds == 1, 0.9, 0.1)
            else:
                meta_preds = self.meta_nb.predict(X_final[uncertain_mask])
                probas_base[uncertain_mask] = np.where(meta_preds == 1, 0.9, 0.1)

        # 返回最终的概率 (正类)
        print("预测完成。")
        return np.column_stack([1 - probas_base, probas_base])

    def predict(self, X_raw_df):
        """
        对原始特征DataFrame进行完整的预处理和预测，返回分类结果。
        """
        probas = self.predict_proba(X_raw_df)
        # 使用模型保存的最佳阈值进行最终分类
        predictions = (probas[:, 1] >= self.threshold).astype(int)
        return predictions


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









