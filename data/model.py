import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD  # For dimensionality reduction after scanning
import pickle
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels


# --- 自定义深度森林实现 ---
class EnhancedDeepForest(BaseEstimator, ClassifierMixin):
    """
    增强版深度森林实现，包含多粒度扫描和动态级联结构。

    Attributes:
        n_estimators (int): 每个子模型中的树的数量。
        max_layers (int): 级联的最大层数。
        window_sizes (list of int): 多粒度扫描使用的窗口大小列表。
        scan_n_trees (int): 扫描阶段使用的随机树嵌入的树数量。
        scan_max_depth (int): 扫描阶段使用的随机树嵌入的最大深度。
        use_scan (bool): 是否启用多粒度扫描。
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
                 window_sizes=[20, 50],
                 scan_n_trees=100,
                 scan_max_depth=3,
                 use_scan=True,
                 min_delta=0.001,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.window_sizes = window_sizes
        self.scan_n_trees = scan_n_trees
        self.scan_max_depth = scan_max_depth
        self.use_scan = use_scan
        self.min_delta = min_delta
        self.random_state = random_state
        # 移除在 __init__ 中初始化的非参数属性
        # self.scan_layers_ = []
        # self.cascade_layers_ = []
        # self.classes_ = None

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

    def _scan_fit_transform(self, X, y=None, is_fitting=True):
        """
        应用多粒度扫描到输入数据X。
        此处简化处理为对整个特征空间做一次随机映射后再分解。
        更严格的做法是对序列型数据切片后分别处理。
        """
        if not self.use_scan:
            return X

        # 在 fit 时初始化或在 transform 时使用已有的扫描器
        if is_fitting:
            self.scan_transformers_ = []
            transformed_parts = [X]
            for size in self.window_sizes:
                try:
                    # 使用随机树嵌入作为替代方案来进行扫描操作
                    rte = RandomTreesEmbedding(
                        n_estimators=self.scan_n_trees,
                        max_depth=self.scan_max_depth,
                        random_state=self.random_state)

                    # 因为此处不是真正的图像/序列数据，所以采用全局统计代替滑窗
                    # 示例方式之一是取固定长度片段然后聚合
                    # 这里假设所有样本具有相同维度 d=X.shape[1]
                    d = X.shape[1]
                    step = max(1, d // size)
                    slices = []
                    processed_slices = []
                    for start in range(0, d - step * size + 1, step):
                        end = start + step * size
                        slice_data = X[:, start:end]

                        # 如果不能整除则跳过末尾不足的部分
                        if slice_data.shape[1] != step * size:
                            continue

                        # Reshape to batch_size x num_patches x patch_dim
                        reshaped_slice = slice_data.reshape(slice_data.shape[0], size, -1).mean(axis=2)
                        processed_slices.append(reshaped_slice)

                    if processed_slices:
                        # 合并所有处理过的片段
                        combined_slices = np.hstack(processed_slices)
                        # Apply embedding
                        embedded = rte.fit_transform(combined_slices)
                        self.scan_transformers_.append(rte)  # 保存transformer

                        # Reduce dimensions via SVD
                        svd_reducer = TruncatedSVD(n_components=min(embedded.shape[1], 50),
                                                   random_state=self.random_state)  # 限制组件数量
                        reduced_features = svd_reducer.fit_transform(embedded.toarray())
                        self.scan_svd_reducers_.append(svd_reducer)  # 保存SVD reducer
                        transformed_parts.append(reduced_features)
                    else:
                        print(f"[Warning] No valid slices created for window size {size}. Skipping.")

                except Exception as e:
                    print(f"[Warning] Failed during multi-grained scanning with window size {size}: {e}")
                    # 确保即使失败也添加占位符，保持索引一致
                    self.scan_transformers_.append(None)
                    self.scan_svd_reducers_.append(None)
                    continue
        else:  # Transforming new data
            check_is_fitted(self, ['scan_transformers_', 'scan_svd_reducers_'])
            transformed_parts = [X]
            transformer_idx = 0
            for size in self.window_sizes:
                try:
                    rte = self.scan_transformers_[transformer_idx]
                    svd_reducer = self.scan_svd_reducers_[transformer_idx]

                    if rte is None or svd_reducer is None:
                        print(f"[Warning] Skipping window size {size} as it failed during fitting.")
                        transformer_idx += 1
                        continue

                    d = X.shape[1]
                    step = max(1, d // size)
                    processed_slices = []
                    for start in range(0, d - step * size + 1, step):
                        end = start + step * size
                        slice_data = X[:, start:end]

                        if slice_data.shape[1] != step * size:
                            continue

                        reshaped_slice = slice_data.reshape(slice_data.shape[0], size, -1).mean(axis=2)
                        processed_slices.append(reshaped_slice)

                    if processed_slices:
                        combined_slices = np.hstack(processed_slices)
                        embedded = rte.transform(combined_slices)  # 使用transform
                        reduced_features = svd_reducer.transform(embedded.toarray())  # 使用transform
                        transformed_parts.append(reduced_features)

                except Exception as e:
                    print(f"[Warning] Error transforming with window size {size}: {e}")
                finally:
                    transformer_idx += 1

        result = np.hstack(transformed_parts)
        return result.astype(np.float32)

    def _evaluate_layer_gain(self, prev_X_with_preds, current_X_with_preds, y_true):
        """
        评估当前层带来的性能增益。
        """
        # 使用简单的验证集而不是交叉验证以提高速度
        # 或者使用更轻量级的模型
        try:
            scores_prev = cross_val_score(GaussianNB(), prev_X_with_preds, y_true, cv=3, scoring='accuracy',
                                          n_jobs=1)  # 限制n_jobs
            score_prev_avg = np.mean(scores_prev)

            scores_curr = cross_val_score(GaussianNB(), current_X_with_preds, y_true, cv=3, scoring='accuracy',
                                          n_jobs=1)
            score_curr_avg = np.mean(scores_curr)

            gain = score_curr_avg - score_prev_avg
            return gain >= self.min_delta, gain
        except:
            # 如果评估失败，默认认为有增益或第一层必须加入
            return True, 0.0

    def fit(self, X, y):
        """
        训练深度森林模型。
        """
        # 在 fit 开始时初始化所有实例变量
        self.classes_ = unique_labels(y)
        self.scan_layers_ = []
        self.cascade_layers_ = []
        self.scan_transformers_ = []
        self.scan_svd_reducers_ = []

        original_X_dtype = X.dtype

        # Step 1: Multi-Grained Scanning
        print("Starting Multi-Grained Scanning...")
        scanned_X_train = self._scan_fit_transform(X.copy(), y, is_fitting=True) if self.use_scan else X.copy()
        current_input_for_training = scanned_X_train.copy()

        print(f"Initial shape after scanning: {current_input_for_training.shape}")

        # Initialize variables for cascade structure learning loop
        last_performance = float('-inf')
        best_performance_so_far = float('-inf')
        no_improvement_counter = 0
        patience_limit = 2  # Number of layers without improvement before stopping early

        layer_index = 0
        while layer_index < self.max_layers:
            print(f"\n--- Training Cascade Layer {layer_index + 1} ---")

            estimators_in_this_layer = {}
            temp_predictions_on_current_batch = []

            # Train each estimator type on the current input features
            for name, clf_class_instance in self._create_base_estimators():
                # 确保克隆的分类器具有正确的参数
                params = clf_class_instance.get_params()
                # 特别注意 n_features 参数，它应该基于当前输入
                cloned_clf = type(clf_class_instance)(**params)
                cloned_clf.fit(current_input_for_training, y)
                estimators_in_this_layer[name] = cloned_clf

                # Predict probabilities on training set for next level inputs
                pred_probs = cloned_clf.predict_proba(current_input_for_training)
                temp_predictions_on_current_batch.append(pred_probs)

            # Concatenate predictions from all models trained at this layer
            concatenated_new_features_from_this_layer = np.hstack(temp_predictions_on_current_batch)

            # Combine previous features with newly generated ones for evaluation & next iteration
            combined_features_evaluating_this_level = np.hstack([
                current_input_for_training,
                concatenated_new_features_from_this_layer
            ])

            # Evaluate performance gain compared to just using old features alone
            does_add_value_here, delta_acc = self._evaluate_layer_gain(
                current_input_for_training,
                combined_features_evaluating_this_level,
                y
            )

            print(f"Evaluation Gain Check -> Improvement? : {does_add_value_here}, Delta Accuracy={delta_acc:.4f}")

            # Update internal state only when beneficial
            if does_add_value_here or len(self.cascade_layers_) == 0:  # Always keep first layer
                self.cascade_layers_.append(estimators_in_this_layer)
                current_input_for_training = combined_features_evaluating_this_level
                best_performance_so_far = max(best_performance_so_far, delta_acc)
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                print("[Info] No significant performance increase detected; considering pruning...")

            # Early stopping condition based on lack of progress over several iterations
            if no_improvement_counter >= patience_limit:
                print("\nEarly Stopping Triggered due to stagnant performance!")
                break

            layer_index += 1

        print(
            f"\nFinal number of effective cascade layers built: {len(self.cascade_layers_)} / Max allowed was {self.max_layers}\n")
        return self

    def predict_proba(self, X):
        """
        获取测试集上的类别概率分布。
        """
        check_is_fitted(self, ['cascade_layers_', 'classes_'])

        # Perform initial transformation through scanner stage
        processed_X = self._scan_fit_transform(X.copy(), is_fitting=False) if self.use_scan else X.copy()

        # Propagate samples forward through cascaded levels
        for idx, layer_models_dict in enumerate(self.cascade_layers_, 1):
            individual_model_outputs = []
            for model_name_key, fitted_model_obj in layer_models_dict.items():
                probs_output_by_one_model = fitted_model_obj.predict_proba(processed_X)
                individual_model_outputs.append(probs_output_by_one_model)

            # Merge outputs into one expanded feature vector per sample
            merged_probabilities_as_features = np.hstack(individual_model_outputs)

            # Append these new synthetic features onto existing representation
            processed_X = np.hstack([processed_X, merged_probabilities_as_features])

        # Final prediction uses average across final layer's classifiers
        final_prediction_pool = []
        for _, final_model_ref in self.cascade_layers_[-1].items():
            predicted_distribution = final_model_ref.predict_proba(processed_X)
            final_prediction_pool.append(predicted_distribution)

        mean_of_final_distributions = np.mean(final_prediction_pool, axis=0)
        return mean_of_final_distributions

    def predict(self, X):
        """
        根据最高置信度返回对应的类别索引。
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# --- 包装器保持一致 ---
class CascadeForestWrapper(ClassifierMixin):
    def __init__(self, cascade_forest, **kwargs):
        self.cascade_forest = cascade_forest
        # 不在 __init__ 中设置 self.classes_

    def fit(self, X, y):
        self.cascade_forest.fit(X, y)
        # fit 后再设置 classes_
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
    """辅助函数用于保存对象至磁盘文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath):
    """辅助函数加载已存在的pickle格式文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 数据预处理流程保持不变...
    print("Loading data...")
    # 假设 clean.csv 存在且路径正确
    try:
        data = pd.read_csv('./clean.csv').drop(columns=['company_id'])
    except FileNotFoundError:
        print("Error: File './clean.csv' not found. Please ensure the file exists.")
        exit(1)

    print("Encoding categorical variables...")
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop(columns=['target'])
    y = data['target']

    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("Applying SMOTEENN resampling...")
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

    print("Selecting features...")
    selector = SelectFromModel(xgb.XGBClassifier(n_estimators=100, random_state=42))
    X_selected = selector.fit_transform(X_resampled, y_resampled)

    save_object(selector, './feature_selector_xgboost_cemmdan.pkl')
    save_object(imputer, './imputer.pkl')
    save_object(scaler, './scaler.pkl')

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.1, random_state=42)
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # 替换为我们新的增强版本模型
    print("Initializing enhanced deep forest model...")
    # 为了快速测试，禁用扫描并减少层数和估计器数量
    base_enhanced_df = EnhancedDeepForest(n_estimators=50, max_layers=3, window_sizes=[10, 25],
                                          scan_n_trees=30, scan_max_depth=2, use_scan=False,  # 禁用扫描以避免复杂性
                                          min_delta=0.0005, random_state=42)

    enhanced_df = CascadeForestWrapper(base_enhanced_df)

    # 简化AdaBoost配置
    adaboost = AdaBoostClassifier(n_estimators=50, algorithm='SAMME', random_state=42)

    print("Creating stacking classifier...")
    stacking_model = StackingClassifier(
        estimators=[
            ('enhanced_df', enhanced_df),
            ('adaboost', adaboost)
        ],
        final_estimator=GaussianNB(),
        cv=3,  # 减少CV折数
        n_jobs=1  # 限制并行度以避免潜在问题
    )

    print("Training stacking model...")
    try:
        stacking_model.fit(X_train, y_train)
        print("Model training completed successfully.")

        save_object(stacking_model, '../model_pipeline_xgboost_cemmdan.pkl')
        print("Model saved to '../model_pipeline_xgboost_cemmdan.pkl'")

        print("Evaluating model...")
        y_pred_test = stacking_model.predict(X_test)
        y_proba_test = stacking_model.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred_test, zero_division=0)
        auc = roc_auc_score(y_test, y_proba_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred_test)

        # 假设这是评分公式
        final_score = 30 * recall + 50 * auc + 20 * precision

        print("--- Model Evaluation on Test Set ---")
        print(f"Accuracy:  {accuracy:.6f}")
        print(f"Recall:    {recall:.6f}")
        print(f"AUC:       {auc:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Final Score: {final_score:.6f}")
        print("------------------------------------")
    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")
        import traceback

        traceback.print_exc()





