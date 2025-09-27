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
        self.scan_layers_ = []  # 存储扫描后的表示转换器
        self.cascade_layers_ = []  # 存储每层的基分类器组
        self.classes_ = None
        self._fitted_feature_count = None  # Track feature count seen during fit

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

    def _scan_fit_transform(self, X, y=None, is_fitting=False):
        """
        应用多粒度扫描到输入数据X。
        此处简化处理为对整个特征空间做一次随机映射后再分解。
        更严格的做法是对序列型数据切片后分别处理。
        """
        if not self.use_scan:
            return X

        transformed_parts = [X]

        # Only perform actual scanning logic during fitting or if we know the structure from fitting
        if is_fitting or hasattr(self, '_scan_slices_info'):
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

                    if is_fitting:
                        self._scan_slices_info = []  # Store info needed for transform

                    slice_idx = 0
                    slices = []
                    for start in range(0, d - step * size + 1, step):
                        end = start + step * size
                        slice_data = X[:, start:end]

                        # 如果不能整除则跳过末尾不足的部分
                        if slice_data.shape[1] != step * size:
                            continue

                        # Reshape to batch_size x num_patches x patch_dim
                        reshaped_slice = slice_data.reshape(slice_data.shape[0], size, -1).mean(axis=2)

                        # Apply embedding and reduce dimensions via SVD
                        embedded = rte.fit_transform(reshaped_slice) if is_fitting else \
                            rte.transform(reshaped_slice)

                        # During fitting, create and store SVD reducer
                        if is_fitting:
                            svd_reducer = TruncatedSVD(n_components=min(embedded.shape) - 1,
                                                       random_state=self.random_state)
                            reduced_features = svd_reducer.fit_transform(embedded.toarray())
                            self._scan_slices_info.append(
                                (start, end, step, size, rte, svd_reducer))  # Save transformers too
                        else:  # During prediction, use stored reducers
                            svd_reducer = self._scan_slices_info[slice_idx][5]
                            reduced_features = svd_reducer.transform(embedded.toarray())

                        transformed_parts.append(reduced_features)
                        slice_idx += 1

                except Exception as e:
                    print(f"[Warning] Failed during multi-grained scanning with window size {size}: {e}")
                    continue

            result = np.hstack(transformed_parts)
            if is_fitting:
                self._fitted_feature_count = result.shape[1]  # Record expected feature count
            return result.astype(np.float32)

        else:  # Not fitting and no scan info saved yet, just pass through original features
            # This handles cases where an unfitted model might be used directly for predict_proba/predict
            # But typically should not happen if called correctly within pipeline flow
            return X

    def _evaluate_layer_gain(self, prev_X_with_preds, current_X_with_preds, y_true):
        """
        评估当前层带来的性能增益。
        """
        scores_prev = cross_val_score(GaussianNB(), prev_X_with_preds, y_true, cv=3, scoring='accuracy')
        score_prev_avg = np.mean(scores_prev)

        scores_curr = cross_val_score(GaussianNB(), current_X_with_preds, y_true, cv=3, scoring='accuracy')
        score_curr_avg = np.mean(scores_curr)

        gain = score_curr_avg - score_prev_avg
        return gain >= self.min_delta, gain

    def fit(self, X, y):
        """
        训练深度森林模型。
        """
        self.classes_ = np.unique(y)
        original_X_dtype = X.dtype

        # Step 1: Multi-Grained Scanning
        print("Starting Multi-Grained Scanning...")
        scanned_X_train = self._scan_fit_transform(X.copy(), is_fitting=True) if self.use_scan else X.copy()
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
                cloned_clf = type(clf_class_instance)(**clf_class_instance.get_params())
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
        if not hasattr(self, 'cascade_layers_') or len(self.cascade_layers_) == 0:
            raise RuntimeError("You must call 'fit' prior to making predictions!")

        # Perform initial transformation through scanner stage
        processed_X = self._scan_fit_transform(X.copy()) if self.use_scan else X.copy()

        # Ensure consistent number of features passed between layers by padding/truncating if necessary
        # However better approach would have been ensuring consistency throughout design phase.
        # Here's quick fix attempt inside predict path assuming fitted model expects certain dim now:
        if hasattr(self, '_fitted_feature_count'):  # If we tracked it during fit
            required_num_features_at_first_layer = getattr(self, "_first_layer_expected_features", None)
            if required_num_features_at_first_layer is not None and processed_X.shape[
                1] != required_num_features_at_first_layer:
                diff = required_num_features_at_first_layer - processed_X.shape[1]
                if diff > 0:
                    print(f"[Warning] Padding input features ({processed_X.shape}) to match training time dims.")
                    pad_width = ((0, 0), (0, diff))
                    processed_X = np.pad(processed_X, pad_width, 'constant')
                elif diff < 0:
                    print(f"[Warning] Truncating input features ({processed_X.shape}) to match training time dims.")
                    processed_X = processed_X[:, :required_num_features_at_first_layer]

        # Propagate samples forward through cascaded levels
        for idx, layer_models_dict in enumerate(self.cascade_layers_, 1):

            individual_model_outputs = []
            for model_name_key, fitted_model_obj in layer_models_dict.items():

                # Verify that the input matches what the specific model inside cascade expects
                # Since our construction ensures they see identical augmented views up until their own point,
                # mismatch can still occur especially post-scanning changes unless strictly controlled

                try:
                    probs_output_by_one_model = fitted_model_obj.predict_proba(processed_X)

                except ValueError as ve:
                    # Catch case like "RandomForestClassifier is expecting NNN features"
                    exp_feats_str = str(ve).split('expecting ')[-1].split(' features')[0]
                    exp_feats = int(exp_feats_str) if exp_feats_str.isdigit() else processed_X.shape[
                                                                                       1] + 10  # fallback guess

                    act_feats = processed_X.shape[1]
                    print(
                        f"[Debug] Mismatch error caught in layer {idx}-{model_name_key}. Expected feats:{exp_feats}, Got:{act_feats}")

                    # Attempt correction strategy similar above but more targeted locally here
                    if abs(act_feats - exp_feats) > 0:
                        diff = exp_feats - act_feats
                        if diff > 0:
                            print("[FixAttempt] Zero-padding extra space temporarily.")
                            pad_wid = ((0, 0), (0, diff))
                            padded_temp = np.pad(processed_X, pad_wid, 'constant')
                            probs_output_by_one_model = fitted_model_obj.predict_proba(padded_temp)
                        else:
                            print("[FixAttempt] Cropping excess columns temporarily.")
                            cropped_temp = processed_X[:, :exp_feats]
                            probs_output_by_one_model = fitted_model_obj.predict_proba(cropped_temp)

                    else:
                        raise ve  # Re-raise if nothing obvious wrong found

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
    data = pd.read_csv('./clean.csv').drop(columns=['company_id'])

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

    # 替换为我们新的增强版本模型
    print("Initializing enhanced deep forest model...")
    base_enhanced_df = EnhancedDeepForest(n_estimators=80, max_layers=4, window_sizes=[10, 25],
                                          scan_n_trees=50, scan_max_depth=2, use_scan=False,
                                          min_delta=0.0005, random_state=42)

    enhanced_df = CascadeForestWrapper(base_enhanced_df)

    adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=42)

    print("Creating stacking classifier...")
    stacking_model = StackingClassifier(
        estimators=[
            ('enhanced_df', enhanced_df),
            ('adaboost', adaboost)
        ],
        final_estimator=GaussianNB(),
        n_jobs=-1
    )

    print("Training stacking model...")
    stacking_model.fit(X_train, y_train)

    save_object(stacking_model, '../model_pipeline_xgboost_cemmdan.pkl')

    print("Evaluating model...")
    y_pred_test = stacking_model.predict(X_test)
    y_proba_test = stacking_model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test)

    final_score = 30 * recall + 50 * auc + 20 * precision

    print("--- Model Evaluation on Test Set ---")
    print(f"Recall:    {recall:.6f}")
    print(f"AUC:       {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Final Score: {final_score:.6f}")
    print("------------------------------------")
