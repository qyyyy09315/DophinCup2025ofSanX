import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as ImbPipeline  # Avoid naming conflict
# Note: If imblearn is not installed, run 'pip install imbalanced-learn'
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Import XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    print("Warning: xgboost library not found. Please run 'pip install xgboost'.")
    XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def save_object(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def cascade_predict_single_model(base_model, meta_model, X, threshold=0.5):
    """
    Modified cascade_predict for a single base model.
    Fixed error handling when meta_model is None.
    """
    try:
        probas_base = base_model.predict_proba(X)[:, 1]
    except Exception:
        # Fallback if predict_proba is not available
        probas_base = base_model.predict(X).astype(float)

    preds = (probas_base >= threshold).astype(int)

    # Simplified uncertainty judgment: assume uncertain when probability is near 0.5
    # This logic can be refined based on specific needs.
    uncertain_mask = (probas_base > 0.3) & (probas_base < 0.7)  # Thresholds can be adjusted

    # Only use meta_model if it's not None and there are uncertain samples
    if meta_model is not None and uncertain_mask.sum() > 0:
        meta_preds = meta_model.predict(X[uncertain_mask])
        preds[uncertain_mask] = meta_preds

    return preds, probas_base


# Define a block with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        # Main path
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Shortcut connection path (if input/output dimensions differ)
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.relu1(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.linear2(out)
        # Residual connection
        out += identity
        out = self.relu2(out)
        if self.dropout2:
            out = self.dropout2(out)

        return out


class DeepFeatureSelector(nn.Module):
    """Deeper fully connected network with residual connections."""

    def __init__(self, input_dim):
        super().__init__()
        # Build network using residual blocks
        self.block1 = ResidualBlock(input_dim, 1024, 0.4)
        self.block2 = ResidualBlock(1024, 512, 0.3)
        self.block3 = ResidualBlock(512, 256, 0.3)
        self.block4 = ResidualBlock(256, 128, 0.2)

        # Output layer
        self.output_layer = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_dnn_feature_selector_torch(X, y, input_dim, epochs=200, batch_size=128, lr=1e-3, device="cpu"):
    # Ensure model is on the specified device
    model = DeepFeatureSelector(input_dim).to(device)
    criterion = nn.BCELoss()
    # Consider using weight_decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Move data to the specified device
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for xb, yb in loader:
            # Data is already on the device, no need to move again
            # xb, yb = xb.to(device), yb.to(device)
            yb = yb.unsqueeze(1)  # Adjust label shape
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        # Optional: Print average loss
        # if (epoch + 1) % 50 == 0:
        #     avg_loss = total_loss / num_batches
        #     print(f"Epoch {epoch+1}/{epochs}, Avg Loss={avg_loss:.4f}")

    # Extract first layer weights to calculate feature importance
    # Note: The first layer is now a ResidualBlock; we access its internal linear1 layer
    first_linear_layer = None
    if hasattr(model, 'block1') and hasattr(model.block1, 'linear1'):
        first_linear_layer = model.block1.linear1
    else:
        # If structure changes, try to find the first nn.Linear layer
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear_layer = module
                break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.detach().cpu().numpy()  # shape=(1024, input_dim)
        feature_importance = np.mean(np.abs(weights), axis=0)
    else:
        # If linear layer cannot be found, return zero importance (shouldn't happen theoretically)
        print("Warning: Could not find the first linear layer to compute feature importance.")
        feature_importance = np.zeros(input_dim)

    return feature_importance


def fbeta_threshold(y_true, probas, beta=1.0):
    """
    Find the optimal threshold that maximizes F-beta score.

    Args:
        y_true (array-like): True binary labels.
        probas (array-like): Predicted probabilities for the positive class.
        beta (float): Weight of recall in the combined score. Default is 1.0 (F1).

    Returns:
        tuple: Best threshold value and corresponding maximum F-beta score.
    """
    thresholds = np.linspace(0, 1, 1001)[1:-1]  # Exclude 0 and 1 for numerical stability
    best_t, max_fbeta = 0.5, -1

    for t in thresholds:
        preds = (probas >= t).astype(int)

        # Calculate metrics avoiding division by zero
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # All predictions belong to one class
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # All predicted negative
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # All predicted positive
        else:
            # Irregular matrix, pad missing entries
            padded_cm = np.pad(cm, ((0, 2 - cm.shape[0]), (0, 2 - cm.shape[1])), mode='constant')
            tn, fp, fn, tp = padded_cm.ravel()

        precision_val = tp / (tp + fp + 1e-15)
        recall_val = tp / (tp + fn + 1e-15)

        numerator = (1 + beta ** 2) * (precision_val * recall_val)
        denominator = ((beta ** 2 * precision_val) + recall_val + 1e-15)

        if denominator > 0:
            fbeta = numerator / denominator
        else:
            fbeta = 0.0

        if fbeta > max_fbeta:
            max_fbeta, best_t = fbeta, t

    return best_t, max_fbeta


if __name__ == "__main__":
    if not XGB_AVAILABLE:
        print("Error: Missing required dependency 'xgboost'. Exiting program.")
        exit(1)

    print("=" * 70)
    print("Starting Training:")
    print("DNN Feature Selection (with ResNet) -> ")
    print("Combined Sampling (ADASYN + ClusterCentroids) -> ")
    print("Cost-Sensitive XGBoost -> Cascade Correction (GaussianNB)")
    print("(Utilizing GPU acceleration via PyTorch/XGBoost if available)")
    print("=" * 70)

    # --- Configuration ---
    data_path = "./clean.csv"
    # Check if data file exists
    if not os.path.exists(data_path):
        # Create example data for demonstration purposes. Replace with real data path in practice.
        print(f"Warning: Data file '{data_path}' not found. Generating sample data...")
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.rand(1000) * 100,
            'category_A': ['Y'] * 300 + ['N'] * 700,
            'target': ([1] * 150 + [0] * 150) + ([1] * 50 + [0] * 650)  # Create imbalance
        })
        sample_data = pd.get_dummies(sample_data, drop_first=True)
        sample_data.to_csv(data_path, index=False)
        print(f"Sample data generated and saved to {data_path}")

    test_size = 0.10
    random_state = 42
    beta_for_threshold = 2.0  # Optimize for F2-score during threshold selection
    # Key modification: Determine device, prefer CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computation device: {device}")

    # --- Cost-Sensitive XGBoost Hyperparameters ---
    # Calculate scale_pos_weight for cost-sensitive learning
    # We'll fit the scaler/resampler first to get accurate counts later.
    # For initial setup, let's define placeholders or use rough estimates.
    # It will be updated after resampling step.
    # Initial placeholder values
    pos_count_initial = (pd.read_csv(data_path)['target'] == 1).sum()
    neg_count_initial = (pd.read_csv(data_path)['target'] == 0).sum()
    scale_pos_weight_initial = neg_count_initial / pos_count_initial if pos_count_initial > 0 else 1

    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # Used for early stopping
        'random_state': random_state,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight_initial,  # Placeholder for cost sensitivity
        # Enable GPU (if available and XGBoost was built with GPU support)
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
    }
    print(f"Initial XGBoost params (including scale_pos_weight estimate): {xgb_params}")

    # --- 1. Load and Preprocess Data ---
    data = pd.read_csv(data_path)
    # Drop company_id column (if present)
    if 'company_id' in data.columns:
        data = data.drop(columns=["company_id"])
    # Perform One-Hot Encoding
    data = pd.get_dummies(data, drop_first=True)
    if "target" not in data.columns:
        raise KeyError("Target column 'target' not found in data.")

    X_all = data.drop(columns=["target"]).values
    y_all = data["target"].values
    print(f"Data loaded: X={X_all.shape}, y={y_all.shape}, Positive={y_all.sum()}, Negative={(y_all == 0).sum()}")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    save_object(imputer, "./imputer.pkl")
    save_object(scaler, "./scaler.pkl")

    # --- 2. DNN Feature Weighting ---
    print("Training Torch DNN (with residual connections) to obtain feature weights ...")
    # Key modification: Pass device to training function
    feature_importance = train_dnn_feature_selector_torch(
        X_scaled, y_all,
        input_dim=X_scaled.shape[1],
        device=device  # Use specified device
    )
    ranked_idx = np.argsort(-feature_importance)
    # Select features based on ranking (using all here, but could select top N)
    X_selected = X_scaled[:, ranked_idx]
    print(f"Torch DNN feature weighting completed. Total features used={X_selected.shape[1]}")

    save_object(ranked_idx, "./dnn_feature_ranking.pkl")

    # --- 3. Split Train/Validation Sets ---
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_selected, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    print(f"Train set: {X_train_full.shape}, Validation set: {X_holdout.shape}")

    # --- 4. Combined Sampling Strategy (ADASYN + ClusterCentroids) ---
    print("Performing combined sampling: ADASYN followed by ClusterCentroids undersampling ...")
    # Define the combined sampling pipeline
    sampler_pipeline = ImbPipeline([
        ('over_sampler', ADASYN(random_state=random_state)),
        ('under_sampler', ClusterCentroids(random_state=random_state))
    ])

    X_resampled, y_resampled = sampler_pipeline.fit_resample(X_train_full, y_train_full)
    print(
        f"After combined sampling: X={X_resampled.shape}, Positives={y_resampled.sum()}, Negatives={(y_resampled == 0).sum()}")

    # Update XGBoost parameters with actual resampled ratio for precise cost-sensitivity
    unique, counts = np.unique(y_resampled, return_counts=True)
    count_dict = dict(zip(unique, counts))
    neg_count_actual = count_dict.get(0, 0)
    pos_count_actual = count_dict.get(1, 0)
    scale_pos_weight_actual = neg_count_actual / pos_count_actual if pos_count_actual > 0 else 1
    xgb_params['scale_pos_weight'] = scale_pos_weight_actual
    print(f"Updated XGBoost scale_pos_weight based on resampled data: {scale_pos_weight_actual}")

    # --- 5. Train Base Model (Cost-Sensitive XGBoost) ---
    print("Training cost-sensitive XGBoost base model ...")
    # Prepare XGBoost datasets for potential early stopping
    dtrain_full = xgb.DMatrix(X_resampled, label=y_resampled)
    dval = xgb.DMatrix(X_holdout, label=y_holdout)

    evals_result = {}
    # Train XGBoost model with updated parameters
    xgb_model = xgb.train(
        xgb_params,
        dtrain=dtrain_full,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,  # Optional: enable early stopping
        verbose_eval=False,  # Set to True to see training progress
        evals_result=evals_result
    )


    # Wrap model to provide scikit-learn like interface
    class WrappedXGBModel:
        def __init__(self, booster):
            self.booster = booster

        def predict_proba(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            # Return 2D array [[prob_0, prob_1], ...]
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            probs = self.booster.predict(dmatrix)
            return (probs > 0.5).astype(int)


    wrapped_xgb_model = WrappedXGBModel(xgb_model)

    save_object(wrapped_xgb_model, "./base_model_xgboost_cost_sensitive.pkl")
    print("Base model trained: Cost-Sensitive XGBoost")

    # --- 6. Train Cascade Corrector ---
    print("Identifying misclassified samples by base model, training cascade corrector...")
    # Use new prediction method, pass None as meta_model initially
    _, train_probas = cascade_predict_single_model(wrapped_xgb_model, None, X_resampled, threshold=0.5)
    base_train_preds = (train_probas >= 0.5).astype(int)

    misclassified_mask = (base_train_preds != y_resampled)
    X_hard, y_hard = X_resampled[misclassified_mask], y_resampled[misclassified_mask]

    if len(X_hard) > 0 and len(np.unique(y_hard)) > 1:  # Ensure hard examples have both classes
        meta_clf = GaussianNB()
        meta_clf.fit(X_hard, y_hard)
        print(f"Cascade corrector trained successfully. Hard examples count={len(X_hard)}")
    else:
        # If no misclassified samples or only one class in hard set, train on full resampled data
        meta_clf = GaussianNB()
        meta_clf.fit(X_resampled, y_resampled)
        print("No sufficient misclassified samples found. Corrector trained on full resampled data.")

    # --- 7. Threshold Selection (Optimized for F-beta Score) ---
    # Use new prediction function
    _, holdout_probas = cascade_predict_single_model(wrapped_xgb_model, meta_clf, X_holdout, threshold=0.5)
    best_thresh, max_fbeta = fbeta_threshold(y_holdout, holdout_probas, beta=beta_for_threshold)
    print(
        f"F{beta_for_threshold} optimized threshold search complete. Best threshold={best_thresh:.4f}, Max F{beta_for_threshold}={max_fbeta:.4f}")

    y_pred_holdout = (holdout_probas >= best_thresh).astype(int)
    recall = recall_score(y_holdout, y_pred_holdout, zero_division=0)
    auc = roc_auc_score(y_holdout, holdout_probas)
    precision = precision_score(y_holdout, y_pred_holdout, zero_division=0)
    f1 = f1_score(y_holdout, y_pred_holdout, zero_division=0)
    # Custom scoring formula (Note: Original comment coefficients did not sum to 100%, preserved as-is)
    final_score = 30 * recall + 50 * auc + 20 * precision

    print(f"Evaluation Metrics (Threshold={best_thresh:.4f}):")
    print(f"Recall={recall:.5f}, AUC={auc:.5f}, Precision={precision:.5f}, F1={f1:.5f}, FinalScore={final_score:.5f}")

    # --- 8. Save Complete Model Pipeline ---
    model_dict = {
        "imputer": imputer,
        "scaler": scaler,
        "dnn_feature_ranking": ranked_idx,
        "base_models": wrapped_xgb_model,  # Single model
        "base_types": ["XGBoost"],  # List of types
        "meta_nb": meta_clf,
        "threshold": float(best_thresh),
    }
    save_object(model_dict, "./model_pipeline_cascade_dnn_torch_improved.pkl")
    print("Complete model pipeline saved to ./model_pipeline_cascade_dnn_torch_improved.pkl")
    print("=" * 70)

