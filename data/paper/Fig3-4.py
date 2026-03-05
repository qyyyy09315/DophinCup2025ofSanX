import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ==============================
# 全局绘图设置：严格遵循 SCI 排版规范
# ==============================
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",  # 开源无衬线字体，兼容 LaTeX 和 Word
    "svg.fonttype": "none",        # SVG 中文字为文本而非路径
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,    # 减少边缘空白
    "axes.linewidth": 0.6,         # 细轴线
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# ----------------------------
# 1. 加载数据（使用 CSV 镜像，避免 xlrd 依赖）
# ----------------------------
print("Loading UCI Credit Card dataset from CSV mirror...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, skiprows=1)

df.rename(columns={'PAY_1': 'PAY_0', 'default.payment.next.month': 'default'}, inplace=True)

# 原始关键特征
base_features = ['LIMIT_BAL', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']

# 构造金融意义明确的衍生特征
df['BILL_PAY_RATIO'] = df['BILL_AMT1'] / (df['PAY_AMT1'] + 1)
df['HIGH_DELAY'] = (df['PAY_0'] >= 2).astype(int)

feature_cols = base_features + ['BILL_PAY_RATIO', 'HIGH_DELAY']
X = df[feature_cols]
y = df['default']

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 2. 训练 LightGBM 模型
# ----------------------------
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'verbose': -1,
    'seed': 42
}

print("Training LightGBM model (as proxy for X-ECML)...")
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# 评估
y_pred_proba = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")

# ----------------------------
# 3. SHAP 解释
# ----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# ----------------------------
# 4. 全局解释：Summary Plot（单栏图，8.4 cm 宽）
# ----------------------------
# 美化特征名称（用于绘图）
feature_names_pretty = [
    "Credit Limit",
    "Recent Payment Status",
    "Latest Bill Amount",
    "Last Payment Amount",
    "Bill/Payment Ratio",
    "Severe Delay Flag"
]

plt.figure(figsize=(3.3, 2.8))  # 单栏宽度：3.3 英寸 ≈ 8.4 cm
shap.summary_plot(
    shap_values,
    X_test.values,
    feature_names=feature_names_pretty,
    show=False,
    plot_size=None,
    color=plt.get_cmap("RdBu_r"),  # Red-Blue reversed: high risk = red
    max_display=6
)
plt.xlabel("SHAP value (impact on model output)", fontsize=9)
plt.tight_layout(pad=0.3)
plt.savefig("shap_summary.svg", format="svg")
plt.close()

# ----------------------------
# 5. 局部解释：Force Plot（双栏图，适合横向展示）
# ----------------------------
y_pred = (y_pred_proba > 0.5).astype(int)
tp_mask = (y_test == 1) & (y_pred == 1)
if tp_mask.sum() == 0:
    default_indices = np.where(y_test == 1)[0]
    target_idx = default_indices[np.argmax(y_pred_proba[default_indices])]
else:
    tp_indices = np.where(tp_mask)[0]
    target_idx = tp_indices[np.argmax(y_pred_proba[tp_indices])]

# Force plot：宽度设为双栏（6.8 英寸），高度压缩
plt.figure(figsize=(6.8, 1.6))  # 双栏宽度：6.8 英寸 ≈ 17.3 cm
shap.plots.force(
    base_value=explainer.expected_value,
    shap_values=shap_values[target_idx],
    features=X_test.iloc[target_idx].values,
    feature_names=feature_names_pretty,
    matplotlib=True,
    show=False,
    contribution_threshold=0.05
)
plt.tight_layout(pad=0.1)
plt.savefig("shap_force_example.svg", format="svg")
plt.close()

print("✅ SHAP figures saved in publication-ready SVG format:")
print("   • shap_summary.svg     (single-column width)")
print("   • shap_force_example.svg (double-column width)")