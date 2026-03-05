import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 设置绘图风格：SCI 兼容
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False,
    "font.family": "sans-serif",
    "svg.fonttype": "none"
})

# ----------------------------
# 1. 加载并预处理 UCI 信用卡数据
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, skiprows=1)
df.rename(columns={'default payment next month': 'default'}, inplace=True)

# 原始关键特征
base_features = ['LIMIT_BAL', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']

# 构造金融意义明确的衍生特征（模拟多粒度行为）
df['BILL_PAY_RATIO'] = df['BILL_AMT1'] / (df['PAY_AMT1'] + 1)  # 避免除零
df['HIGH_DELAY'] = (df['PAY_0'] >= 2).astype(int)              # 近期严重逾期标志

# 最终特征列表（对应论文中的“多粒度特征融合”）
feature_cols = base_features + ['BILL_PAY_RATIO', 'HIGH_DELAY']
X = df[feature_cols]
y = df['default']

# 划分训练/测试集（保持分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 2. 训练 LightGBM 模型（作为 X-ECML 的可解释代理）
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

# 评估性能
y_pred_proba = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")

# ----------------------------
# 3. SHAP 解释（使用 TreeExplainer，高效且精确）
# ----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# 注意：LightGBM 二分类返回的是 log-odds，shap_values 是针对正类的

# ----------------------------
# 4. 全局解释：Summary Plot（蜜蜂图）
# ----------------------------
plt.figure(figsize=(6, 4))
shap.summary_plot(
    shap_values,
    X_test.values,
    feature_names=feature_cols,
    show=False,
    plot_size=None,
    color=plt.get_cmap("coolwarm")
)
plt.title("")
plt.tight_layout()
plt.savefig("shap_summary.svg", format="svg", bbox_inches="tight")
plt.close()

# ----------------------------
# 5. 局部解释：选取一个高风险 True Positive 样本
# ----------------------------
y_pred = (y_pred_proba > 0.5).astype(int)
tp_mask = (y_test == 1) & (y_pred == 1)
if tp_mask.sum() == 0:
    # 若无TP，选预测概率最高的真实违约样本
    default_indices = np.where(y_test == 1)[0]
    target_idx = default_indices[np.argmax(y_pred_proba[default_indices])]
else:
    tp_indices = np.where(tp_mask)[0]
    target_idx = tp_indices[np.argmax(y_pred_proba[tp_indices])]

# 转为 DataFrame 行便于标注
sample_features = X_test.iloc[target_idx]

# 生成 Force Plot（使用 matplotlib 后端）
plt.figure(figsize=(8, 2))
shap.plots.force(
    base_value=explainer.expected_value,
    shap_values=shap_values[target_idx],
    features=sample_features.values,
    feature_names=feature_cols,
    matplotlib=True,
    show=False,
    contribution_threshold=0.05
)
plt.title("")
plt.tight_layout()
plt.savefig("shap_force_example.svg", format="svg", bbox_inches="tight")
plt.close()

print("✅ 模型训练完成，SHAP 图已保存为 SVG：")
print("   - shap_summary.svg")
print("   - shap_force_example.svg")