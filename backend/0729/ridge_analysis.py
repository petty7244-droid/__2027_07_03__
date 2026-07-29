import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'backend\0729\地區房價資料檔.csv')

print("=" * 60)
print("1. 基本資料概覽")
print("=" * 60)
print(f"資料筆數: {df.shape[0]}, 特徵數: {df.shape[1]}")
print(f"\n欄位名稱: {list(df.columns)}")
print(f"\n資料型態:\n{df.dtypes}")
print(f"\n缺失值統計:\n{df.isnull().sum()}")
print(f"\n描述性統計:\n{df.describe()}")

# 確認目標變數分布
target = '房價'
features = [c for c in df.columns if c != target]

print("\n" + "=" * 60)
print("2. 目標變數（房價）分布檢查")
print("=" * 60)
skew = df[target].skew()
kurt = df[target].kurtosis()
print(f"偏度 (Skewness): {skew:.4f}")
print(f"峰度 (Kurtosis): {kurt:.4f}")
stat, p_value = stats.normaltest(df[target])
print(f"D'Agostino-Pearson 常態檢定: stat={stat:.4f}, p-value={p_value:.4e}")

print("\n" + "=" * 60)
print("3. 線性回歸假設檢查")
print("=" * 60)

# 3a. 相關係數矩陣
corr = df.corr()
print("\n3a. 與房價的相關係數:")
target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
for f, v in target_corr.items():
    stars = "***" if abs(v) >= 0.7 else "**" if abs(v) >= 0.5 else "*" if abs(v) >= 0.3 else ""
    print(f"  {f:8s}: {v:+.4f}  {stars}")

# 3b. 特徵間的共線性 (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_for_vif = df[features].copy()
# 處理常數項
X_for_vif = X_for_vif.assign(const=1)
vif_data = pd.DataFrame()
vif_data['feature'] = X_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])]
vif_data = vif_data[vif_data['feature'] != 'const'].sort_values('VIF', ascending=False)
print("\n3b. 變異數膨脹因子 (VIF):")
for _, row in vif_data.iterrows():
    if row['VIF'] > 10:
        flag = " HIGH"
    elif row['VIF'] > 5:
        flag = " MODERATE"
    else:
        flag = " OK"
    print(f"  {row['feature']:8s}: {row['VIF']:.2f}{flag}")

# 3c. 離群值檢查 (Z-score)
z_scores = np.abs(stats.zscore(df[features]))
outliers = (z_scores > 3).sum()
outlier_rows = (z_scores > 3).any(axis=1).sum()
print(f"\n3c. 離群值檢查 (Z-score > 3):")
for i, col in enumerate(features):
    cnt = (z_scores[:, i] > 3).sum()
    if cnt > 0:
        print(f"  {col:8s}: {cnt} 個離群值")
print(f"  總計 {outlier_rows} 筆資料包含離群值")

# 3d. 線性關係檢查 (Pearson r)
print("\n3d. 與房價的線性關係強度:")
for f in features:
    r, p = stats.pearsonr(df[f], df[target])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {f:8s}: r={r:+.4f}, p={p:.4e} {sig}")

# 3e. 殘差分析（先做簡單的OLS）
print("\n" + "=" * 60)
print("4. Ridge 回歸模型建立")
print("=" * 60)

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 用交叉驗證選最佳 alpha
alphas = np.logspace(-2, 4, 50)
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=10)
ridge_cv.fit(X_train_scaled, y_train)
best_alpha = ridge_cv.alpha_
print(f"\n最佳 alpha (交叉驗證): {best_alpha:.4f}")

# 用最佳 alpha 訓練最終模型
ridge = Ridge(alpha=best_alpha, random_state=42)
ridge.fit(X_train_scaled, y_train)

y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)

print("\n--- 訓練集 ---")
print(f"R2:         {r2_score(y_train, y_train_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"MAE:        {mean_absolute_error(y_train, y_train_pred):.4f}")

print("\n--- 測試集 ---")
print(f"R2:         {r2_score(y_test, y_test_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"MAE:        {mean_absolute_error(y_test, y_test_pred):.4f}")

# 交叉驗證分數
cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=10, scoring='r2')
print(f"\n10 折交叉驗證 R2 平均: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 特徵重要性
print("\n" + "=" * 60)
print("5. 特徵重要性 (標準化後係數)")
print("=" * 60)
coef_df = pd.DataFrame({'feature': features, 'coefficient': ridge.coef_})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
for _, row in coef_df.iterrows():
    direction = "正向" if row['coefficient'] > 0 else "負向"
    print(f"  {row['feature']:8s}: {row['coefficient']:+.4f} ({direction})")

# 殘差分析
residuals = y_test - y_test_pred
print(f"\n殘差統計:")
print(f"  平均: {residuals.mean():.4f}")
print(f"  標準差: {residuals.std():.4f}")
stat2, p2 = stats.normaltest(residuals)
print(f"  常態檢定: p-value={p2:.4e}")

# ===== 圖表 =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 房價分布
axes[0, 0].hist(df[target], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'{target} 分布')
axes[0, 0].set_xlabel(target)
axes[0, 0].set_ylabel('次數')

# 2. 相關係數熱圖
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0,
            ax=axes[0, 1], cbar_kws={'shrink': 0.8})
axes[0, 1].set_title('相關係數矩陣')

# 3. 預測 vs 實際
axes[0, 2].scatter(y_test, y_test_pred, alpha=0.6)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0, 2].set_xlabel('實際房價')
axes[0, 2].set_ylabel('預測房價')
axes[0, 2].set_title(f'預測 vs 實際 (R2={r2_score(y_test, y_test_pred):.3f})')

# 4. 殘差圖
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('預測值')
axes[1, 0].set_ylabel('殘差')
axes[1, 0].set_title('殘差圖')

# 5. QQ圖
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q 圖 (殘差)')

# 6. 特徵重要性
colors = ['#e74c3c' if c < 0 else '#3498db' for c in coef_df['coefficient']]
axes[1, 2].barh(coef_df['feature'], coef_df['coefficient'], color=colors)
axes[1, 2].axvline(x=0, color='black', lw=0.5)
axes[1, 2].set_xlabel('係數')
axes[1, 2].set_title('Ridge 回歸係數')

plt.tight_layout()
plt.savefig(r'backend\0729\ridge_analysis.png', dpi=150)
print("\n图表已储存至 ridge_analysis.png")
plt.show()
