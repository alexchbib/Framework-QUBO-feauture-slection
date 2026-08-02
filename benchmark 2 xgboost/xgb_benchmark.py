import os
import csv
import numpy as np

import xgboost as xgb

# ==========================================
# CONFIGURATION
# ==========================================
TOP_N_FEATURES = 447  # Match Benchmark 1 feature count precisely

DATA_DIR = '../data'
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
OUTPUT_DIR = '.'

print("1. Loading Data...")

def load_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data

x_header, x_raw = load_csv(FEATURES_PATH)
y_header, y_raw = load_csv(TARGETS_PATH)

rid_idx_x = x_header.index('RID')
site_idx_x = x_header.index('SITEID')
keep_idx_x = [i for i in range(len(x_header)) if i not in (rid_idx_x, site_idx_x)]
feature_cols = [x_header[i] for i in keep_idx_x]

rid_idx_y = y_header.index('RID')
site_idx_y = y_header.index('SITEID')
keep_idx_y = [i for i in range(len(y_header)) if i not in (rid_idx_y, site_idx_y)]
target_cols = [y_header[i] for i in keep_idx_y]

x_raw.sort(key=lambda r: int(r[rid_idx_x]))
y_raw.sort(key=lambda r: int(r[rid_idx_y]))

def safe_float(val):
    if val in ('NA', '', None):
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

X = np.array([[safe_float(row[i]) for i in keep_idx_x] for row in x_raw])
Y = np.array([[safe_float(row[i]) for i in keep_idx_y] for row in y_raw])

print(f"Loaded {X.shape[0]} unique subjects, {X.shape[1]} features, and {Y.shape[1]} targets.")

# Train/Test Split (80/20 manually with fixed random seed)
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split_idx = int(X.shape[0] * 0.8)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# ==========================================
# DATA PREPROCESSING (X Imputation for XGBoost)
# ==========================================
print("2. Preprocessing Features (Mean Imputation on X)...")
x_means = np.nanmean(X_train, axis=0)
x_means[np.isnan(x_means)] = 0.0
X_train_imp = np.where(np.isnan(X_train), x_means, X_train)
X_test_imp = np.where(np.isnan(X_test), x_means, X_test)

# Mean impute Y_train for training fit per estimator
y_means = np.nanmean(Y_train, axis=0)
Y_train_imp = np.where(np.isnan(Y_train), y_means, Y_train)

# ==========================================
# MULTI-TASK FEATURE LEARNING (XGBoost)
# ==========================================
print("3. Executing Multi-Task Feature Selection (XGBoost)...")

importances = np.zeros(X.shape[1])
xgb_models = []

for i, target_name in enumerate(target_cols):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4
    )
    model.fit(X_train_imp, Y_train_imp[:, i])
    importances += model.feature_importances_
    xgb_models.append(model)

importances /= len(target_cols)

# Rank features by importance
sorted_indices = np.argsort(importances)[::-1]
selected_indices = sorted_indices[:TOP_N_FEATURES]
selected_features = [feature_cols[i] for i in selected_indices]

print(f"   -> Top {TOP_N_FEATURES} Features Selected out of {X.shape[1]}")

feature_output_path = os.path.join(OUTPUT_DIR, 'selected_features_benchmark2.csv')
with open(feature_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Selected_Feature', 'ImportanceScore'])
    for idx in selected_indices:
        writer.writerow([feature_cols[idx], importances[idx]])
print(f"   -> Selected features saved to {feature_output_path}")

# ==========================================
# SECONDARY EVALUATION: DOWNSTREAM MODELS
# ==========================================
print("4. Evaluating Predictive Power on Isolated Panel (No Target Imputation Leak)...")

X_train_sel = X_train_imp[:, selected_indices]
X_test_sel = X_test_imp[:, selected_indices]

preds = np.zeros((X_test_sel.shape[0], len(target_cols)))

for i, target_name in enumerate(target_cols):
    eval_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4
    )
    eval_model.fit(X_train_sel, Y_train_imp[:, i])
    preds[:, i] = eval_model.predict(X_test_sel)

metrics_output_path = os.path.join(OUTPUT_DIR, 'predictive_metrics_benchmark2.txt')
with open(metrics_output_path, 'w') as f:
    f.write("==== Benchmark 2: Predictive Performance (XGBoost) ====\n")
    f.write("Metric Details: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared (R2)\n")
    f.write("Note: Evaluated strictly on observed non-missing test target outcomes.\n\n")

    f.write(f"--- XGBoost Regression (on top {TOP_N_FEATURES} selected features) ---\n")
    for i, target_name in enumerate(target_cols):
        valid_test_mask = ~np.isnan(Y_test[:, i])
        y_t = Y_test[valid_test_mask, i]
        p_t = preds[valid_test_mask, i]
        
        mse = np.mean((y_t - p_t)**2)
        mae = np.mean(np.abs(y_t - p_t))
        ss_res = np.sum((y_t - p_t)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        f.write(f"Target: {target_name} | R2: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}\n")
    f.write("\n")

print(f"   -> Metrics evaluated and saved to {metrics_output_path}")
print("Benchmark 2 Execution Complete!")
