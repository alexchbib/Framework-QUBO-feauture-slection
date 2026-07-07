import os
import csv
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MANUAL_LAMBDA = 0.5  # Regularization strength for L2,1 norm
LEARNING_RATE = 0.01
N_ITERS = 1000

# File paths
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

print(f"Loaded {X.shape[0]} subjects, {X.shape[1]} features, and {Y.shape[1]} targets.")

# Train/Test Split (80/20 manually)
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split_idx = int(X.shape[0] * 0.8)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# ==========================================
# DATA PREPROCESSING
# ==========================================
print("2. Imputing Missing Data (Mean Imputation)...")
x_means = np.nanmean(X_train, axis=0)
x_means[np.isnan(x_means)] = 0.0 # fallback for all-nan columns
X_train_imp = np.where(np.isnan(X_train), x_means, X_train)
X_test_imp = np.where(np.isnan(X_test), x_means, X_test)

y_means = np.nanmean(Y_train, axis=0)
Y_train_imp = np.where(np.isnan(Y_train), y_means, Y_train)
Y_test_imp = np.where(np.isnan(Y_test), y_means, Y_test)

print("3. Standardizing Features...")
x_stds = np.std(X_train_imp, axis=0)
x_stds[x_stds == 0] = 1.0
X_train_scaled = (X_train_imp - x_means) / x_stds
X_test_scaled = (X_test_imp - x_means) / x_stds

# ==========================================
# MULTI-TASK FEATURE LEARNING (L2,1-Norm Proximal Gradient)
# ==========================================
print(f"4. Executing Multi-Task Feature Selection (L2,1-Norm, Lambda={MANUAL_LAMBDA})...")

N, d = X_train_scaled.shape
T = Y_train_imp.shape[1]
W = np.zeros((d, T))

alpha = LEARNING_RATE / N
for _ in range(N_ITERS):
    # Gradient of 0.5 * ||XW - Y||_F^2
    grad = X_train_scaled.T.dot(X_train_scaled.dot(W) - Y_train_imp) / N
    W_temp = W - alpha * grad
    
    # Block Soft Thresholding for L2,1 penalty
    norms = np.linalg.norm(W_temp, axis=1)
    threshold = alpha * MANUAL_LAMBDA
    
    # Apply soft thresholding
    mask = norms > threshold
    scaling = (1 - threshold / norms[mask])
    W[mask] = W_temp[mask] * scaling[:, np.newaxis]
    W[~mask] = 0.0

feature_norms = np.linalg.norm(W, axis=1)
selected_indices = np.where(feature_norms > 0)[0]
selected_features = [feature_cols[i] for i in selected_indices]

print(f"   -> Features Selected: {len(selected_features)} out of {X.shape[1]}")

feature_output_path = os.path.join(OUTPUT_DIR, 'selected_features_benchmark1.csv')
with open(feature_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Selected_Feature'])
    for feat in selected_features:
        writer.writerow([feat])
print(f"   -> Selected features saved to {feature_output_path}")

# ==========================================
# SECONDARY EVALUATION: DOWNSTREAM MODELS
# ==========================================
print("5. Evaluating Predictive Power on Isolated Panel...")

if len(selected_features) == 0:
    print("No features were selected. The regularization penalty may be too high. Exiting.")
    exit()

X_train_sel = X_train_scaled[:, selected_indices]
X_test_sel = X_test_scaled[:, selected_indices]

# Closed-form Ridge Regression for Evaluation
ridge_lambda = 1.0
I = np.eye(len(selected_indices))
W_ridge = np.linalg.inv(X_train_sel.T.dot(X_train_sel) + ridge_lambda * I).dot(X_train_sel.T).dot(Y_train_imp)

preds_ridge = X_test_sel.dot(W_ridge)
preds_lasso = X_test_scaled.dot(W)

models = {
    'Multi-Task Linear (L2,1 Lasso)': preds_lasso,
    'Ridge Regression (on selected panel)': preds_ridge
}

metrics_output_path = os.path.join(OUTPUT_DIR, 'predictive_metrics_benchmark1.txt')
with open(metrics_output_path, 'w') as f:
    f.write("==== Benchmark 1: Predictive Performance ====\n")
    f.write("Metric Details: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared (R2)\n\n")

    for model_name, preds in models.items():
        f.write(f"--- {model_name} ---\n")
        for i, target_name in enumerate(target_cols):
            y_t = Y_test_imp[:, i]
            p_t = preds[:, i]
            
            mse = np.mean((y_t - p_t)**2)
            mae = np.mean(np.abs(y_t - p_t))
            ss_res = np.sum((y_t - p_t)**2)
            ss_tot = np.sum((y_t - np.mean(y_t))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            f.write(f"Target: {target_name} | R2: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}\n")
        f.write("\n")

print(f"   -> Metrics evaluated and saved to {metrics_output_path}")
print("Benchmark 1 Execution Complete!")
