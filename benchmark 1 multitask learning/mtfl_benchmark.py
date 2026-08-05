import os
import sys
import csv
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.preprocessing import (
    load_and_preprocess_adni_data,
    compute_observed_scaling,
    apply_scaling_and_imputation,
    get_kfold_splits
)

# ==========================================
# CONFIGURATION
# ==========================================
LAMBDA_VAL = 0.05
MAX_ITERS = 5000
TOLERANCE = 1e-8
N_SPLITS = 5

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
OUTPUT_DIR = os.path.dirname(__file__)

print("1. Loading Data & Applying Preprocessing Pipeline...")
X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(FEATURES_PATH, TARGETS_PATH, purge_admin=True)
print(f"Loaded {X.shape[0]} unique subjects, {X.shape[1]} clean clinical features, and {Y.shape[1]} targets.")

def solve_fista_l21_mtfl(X_sc, Y_sc, target_mask=None, lambda_val=0.05, max_iters=5000, tol=1e-8):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for L2,1 Group Lasso.
    Features target observation masking to compute gradients strictly on observed targets.
    """
    N, d = X_sc.shape
    T = Y_sc.shape[1]
    
    if target_mask is None:
        target_mask = np.ones_like(Y_sc)
    
    # Compute exact largest singular value / spectral norm
    s_val = np.linalg.svd(X_sc, compute_uv=False)
    L = (s_val[0]**2) / N
    step = 1.0 / L
    
    W = np.zeros((d, T), dtype=np.float64)
    Z = W.copy()
    t_fista = 1.0
    
    def compute_obj(W_curr):
        diff = (X_sc.dot(W_curr) - Y_sc) * target_mask
        loss = 0.5 * np.sum(diff**2) / N
        reg = lambda_val * np.sum(np.linalg.norm(W_curr, axis=1))
        return loss + reg
        
    obj_old = compute_obj(W)
    
    for it in range(max_iters):
        # Gradient of smooth loss term w.r.t Z
        diff_z = (X_sc.dot(Z) - Y_sc) * target_mask
        grad = X_sc.T.dot(diff_z) / N
        W_temp = Z - step * grad
        
        # Block Soft-Thresholding for L2,1 group norm
        norms = np.linalg.norm(W_temp, axis=1)
        thresh = step * lambda_val
        
        mask = norms > thresh
        scaling = np.zeros_like(norms)
        scaling[mask] = (1.0 - thresh / norms[mask])
        W_next = W_temp * scaling[:, np.newaxis]
        
        obj_new = compute_obj(W_next)
        rel_change = abs(obj_old - obj_new) / (obj_old + 1e-12)
        
        if rel_change < tol and it > 20:
            W = W_next
            break
            
        obj_old = obj_new
        
        # FISTA Nesterov acceleration update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_fista**2)) / 2.0
        Z = W_next + ((t_fista - 1.0) / t_next) * (W_next - W)
        W = W_next
        t_fista = t_next
        
    return W

print(f"2. Executing {N_SPLITS}-Fold Cross-Validation with FISTA...")

all_fold_metrics = {model: {t: {'r2': [], 'mae': [], 'mse': []} for t in target_cols} 
                    for model in ['Multi-Task L2,1 Lasso (FISTA)', 'Ridge Refit (Selected Panel)']}

selected_feature_counts = []
global_feature_importance = np.zeros(X.shape[1])

splits = get_kfold_splits(X.shape[0], n_splits=N_SPLITS, seed=42)

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    # Compute scaling on OBSERVED values before imputation
    x_means, x_stds = compute_observed_scaling(X_train)
    X_tr_sc, X_te_sc, X_tr_imp, X_te_imp = apply_scaling_and_imputation(X_train, X_test, x_means, x_stds)
    
    # Target outcome scaling on observed entries
    y_means = np.nanmean(Y_train, axis=0)
    y_stds = np.nanstd(Y_train, axis=0)
    y_stds[np.isnan(y_stds) | (y_stds == 0)] = 1.0
    
    target_mask = ~np.isnan(Y_train)
    Y_tr_imp = np.where(np.isnan(Y_train), 0.0, Y_train)
    Y_tr_sc = (Y_tr_imp - y_means) / y_stds
    Y_tr_sc[~target_mask] = 0.0
    
    W_opt = solve_fista_l21_mtfl(X_tr_sc, Y_tr_sc, target_mask=target_mask.astype(float), lambda_val=LAMBDA_VAL, max_iters=MAX_ITERS, tol=TOLERANCE)
    
    feat_norms = np.linalg.norm(W_opt, axis=1)
    global_feature_importance += feat_norms
    
    sel_idx = np.where(feat_norms > 1e-5)[0]
    if len(sel_idx) == 0:
        sel_idx = np.argsort(feat_norms)[::-1][:50]
        
    selected_feature_counts.append(len(sel_idx))
    
    # Predictions
    preds_lasso_sc = X_te_sc.dot(W_opt)
    preds_lasso = preds_lasso_sc * y_stds + y_means
    
    # Ridge refit on selected panel
    X_tr_sel = X_tr_sc[:, sel_idx]
    X_te_sel = X_te_sc[:, sel_idx]
    preds_ridge = np.zeros((X_te_sel.shape[0], Y.shape[1]))
    
    for t_i in range(Y.shape[1]):
        valid_train = ~np.isnan(Y_train[:, t_i])
        if np.sum(valid_train) > 5:
            X_tr_t = X_tr_sel[valid_train]
            y_tr_t = (Y_train[valid_train, t_i] - y_means[t_i]) / y_stds[t_i]
            
            ridge_alpha = 10.0
            I = np.eye(X_tr_t.shape[1])
            w_t = np.linalg.solve(X_tr_t.T.dot(X_tr_t) + ridge_alpha * I, X_tr_t.T.dot(y_tr_t))
            preds_ridge[:, t_i] = (X_te_sel.dot(w_t)) * y_stds[t_i] + y_means[t_i]
        else:
            preds_ridge[:, t_i] = y_means[t_i]
            
    models_preds = {
        'Multi-Task L2,1 Lasso (FISTA)': preds_lasso,
        'Ridge Refit (Selected Panel)': preds_ridge
    }
    
    for model_name, preds in models_preds.items():
        for t_i, target_name in enumerate(target_cols):
            valid_test = ~np.isnan(Y_test[:, t_i])
            if np.sum(valid_test) > 0:
                y_true = Y_test[valid_test, t_i]
                y_pred = preds[valid_test, t_i]
                
                mse = np.mean((y_true - y_pred)**2)
                mae = np.mean(np.abs(y_true - y_pred))
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                all_fold_metrics[model_name][target_name]['r2'].append(r2)
                all_fold_metrics[model_name][target_name]['mae'].append(mae)
                all_fold_metrics[model_name][target_name]['mse'].append(mse)

# Save top selected features
avg_importance = global_feature_importance / N_SPLITS
top_indices = np.argsort(avg_importance)[::-1]
mean_selected_count = int(np.mean(selected_feature_counts))
selected_indices_final = [idx for idx in top_indices if avg_importance[idx] > 0][:mean_selected_count]

feature_output_path = os.path.join(OUTPUT_DIR, 'selected_features_benchmark1.csv')
with open(feature_output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Selected_Feature', 'L21_Norm_Importance'])
    for idx in selected_indices_final:
        writer.writerow([feature_cols[idx], avg_importance[idx]])

metrics_output_path = os.path.join(OUTPUT_DIR, 'predictive_metrics_benchmark1.txt')
with open(metrics_output_path, 'w', encoding='utf-8') as f:
    f.write("==== Benchmark 1: Multi-Task L2,1-Norm (FISTA Converged) ====\n")
    f.write(f"Validation Protocol: {N_SPLITS}-Fold Cross-Validation\n")
    f.write(f"Mean Features Selected: {mean_selected_count} out of {X.shape[1]}\n\n")
    
    for model_name, targets_dict in all_fold_metrics.items():
        f.write(f"--- {model_name} ---\n")
        for target_name, m_dict in targets_dict.items():
            r2_m, r2_std = np.mean(m_dict['r2']), np.std(m_dict['r2'])
            mae_m, mae_std = np.mean(m_dict['mae']), np.std(m_dict['mae'])
            mse_m, mse_std = np.mean(m_dict['mse']), np.std(m_dict['mse'])
            
            ci95_r2 = 1.96 * r2_std / np.sqrt(len(m_dict['r2']))
            ci95_mae = 1.96 * mae_std / np.sqrt(len(m_dict['mae']))
            
            f.write(f"Target: {target_name}\n")
            f.write(f"  - R2:  {r2_m:.4f} +/- {ci95_r2:.4f} (95% CI: [{r2_m - ci95_r2:.4f}, {r2_m + ci95_r2:.4f}])\n")
            f.write(f"  - MAE: {mae_m:.4f} +/- {ci95_mae:.4f}\n")
            f.write(f"  - MSE: {mse_m:.4f}\n\n")

print("Benchmark 1 Execution Complete!")
print(f"Metrics saved to {metrics_output_path}")
