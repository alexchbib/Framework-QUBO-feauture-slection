import os
import sys
sys.modules['bottleneck'] = None
sys.modules['pyarrow'] = None
import csv
import numpy as np

import xgboost as xgb

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.preprocessing import (
    load_and_preprocess_adni_data,
    get_kfold_splits
)

# ==========================================
# CONFIGURATION
# ==========================================
TOP_N_FEATURES = 58  # Match FISTA selected feature count precisely (58 features)
N_SPLITS = 5

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
OUTPUT_DIR = os.path.dirname(__file__)

print("1. Loading Data & Applying Preprocessing Pipeline...")
X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(FEATURES_PATH, TARGETS_PATH, purge_admin=True)
print(f"Loaded {X.shape[0]} unique subjects, {X.shape[1]} clean clinical features, and {Y.shape[1]} targets.")

print(f"2. Executing {N_SPLITS}-Fold Cross-Validation for XGBoost Baseline...")

all_fold_metrics = {t: {'r2': [], 'mae': [], 'mse': []} for t in target_cols}
global_feature_importance = np.zeros(X.shape[1])

splits = get_kfold_splits(X.shape[0], n_splits=N_SPLITS, seed=42)

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    y_means = np.nanmean(Y_train, axis=0)
    Y_tr_imp = np.where(np.isnan(Y_train), y_means, Y_train)
    
    importances_fold = np.zeros(X.shape[1])
    xgb_models = []
    
    for i, target_name in enumerate(target_cols):
        model = xgb.XGBRegressor(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=4
        )
        model.fit(X_train, Y_tr_imp[:, i])
        importances_fold += model.feature_importances_
        xgb_models.append(model)
        
    importances_fold /= len(target_cols)
    global_feature_importance += importances_fold
    
    sorted_idx = np.argsort(importances_fold)[::-1]
    sel_idx = sorted_idx[:TOP_N_FEATURES]
    
    X_tr_sel = X_train[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]
    
    preds = np.zeros((X_test.shape[0], len(target_cols)))
    
    for i, target_name in enumerate(target_cols):
        eval_model = xgb.XGBRegressor(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=4
        )
        eval_model.fit(X_tr_sel, Y_tr_imp[:, i])
        preds[:, i] = eval_model.predict(X_te_sel)
        
        valid_test = ~np.isnan(Y_test[:, i])
        if np.sum(valid_test) > 0:
            y_true = Y_test[valid_test, i]
            y_pred = preds[valid_test, i]
            
            mse = np.mean((y_true - y_pred)**2)
            mae = np.mean(np.abs(y_true - y_pred))
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            all_fold_metrics[target_name]['r2'].append(r2)
            all_fold_metrics[target_name]['mae'].append(mae)
            all_fold_metrics[target_name]['mse'].append(mse)

avg_importance = global_feature_importance / N_SPLITS
top_feature_indices = np.argsort(avg_importance)[::-1][:TOP_N_FEATURES]

metrics_summary_path = os.path.join(OUTPUT_DIR, 'predictive_metrics_benchmark2.txt')
with open(metrics_summary_path, 'w') as f:
    f.write("=== XGBoost Baseline Performance Metrics (5-Fold CV on Top 59 Features) ===\n")
    for target_name in target_cols:
        r2_m, r2_std = np.mean(all_fold_metrics[target_name]['r2']), np.std(all_fold_metrics[target_name]['r2'])
        mae_m, mae_std = np.mean(all_fold_metrics[target_name]['mae']), np.std(all_fold_metrics[target_name]['mae'])
        mse_m, mse_std = np.mean(all_fold_metrics[target_name]['mse']), np.std(all_fold_metrics[target_name]['mse'])
        
        ci95_r2 = 1.96 * r2_std / np.sqrt(N_SPLITS)
        f.write(f"Target: {target_name}\n")
        f.write(f"  R2:  {r2_m:.4f} +/- {ci95_r2:.4f} (95% CI: [{r2_m - ci95_r2:.4f}, {r2_m + ci95_r2:.4f}])\n")
        f.write(f"  MAE: {mae_m:.4f} +/- {mae_std:.4f}\n")
        f.write(f"  MSE: {mse_m:.4f} +/- {mse_std:.4f}\n\n")

features_output_path = os.path.join(OUTPUT_DIR, 'selected_features_benchmark2.csv')
with open(features_output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Selected_Feature', 'XGB_Importance'])
    for idx in top_feature_indices:
        writer.writerow([feature_cols[idx], f"{avg_importance[idx]:.6f}"])

print(f"3. Results saved to '{metrics_summary_path}' and '{features_output_path}'.")
