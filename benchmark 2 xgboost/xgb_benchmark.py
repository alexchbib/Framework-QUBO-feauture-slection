import os
import sys
sys.modules['bottleneck'] = None
sys.modules['pyarrow'] = None
import csv
import re
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
N_SPLITS = 5
N_INNER_SPLITS = 3

# Hyper-parameter grid searched by inner CV, mirroring Benchmark 1's lambda search.
PARAM_GRID = [(30, 3), (100, 3), (300, 3), (100, 4), (300, 4)]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
OUTPUT_DIR = os.path.dirname(__file__)

# The feature budget is defined by Benchmark 1 -- read it, never restate it.
BM1_METRICS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'benchmark 1 multitask learning',
    'predictive_metrics_benchmark1.txt'))


def read_bm1_feature_budget(path):
    assert os.path.exists(path), (
        f"Benchmark 1 metrics not found at {path}. "
        "Run Benchmark 1 before Benchmark 2 (run_pipeline.py --bm1).")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Mean Features Selected:\s+(\d+)\s+out of\s+(\d+)", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"Could not parse 'Mean Features Selected' from {path}.")


TOP_N_FEATURES, BM1_POOL_SIZE = read_bm1_feature_budget(BM1_METRICS_PATH)
print(f"Matched feature budget from Benchmark 1: {TOP_N_FEATURES} features "
      f"(out of {BM1_POOL_SIZE} candidates)")

print("1. Loading Data & Applying Preprocessing Pipeline...")
X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(
    FEATURES_PATH, TARGETS_PATH, purge_admin=True)
print(f"Loaded {X.shape[0]} unique subjects, {X.shape[1]} clean clinical features, "
      f"and {Y.shape[1]} targets.")

assert X.shape[1] == BM1_POOL_SIZE, (
    f"Candidate pool mismatch: Benchmark 1 used {BM1_POOL_SIZE} features, "
    f"Benchmark 2 loaded {X.shape[1]}. The two benchmarks are not comparable.")


def make_model(n_estimators, max_depth, seed):
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=0
    )


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def select_params_inner_cv(X_tr, Y_tr, seed):
    """Choose (n_estimators, max_depth) by an inner CV inside the training
    fold only. Mirrors select_lambda_inner_cv in Benchmark 1 -- test patients are
    never used for tuning."""
    inner_splits = get_kfold_splits(X_tr.shape[0], n_splits=N_INNER_SPLITS, seed=42)
    best_params, best_score = PARAM_GRID[0], -np.inf

    for n_estimators, max_depth in PARAM_GRID:
        split_scores = []
        for in_tr_idx, in_val_idx in inner_splits:
            target_scores = []
            for i in range(Y_tr.shape[1]):
                v_tr = ~np.isnan(Y_tr[in_tr_idx, i])
                v_val = ~np.isnan(Y_tr[in_val_idx, i])
                if np.sum(v_tr) < 10 or np.sum(v_val) < 2:
                    continue
                model = make_model(n_estimators, max_depth, seed)
                model.fit(X_tr[in_tr_idx][v_tr], Y_tr[in_tr_idx][v_tr, i])
                preds = model.predict(X_tr[in_val_idx])
                target_scores.append(r2_score(Y_tr[in_val_idx][v_val, i], preds[v_val]))
            if target_scores:
                split_scores.append(np.mean(target_scores))
        mean_score = np.mean(split_scores) if split_scores else -np.inf
        if mean_score > best_score:
            best_score, best_params = mean_score, (n_estimators, max_depth)

    return best_params


print(f"2. Executing {N_SPLITS}-Fold Cross-Validation for XGBoost Baseline...")

all_fold_metrics = {t: {'r2': [], 'mae': [], 'mse': []} for t in target_cols}
global_feature_importance = np.zeros(X.shape[1])
chosen_params = []

splits = get_kfold_splits(X.shape[0], n_splits=N_SPLITS, seed=42)

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    n_estimators, max_depth = select_params_inner_cv(X_train, Y_train, 42 + fold)
    chosen_params.append((n_estimators, max_depth))
    print(f"   fold {fold + 1}: inner CV selected "
          f"n_estimators={n_estimators}, max_depth={max_depth}")

    # ---- Stage 1: rank features by importance -------------------------------
    importances_fold = np.zeros(X.shape[1])
    for i, target_name in enumerate(target_cols):
        # Train only on patients whose outcome was actually measured.
        valid_train = ~np.isnan(Y_train[:, i])
        model = make_model(n_estimators, max_depth, 42 + fold)
        model.fit(X_train[valid_train], Y_train[valid_train, i])
        importances_fold += model.feature_importances_

    importances_fold /= len(target_cols)
    global_feature_importance += importances_fold

    sorted_idx = np.argsort(importances_fold)[::-1]
    sel_idx = sorted_idx[:TOP_N_FEATURES]

    # ---- Stage 2: refit on the selected panel and score ---------------------
    X_tr_sel = X_train[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]
    preds = np.zeros((X_test.shape[0], len(target_cols)))

    for i, target_name in enumerate(target_cols):
        valid_train = ~np.isnan(Y_train[:, i])
        eval_model = make_model(n_estimators, max_depth, 42 + fold)
        eval_model.fit(X_tr_sel[valid_train], Y_train[valid_train, i])
        preds[:, i] = eval_model.predict(X_te_sel)

        valid_test = ~np.isnan(Y_test[:, i])
        if np.sum(valid_test) > 0:
            y_true = Y_test[valid_test, i]
            y_pred = preds[valid_test, i]

            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = r2_score(y_true, y_pred)

            all_fold_metrics[target_name]['r2'].append(r2)
            all_fold_metrics[target_name]['mae'].append(mae)
            all_fold_metrics[target_name]['mse'].append(mse)

avg_importance = global_feature_importance / N_SPLITS
top_feature_indices = np.argsort(avg_importance)[::-1][:TOP_N_FEATURES]

metrics_summary_path = os.path.join(OUTPUT_DIR, 'predictive_metrics_benchmark2.txt')
with open(metrics_summary_path, 'w') as f:
    f.write(f"=== XGBoost Baseline Performance Metrics "
            f"(5-Fold CV on Top {TOP_N_FEATURES} Features) ===\n")
    f.write(f"Feature budget inherited from Benchmark 1: {TOP_N_FEATURES} "
            f"of {BM1_POOL_SIZE} candidates\n")
    f.write(f"Hyper-parameters per fold (inner {N_INNER_SPLITS}-fold CV, "
            f"n_estimators/max_depth): "
            f"{', '.join(f'{a}/{b}' for a, b in chosen_params)}\n")
    f.write("Missing training outcomes are masked out, not mean-imputed.\n\n")
    for target_name in target_cols:
        r2_m = np.mean(all_fold_metrics[target_name]['r2'])
        r2_std = np.std(all_fold_metrics[target_name]['r2'])
        mae_m = np.mean(all_fold_metrics[target_name]['mae'])
        mae_std = np.std(all_fold_metrics[target_name]['mae'])
        mse_m = np.mean(all_fold_metrics[target_name]['mse'])
        mse_std = np.std(all_fold_metrics[target_name]['mse'])

        ci95_r2 = 1.96 * r2_std / np.sqrt(N_SPLITS)
        f.write(f"Target: {target_name}\n")
        f.write(f"  R2:  {r2_m:.4f} +/- {ci95_r2:.4f} "
                f"(95% CI: [{r2_m - ci95_r2:.4f}, {r2_m + ci95_r2:.4f}])\n")
        f.write(f"  MAE: {mae_m:.4f} +/- {mae_std:.4f}\n")
        f.write(f"  MSE: {mse_m:.4f} +/- {mse_std:.4f}\n\n")

features_output_path = os.path.join(OUTPUT_DIR, 'selected_features_benchmark2.csv')
with open(features_output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Selected_Feature', 'XGB_Importance'])
    for idx in top_feature_indices:
        writer.writerow([feature_cols[idx], f"{avg_importance[idx]:.6f}"])

print(f"3. Results saved to '{metrics_summary_path}' and '{features_output_path}'.")
