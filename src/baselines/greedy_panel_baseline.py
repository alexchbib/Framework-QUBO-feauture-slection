import os
import sys
import csv
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.preprocessing import (
    load_and_preprocess_adni_data,
    compute_observed_scaling,
    apply_scaling_and_imputation,
    get_kfold_splits
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
B1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark 1 multitask learning'))

FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
MAPPING_PATH = os.path.join(B1_DIR, 'feature_to_panel_mapping.csv')
COSTS_PATH = os.path.join(B1_DIR, 'panel_costs.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'greedy_baseline_metrics.txt')

print("1. Loading Data & Clinical Panel Provenance...")
X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(FEATURES_PATH, TARGETS_PATH, purge_admin=True)

feature_to_panel = {}
with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        feature_to_panel[row['Feature_Name']] = row['Panel_Name']

panel_costs = {}
with open(COSTS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        panel_costs[row['Panel_Name']] = float(row['Cost_USD'])

# Group feature indices by panel
panel_to_indices = {}
for i, feat in enumerate(feature_cols):
    panel = feature_to_panel.get(feat, "Demographics & Medical History")
    if panel not in panel_to_indices:
        panel_to_indices[panel] = []
    panel_to_indices[panel].append(i)

print(f"Loaded {len(panel_to_indices)} active clinical panels.")

# Identify imaging & biomarker expensive panels to prune
prunable_panels = [p for p in panel_to_indices.keys() if panel_costs.get(p, 0) >= 500]

print("2. Executing Greedy Panel Pruning Heuristic (Classical Benchmark)...")

def fit_eval_panel_subset(panel_subset):
    indices = []
    for p in panel_subset:
        indices.extend(panel_to_indices.get(p, []))
    indices = sorted(list(set(indices)))
    
    if not indices:
        return 0.0, 0.0
        
    splits = get_kfold_splits(X.shape[0], n_splits=5, seed=42)
    r2_list = []
    
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        x_means, x_stds = compute_observed_scaling(X_train)
        X_tr_sc, X_te_sc, _, _ = apply_scaling_and_imputation(X_train, X_test, x_means, x_stds)
        
        y_means = np.nanmean(Y_train, axis=0)
        y_stds = np.nanstd(Y_train, axis=0)
        y_stds[np.isnan(y_stds) | (y_stds == 0)] = 1.0
        
        X_tr_sub = X_tr_sc[:, indices]
        X_te_sub = X_te_sc[:, indices]
        
        fold_r2s = []
        for t_i in range(Y.shape[1]):
            valid_tr = ~np.isnan(Y_train[:, t_i])
            valid_te = ~np.isnan(Y_test[:, t_i])
            
            if np.sum(valid_tr) > 5 and np.sum(valid_te) > 0:
                X_tr_t = X_tr_sub[valid_tr]
                y_tr_t = (Y_train[valid_tr, t_i] - y_means[t_i]) / y_stds[t_i]
                
                ridge_alpha = 10.0
                I = np.eye(X_tr_t.shape[1])
                w_t = np.linalg.solve(X_tr_t.T.dot(X_tr_t) + ridge_alpha * I, X_tr_t.T.dot(y_tr_t))
                
                y_pred = (X_te_sub.dot(w_t)) * y_stds[t_i] + y_means[t_i]
                y_true = Y_test[valid_te, t_i]
                
                ss_res = np.sum((y_true - y_pred[valid_te])**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                fold_r2s.append(r2)
                
        r2_list.append(np.mean(fold_r2s))
        
    total_cost = sum(panel_costs.get(p, 0) for p in panel_subset)
    return np.mean(r2_list), total_cost

# Greedy elimination trace
current_panels = list(panel_to_indices.keys())
trace = []

initial_r2, initial_cost = fit_eval_panel_subset(current_panels)
trace.append((len(current_panels), initial_cost, initial_r2, "Full Panel Set"))

while len(current_panels) > 3:
    candidates = [p for p in current_panels if p in prunable_panels]
    if not candidates:
        break
        
    best_p_to_remove = None
    best_score = -999.0
    
    for p in candidates:
        test_subset = [x for x in current_panels if x != p]
        r2_sub, cost_sub = fit_eval_panel_subset(test_subset)
        # Efficiency metric: preserve max R2 per dollar saved
        score = r2_sub
        if score > best_score:
            best_score = score
            best_p_to_remove = p
            
    if best_p_to_remove is None:
        break
        
    current_panels.remove(best_p_to_remove)
    prunable_panels.remove(best_p_to_remove)
    r2_curr, cost_curr = fit_eval_panel_subset(current_panels)
    trace.append((len(current_panels), cost_curr, r2_curr, f"Removed {best_p_to_remove}"))

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write("==== Classical Greedy Panel Pruning Baseline ====\n")
    f.write("Protocol: Iterative Cost-Aware Greedy Elimination (5-Fold CV)\n\n")
    f.write("Step | Active Panels | Billed Cost ($) | Mean R2 Score | Action\n")
    f.write("-" * 75 + "\n")
    for i, (n_p, cost, r2_val, action) in enumerate(trace):
        f.write(f"{i:4d} | {n_p:13d} | ${cost:13,.2f} | {r2_val:13.4f} | {action}\n")

print(f"Greedy Panel Baseline Complete! Metrics saved to {OUTPUT_PATH}")
