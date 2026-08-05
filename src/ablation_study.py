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

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark 1 multitask learning', 'feature_to_panel_mapping.csv'))

# Load provenance mapping
feature_to_panel = {}
with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        feature_to_panel[row['Feature_Name']] = row['Panel_Name']

X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(FEATURES_PATH, TARGETS_PATH, purge_admin=True)

cognitive_panel_names = {
    "ADAS-Cog Assessment",
    "MMSE Assessment",
    "Clinical Dementia Rating (CDR)",
    "Functional Assessment Questionnaire (FAQ)",
    "Rey Auditory Verbal Learning Test (RAVLT)",
    "Trail Making Test (TMT)",
    "Boston Naming Test",
    "Clock Drawing Test",
    "Copy Drawing Test",
    "Category Fluency Test",
    "Logical Memory Test",
    "Psychometric Battery (Other)"
}

target_feature_names = {'TOTAL13', 'CDRSB', 'MMSCORE'}

biomarker_mask = np.array([feature_to_panel.get(col, '') not in cognitive_panel_names for col in feature_cols])
cognitive_mask = ~biomarker_mask
no_t0_mask = np.array([col not in target_feature_names for col in feature_cols])

subsets = {
    "Full Model (All Modalities)": np.ones(X.shape[1], dtype=bool),
    "Excluding Endpoint Totals (t=0)": no_t0_mask,
    "Pure Biomarkers ONLY": biomarker_mask,
    "Cognitive Tests ONLY": cognitive_mask
}

splits = get_kfold_splits(X.shape[0], n_splits=5, seed=42)

def solve_fista_l21_mtfl(X_sc, Y_sc, lambda_val=0.05, max_iters=5000, tol=1e-8):
    N, d = X_sc.shape
    T = Y_sc.shape[1]
    
    s_val = np.linalg.svd(X_sc, compute_uv=False)
    L = (s_val[0]**2) / N
    step = 1.0 / L
    
    W = np.zeros((d, T), dtype=np.float64)
    Z = W.copy()
    t_fista = 1.0
    
    def compute_obj(W_curr):
        loss = 0.5 * np.sum((X_sc.dot(W_curr) - Y_sc)**2) / N
        reg = lambda_val * np.sum(np.linalg.norm(W_curr, axis=1))
        return loss + reg
        
    obj_old = compute_obj(W)
    
    for it in range(max_iters):
        grad = X_sc.T.dot(X_sc.dot(Z) - Y_sc) / N
        W_temp = Z - step * grad
        
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
        
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_fista**2)) / 2.0
        Z = W_next + ((t_fista - 1.0) / t_next) * (W_next - W)
        W = W_next
        t_fista = t_next
        
    return W

print("\n=================================================================")
print("RUNNING OFFICIAL ABLATION STUDY SUITE (5-FOLD CV)")
print("=================================================================\n")

results = []

for name, mask in subsets.items():
    X_sub = X[:, mask]
    fista_r2 = {t: [] for t in target_cols}
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        x_means, x_stds = compute_observed_scaling(X_train)
        X_tr_sc, X_te_sc, _, _ = apply_scaling_and_imputation(X_train, X_test, x_means, x_stds)
        
        y_means = np.nanmean(Y_train, axis=0)
        y_stds = np.nanstd(Y_train, axis=0)
        y_stds[np.isnan(y_stds) | (y_stds == 0)] = 1.0
        
        Y_tr_imp = np.where(np.isnan(Y_train), y_means, Y_train)
        Y_tr_sc = (Y_tr_imp - y_means) / y_stds
        
        W_opt = solve_fista_l21_mtfl(X_tr_sc, Y_tr_sc, lambda_val=0.05, max_iters=5000, tol=1e-8)
        preds_sc = X_te_sc.dot(W_opt)
        preds = preds_sc * y_stds + y_means
        
        for t_i, target_name in enumerate(target_cols):
            valid_test = ~np.isnan(Y_test[:, t_i])
            if np.sum(valid_test) > 0:
                y_true = Y_test[valid_test, t_i]
                y_pred = preds[valid_test, t_i]
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                fista_r2[target_name].append(r2)
                
    row = [name, f"{X_sub.shape[1]} features"]
    for target_name in target_cols:
        r_list = fista_r2[target_name]
        r_m, r_sd = np.mean(r_list), np.std(r_list)
        ci = 1.96 * r_sd / np.sqrt(5)
        row.append(f"{r_m:.4f} [{r_m-ci:.4f}, {r_m+ci:.4f}]")
    results.append(row)

print("=================================================================")
print("OFFICIAL ABLATION STUDY RESULTS MATRIX")
print("=================================================================")
print(f"{'Feature Modality Subset':<35} | {'Features':<15} | {'ADAS13 R2 (95% CI)':<22} | {'CDRSB R2 (95% CI)':<22} | {'MMSE R2 (95% CI)':<22}")
print("-" * 125)
for row in results:
    print(f"{row[0]:<35} | {row[1]:<15} | {row[2]:<22} | {row[3]:<22} | {row[4]:<22}")
print("=================================================================\n")
