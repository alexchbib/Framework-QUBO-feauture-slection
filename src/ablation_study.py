import os
import sys
import csv
import numpy as np

sys.modules['bottleneck'] = None
sys.modules['pyarrow'] = None
import xgboost as xgb

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.preprocessing import (
    load_and_preprocess_adni_data,
    compute_observed_scaling,
    apply_scaling_and_imputation,
    get_kfold_splits
)
from src.common.fista_solver import solve_fista_l21_mtfl, select_lambda_inner_cv

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FEATURES_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_features.csv')
TARGETS_PATH = os.path.join(DATA_DIR, 'adni_longitudinal_targets.csv')
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark 1 multitask learning', 'feature_to_panel_mapping.csv'))
OUTPUT_TXT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ablation_results.txt'))

# Load provenance mapping
feature_to_panel = {}
with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        feature_to_panel[row['Feature_Name']] = row['Panel_Name']

X, Y, feature_cols, target_cols, rids = load_and_preprocess_adni_data(FEATURES_PATH, TARGETS_PATH, purge_admin=True)

# Verify mapping completeness with strict fail-safe assertion
for col in feature_cols:
    assert col in feature_to_panel, f"Feature '{col}' is missing from feature_to_panel mapping table!"

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

target_feature_names = {'TOTAL13', 'CDRSB', 'MMSCORE', 'TOTSCORE', 'ADAS11'}

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

print("\n=================================================================")
print("RUNNING OFFICIAL ABLATION STUDY SUITE (5-FOLD CV)")
print("=================================================================\n")

results = []

for name, mask in subsets.items():
    X_sub = X[:, mask]
    fista_r2 = {t: [] for t in target_cols}
    xgb_r2 = {t: [] for t in target_cols}
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # --- 1. FISTA MTFL ARM ---
        x_means, x_stds = compute_observed_scaling(X_train)
        X_tr_sc, X_te_sc, _, _ = apply_scaling_and_imputation(X_train, X_test, x_means, x_stds)
        
        y_means = np.nanmean(Y_train, axis=0)
        y_stds = np.nanstd(Y_train, axis=0)
        y_stds[np.isnan(y_stds) | (y_stds == 0)] = 1.0
        
        target_mask = ~np.isnan(Y_train)
        Y_tr_imp = np.where(np.isnan(Y_train), 0.0, Y_train)
        Y_tr_sc = (Y_tr_imp - y_means) / y_stds
        Y_tr_sc[~target_mask] = 0.0
        
        best_lambda = select_lambda_inner_cv(X_tr_sc, Y_tr_sc, target_mask)
        W_opt = solve_fista_l21_mtfl(X_tr_sc, Y_tr_sc, target_mask=target_mask.astype(float), lambda_val=best_lambda, max_iters=5000, tol=1e-8)
        preds_sc = X_te_sc.dot(W_opt)
        preds_fista = preds_sc * y_stds + y_means
        
        for t_i, target_name in enumerate(target_cols):
            valid_test = ~np.isnan(Y_test[:, t_i])
            if np.sum(valid_test) > 0:
                y_true = Y_test[valid_test, t_i]
                y_pred = preds_fista[valid_test, t_i]
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                fista_r2[target_name].append(r2)
                
        # --- 2. XGBOOST DECISION TREE ARM ---
        Y_tr_imp_raw = np.where(np.isnan(Y_train), y_means, Y_train)
        preds_xgb = np.zeros((X_test.shape[0], len(target_cols)))
        
        for t_i, target_name in enumerate(target_cols):
            model = xgb.XGBRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                n_jobs=4
            )
            model.fit(X_train, Y_tr_imp_raw[:, t_i])
            preds_xgb[:, t_i] = model.predict(X_test)
            
            valid_test = ~np.isnan(Y_test[:, t_i])
            if np.sum(valid_test) > 0:
                y_true = Y_test[valid_test, t_i]
                y_pred = preds_xgb[valid_test, t_i]
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                xgb_r2[target_name].append(r2)
                
    # Format FISTA row
    row_fista = [name, "FISTA MTFL", f"{X_sub.shape[1]} features"]
    for target_name in target_cols:
        r_list = fista_r2[target_name]
        r_m, r_sd = np.mean(r_list), np.std(r_list)
        ci = 1.96 * r_sd / np.sqrt(5)
        row_fista.append(f"{r_m:.4f} [{r_m-ci:.4f}, {r_m+ci:.4f}]")
    results.append(row_fista)

    # Format Decision Trees row
    row_xgb = [name, "Decision Trees", f"{X_sub.shape[1]} features"]
    for target_name in target_cols:
        r_list = xgb_r2[target_name]
        r_m, r_sd = np.mean(r_list), np.std(r_list)
        ci = 1.96 * r_sd / np.sqrt(5)
        row_xgb.append(f"{r_m:.4f} [{r_m-ci:.4f}, {r_m+ci:.4f}]")
    results.append(row_xgb)

print("=================================================================")
print("OFFICIAL ABLATION STUDY RESULTS MATRIX")
print("=================================================================")
print(f"{'Feature Modality Subset':<35} | {'Model':<16} | {'ADAS13 R2 (95% CI)':<22} | {'CDRSB R2 (95% CI)':<22} | {'MMSE R2 (95% CI)':<22}")
print("-" * 125)
for row in results:
    print(f"{row[0]:<35} | {row[1]:<16} | {row[3]:<22} | {row[4]:<22} | {row[5]:<22}")
print("=================================================================\n")

# Save formatted ablation results for automated report synchronizer
with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
    f.write("==== Official Ablation Study Results Matrix ====\n")
    f.write(f"{'Feature Modality Subset':<35} | {'Model':<16} | {'Features':<15} | {'ADAS13 R2 (95% CI)':<22} | {'CDRSB R2 (95% CI)':<22} | {'MMSE R2 (95% CI)':<22}\n")
    f.write("-" * 145 + "\n")
    for row in results:
        f.write(f"{row[0]:<35} | {row[1]:<16} | {row[2]:<15} | {row[3]:<22} | {row[4]:<22} | {row[5]:<22}\n")
