import os
import csv
import re
import numpy as np

# Administrative metadata regex to purge before modeling
ADMIN_FEATURE_REGEX = re.compile(
    r'^(RID|SITEID(\..*)?|IMAGEUID(\..*)?|STATUS(\..*)?|VERSION(\..*)?|LONIUID(\..*)?|SOURCE(\..*)?|RAWQC|qc_flag|FSVER|FIELD_STRENGTH|MANUFACTURER|TRACER.*|SCANDATE.*|PROCESSDATE.*|BATCH|KIT|STDS|USERDATE|update_stamp|HAS_QC_ERROR|DD_CRF_VERSION|VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE|ID(\..*)?)$',
    re.IGNORECASE
)

# Known categorical mappings
CATEGORICAL_MAPPINGS = {
    'PTGENDER': {'Female': 0.0, 'Male': 1.0, '0': 0.0, '1': 1.0, 'F': 0.0, 'M': 1.0},
    'PTETHCAT': {'Hisp/Latino': 1.0, 'Not Hisp/Latino': 0.0, 'Unknown': np.nan},
    'PTRACCAT': {'White': 0.0, 'Black': 1.0, 'Asian': 2.0, 'Am Indian/Alaskan': 3.0, 'More than one': 4.0, 'Unknown': np.nan},
    'PTMARRY': {'Married': 0.0, 'Widowed': 1.0, 'Divorced': 2.0, 'Never married': 3.0, 'Unknown': np.nan},
    'PTHAND': {'Right': 0.0, 'Left': 1.0, 'Ambidextrous': 2.0}
}

def safe_encode_val(val, col_name=""):
    """
    Safely converts strings, categorical levels, and numbers to float without
    silently destroying valid categorical text features (Fixes A1).
    """
    if val in ('NA', '', 'None', None, 'N/A', 'nan'):
        return np.nan
    
    # Check known categorical string mappings
    if col_name in CATEGORICAL_MAPPINGS:
        if val in CATEGORICAL_MAPPINGS[col_name]:
            return CATEGORICAL_MAPPINGS[col_name][val]
            
    try:
        return float(val)
    except ValueError:
        # Check general binary / string patterns
        val_str = str(val).strip().lower()
        if val_str in ('yes', 'true', 'completed', 'pass'):
            return 1.0
        elif val_str in ('no', 'false', 'fail'):
            return 0.0
        return np.nan

def load_and_preprocess_adni_data(features_path, targets_path, purge_admin=True, purge_zero_variance=True):
    """
    Loads raw CSV features & targets, applies categorical encoding, purges administrative flags,
    filters out zero-variance/unobserved columns, and returns pristine X, Y matrices.
    """
    with open(features_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        x_header = next(reader)
        x_rows = list(reader)
        
    with open(targets_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        y_header = next(reader)
        y_rows = list(reader)
        
    rid_idx_x = x_header.index('RID')
    rid_idx_y = y_header.index('RID')
    
    x_rows.sort(key=lambda r: int(r[rid_idx_x]))
    y_rows.sort(key=lambda r: int(r[rid_idx_y]))
    
    rids_x = [int(r[rid_idx_x]) for r in x_rows]
    rids_y = [int(r[rid_idx_y]) for r in y_rows]
    assert rids_x == rids_y, "Row alignment mismatch between features and targets!"
    
    keep_x_indices = []
    feature_names = []
    
    for i, col in enumerate(x_header):
        if col == 'RID':
            continue
        if purge_admin and ADMIN_FEATURE_REGEX.match(col):
            continue
        keep_x_indices.append(i)
        feature_names.append(col)
        
    target_indices = [i for i, col in enumerate(y_header) if col not in ('RID', 'SITEID')]
    target_names = [y_header[i] for i in target_indices]
    
    X = np.array([[safe_encode_val(row[i], x_header[i]) for i in keep_x_indices] for row in x_rows], dtype=np.float64)
    Y = np.array([[safe_encode_val(row[i], y_header[i]) for i in target_indices] for row in y_rows], dtype=np.float64)
    
    # Safe global purge: remove columns that are universally dead across ALL subjects.
    # A column that is all-NaN or has a single unique value in the full dataset is dead in
    # every possible train fold, so purging globally is safe and leak-free.
    if purge_zero_variance:
        obs_counts = np.sum(~np.isnan(X), axis=0)
        col_keep = np.ones(X.shape[1], dtype=bool)
        for j in range(X.shape[1]):
            if obs_counts[j] == 0:
                col_keep[j] = False
            else:
                unique_vals = np.unique(X[~np.isnan(X[:, j]), j])
                if len(unique_vals) <= 1:
                    col_keep[j] = False
        n_purged = int(np.sum(~col_keep))
        if n_purged > 0:
            X = X[:, col_keep]
            feature_names = [feature_names[i] for i in range(len(feature_names)) if col_keep[i]]
            print(f"Purged {n_purged} universally dead columns (all-NaN or constant). Remaining: {X.shape[1]} features.")
    
    return X, Y, feature_names, target_names, np.array(rids_x)

def compute_observed_scaling(X_train):
    """
    Computes feature means and standard deviations strictly on OBSERVED non-missing values
    BEFORE mean imputation. Prevents missingness inflation on imaging panels.
    Sets zero-variance/unobserved training features to std=1.0 so they scale safely to 0.
    """
    x_means = np.nanmean(X_train, axis=0) # compute mean strictly on observed entries
    x_means[np.isnan(x_means)] = 0.0 # set NaN means to 0.0
    
    # Compute std strictly on observed entries
    x_stds = np.nanstd(X_train, axis=0) 
    x_stds[np.isnan(x_stds) | (x_stds <= 1e-6)] = 1.0 # prevent division by zero or NaN
    
    return x_means, x_stds

def apply_scaling_and_imputation(X_train, X_test, x_means, x_stds, max_z=10.0):
    """
    Applies mean imputation using training means, followed by standardization using observed stds.
    Clips standardized features to [-max_z, max_z] (default +-10.0) to structurally bound
    extreme z-score outliers from small-sample std estimation.
    """
    X_train_imp = np.where(np.isnan(X_train), x_means, X_train) # impute missing values with training means
    X_test_imp = np.where(np.isnan(X_test), x_means, X_test) # impute missing values with training means
    
    X_train_scaled = np.clip((X_train_imp - x_means) / x_stds, -max_z, max_z) # standardize using training means and stds
    X_test_scaled = np.clip((X_test_imp - x_means) / x_stds, -max_z, max_z) # standardize using training means and stds
    
    return X_train_scaled, X_test_scaled, X_train_imp, X_test_imp

def get_kfold_splits(n_samples, n_splits=5, seed=42):
    """
    Generates K-Fold cross-validation splits.
    """
    np.random.seed(seed) 
    indices = np.random.permutation(n_samples) # shuffle indices
    folds = np.array_split(indices, n_splits) # split indices into k folds
    
    splits = []
    for i in range(n_splits):
        test_idx = folds[i] # test set is the current fold
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i]) # train set is all other folds
        splits.append((train_idx, test_idx))
    return splits
