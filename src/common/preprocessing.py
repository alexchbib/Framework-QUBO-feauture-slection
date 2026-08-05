import os
import csv
import re
import numpy as np

# Administrative metadata regex to purge before modeling (Fixes A6)
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

def load_and_preprocess_adni_data(features_path, targets_path, purge_admin=True):
    """
    Loads raw CSV features & targets, applies categorical encoding, purges administrative flags,
    and returns pristine X, Y matrices, feature names, target names, and patient RIDs.
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
    
    return X, Y, feature_names, target_names, np.array(rids_x)

def compute_observed_scaling(X_train):
    """
    Computes feature means and standard deviations strictly on OBSERVED non-missing values
    BEFORE mean imputation (Fixes A2). Prevents missingness inflation on imaging panels.
    """
    x_means = np.nanmean(X_train, axis=0)
    x_means[np.isnan(x_means)] = 0.0
    
    # Compute std strictly on observed entries
    x_stds = np.nanstd(X_train, axis=0)
    x_stds[np.isnan(x_stds) | (x_stds == 0)] = 1.0
    
    return x_means, x_stds

def apply_scaling_and_imputation(X_train, X_test, x_means, x_stds):
    """
    Applies mean imputation using training means, followed by standardization using observed stds.
    """
    X_train_imp = np.where(np.isnan(X_train), x_means, X_train)
    X_test_imp = np.where(np.isnan(X_test), x_means, X_test)
    
    X_train_scaled = (X_train_imp - x_means) / x_stds
    X_test_scaled = (X_test_imp - x_means) / x_stds
    
    return X_train_scaled, X_test_scaled, X_train_imp, X_test_imp

def get_kfold_splits(n_samples, n_splits=5, seed=42):
    """
    Generates K-Fold cross-validation splits.
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    folds = np.array_split(indices, n_splits)
    
    splits = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, test_idx))
    return splits
