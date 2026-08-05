import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.preprocessing import get_kfold_splits

def solve_fista_l21_mtfl(X_sc, Y_sc, target_mask=None, lambda_val=0.05, max_iters=5000, tol=1e-8):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for L2,1 Group Lasso.
    Features exact target observation masking and provably stable Lipschitz step size under N_l task counts.
    """
    N, d = X_sc.shape
    T = Y_sc.shape[1]
    
    if target_mask is None:
        target_mask = np.ones_like(Y_sc)
        
    N_l = np.sum(target_mask, axis=0)
    N_l[N_l == 0] = 1.0
    min_N_l = np.min(N_l)
    
    # Compute exact largest singular value and provably stable Lipschitz constant for N_l
    s_val = np.linalg.svd(X_sc, compute_uv=False)
    L = (s_val[0]**2) / min_N_l
    step = 1.0 / L
    
    W = np.zeros((d, T), dtype=np.float64)
    Z = W.copy()
    t_fista = 1.0
    
    def compute_obj(W_curr):
        diff = (X_sc.dot(W_curr) - Y_sc) * target_mask
        loss = 0.5 * np.sum(np.sum(diff**2, axis=0) / N_l)
        reg = lambda_val * np.sum(np.linalg.norm(W_curr, axis=1))
        return loss + reg
        
    obj_old = compute_obj(W)
    
    for it in range(max_iters):
        diff_z = (X_sc.dot(Z) - Y_sc) * target_mask / N_l
        grad = X_sc.T.dot(diff_z)
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

def select_lambda_inner_cv(X_tr_sc, Y_tr_sc, target_mask, lambda_candidates=[0.001, 0.01, 0.05, 0.1, 0.5]):
    """
    Inner 3-fold cross-validation to select optimal lambda within each outer training fold.
    """
    inner_splits = get_kfold_splits(X_tr_sc.shape[0], n_splits=3, seed=42)
    best_lambda = lambda_candidates[0]
    best_inner_r2 = -np.inf
    
    for l_cand in lambda_candidates:
        inner_r2s = []
        for in_tr_idx, in_val_idx in inner_splits:
            X_in_tr, X_in_val = X_tr_sc[in_tr_idx], X_tr_sc[in_val_idx]
            Y_in_tr, Y_in_val = Y_tr_sc[in_tr_idx], Y_tr_sc[in_val_idx]
            mask_in_tr = target_mask[in_tr_idx]
            mask_in_val = target_mask[in_val_idx]
            
            W_in = solve_fista_l21_mtfl(X_in_tr, Y_in_tr, target_mask=mask_in_tr.astype(float), lambda_val=l_cand, max_iters=5000, tol=1e-8)
            preds_val = X_in_val.dot(W_in)
            
            val_r2s = []
            for t_i in range(Y_tr_sc.shape[1]):
                valid = mask_in_val[:, t_i]
                if np.sum(valid) > 0:
                    ss_res = np.sum((Y_in_val[valid, t_i] - preds_val[valid, t_i])**2)
                    ss_tot = np.sum((Y_in_val[valid, t_i] - np.mean(Y_in_val[valid, t_i]))**2)
                    val_r2s.append(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)
            if val_r2s:
                inner_r2s.append(np.mean(val_r2s))
        mean_in_score = np.mean(inner_r2s) if inner_r2s else -np.inf
        if mean_in_score > best_inner_r2:
            best_inner_r2 = mean_in_score
            best_lambda = l_cand
            
    return best_lambda
