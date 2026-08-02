# ADNI Multi-Endpoint Feature Selection Framework
## Project Summary & Audited Baselines Report

This document details the complete end-to-end workflow executed to establish a rigorous, audited foundation for a novel QUBO-based clinical trial feature selection algorithm. The ultimate objective is to minimize financial costs and patient burden while maximizing predictive accuracy across multiple Alzheimer's Disease cognitive endpoints.

---

### Phase 1: Data Acquisition & Preprocessing (Audited Cohort)
We leveraged the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** longitudinal dataset to create a robust multi-task learning environment.

1. **Feature Matrix**: Extracted a pristine cohort of **425 unique subjects** and **2,129 clinical features** (deduplicated to eliminate patient row duplication and train-test leakage). Features span demographics, psychometric batteries, structural MRI, ASL MRI, DTI MRI, CSF biomarkers, and Amyloid/Tau PET scans.
2. **Targets (Multi-Endpoint)**: We targeted three continuous cognitive decline scores at Month 24:
   - `ADAS13` (Alzheimer's Disease Assessment Scale - Cognitive Subscale 13)
   - `CDRSB` (Clinical Dementia Rating Sum of Boxes)
   - `MMSE` (Mini-Mental State Examination)
3. **Evaluation Protocol**: Split into an 80/20 train/test split. Feature missingness in $X$ is handled via training-set mean imputation. **Target outcomes ($Y_{test}$) are evaluated strictly on observed non-missing values** without mean imputation leakage.

#### Evaluation Metrics
Because our targets (`ADAS13`, `CDRSB`, `MMSE`) are continuous numerical scores, performance is evaluated using standard regression metrics:
* **$R^2$ (R-Squared)**: Measures the proportion of disease progression variance explained by the model (**Higher is better**, 1.0 is perfect).
* **MAE (Mean Absolute Error)**: Average magnitude of prediction errors in target score points (**Lower is better**).
* **MSE (Mean Squared Error)**: Mean squared prediction error (**Lower is better**).

---

### Phase 2: Benchmark 1 — Multi-Task Feature Learning ($L_{2,1}$-norm)
To provide a mathematical baseline, we implemented continuous Multi-Task Feature Learning using an $L_{2,1}$-norm block-sparsity penalty on standardized target outcomes.

* **Selection**: The algorithm selected a core subset of **447 features**.
* **Performance (Evaluated on Non-Imputed Test Outcomes)**:
  * **Multi-Task $L_{2,1}$ Lasso Model**:
    * `ADAS13` $R^2$: **0.6438** | MAE: 4.95 | MSE: 43.19
    * `CDRSB` $R^2$: **0.7141** | MAE: 0.77 | MSE: 1.37
    * `MMSE` $R^2$: **0.6311** | MAE: 1.51 | MSE: 4.51
  * **Isolated Panel Ridge Regression**:
    * `ADAS13` $R^2$: 0.5192 | MAE: 5.77 | MSE: 58.30
    * `CDRSB` $R^2$: 0.5323 | MAE: 1.12 | MSE: 2.25
    * `MMSE` $R^2$: 0.0854 | MAE: 2.42 | MSE: 11.18
* **Takeaway**: Standardizing target outcomes and proper proximal thresholding eliminated previous gradient divergence, establishing a solid linear multi-task baseline ($R^2 \approx 0.63 - 0.71$).

---

### Phase 3: Benchmark 2 — Modern ML (XGBoost)
To establish a modern non-linear baseline, we deployed multi-output XGBoost tree ensembles.

* **Selection**: Trained XGBoost on the training set, aggregated internal tree Feature Importances, and selected the **Top 447 features** to match Benchmark 1 size for a strict 1-to-1 comparison.
* **Performance (Evaluated on Non-Imputed Test Outcomes)**:
  * `ADAS13` $R^2$: **0.7067** | MAE: 4.19 | MSE: 35.56
  * `CDRSB` $R^2$: **0.7015** | MAE: 0.73 | MSE: 1.43
  * `MMSE` $R^2$: **0.5827** | MAE: 1.60 | MSE: 5.10
* **Data Audit Note**: The previously reported $R^2 \approx 0.89$ was identified as an artifact of train-test data leakage caused by duplicate patient row sampling in the raw R extraction script. After deduplicating patients and evaluating strictly on unseen holdout test subjects, the true non-leaked accuracy ceiling for non-linear tree models is **$R^2 \approx 0.70 - 0.71$**.

---

### Phase 4: Clinical Panel Economics Evaluator
Selecting 447 features is operationally prohibitive if features trigger multiple expensive diagnostic imaging procedures.

1. **Panel Mapping**: Features are mapped across **18 distinct clinical panels** (`feature_to_panel_mapping.csv`).
2. **Administrative Metadata Filtering**: Administrative flags (`SITEID.*`, `IMAGEUID.*`, `STATUS.*`, `VERSION.*`) were filtered out to prevent administrative columns from artificially triggering panel costs.
3. **Cost Evaluation Results**:
   * Both baselines trigger the exact same **12 clinical panels**.
   * Financial Burden per Patient: **$12,450.00**.
   * Triggered Modalities: Structural MRI ($1,500), ASL MRI ($1,500), DTI MRI ($1,500), Tau PET ($3,000), Amyloid PET ($3,000), Lumbar Puncture ($1,000), CDR ($250), ADAS ($300), MMSE ($150), FAQ ($100), Demographics ($50), Logical Memory ($100).

---

### Phase 5: The QUBO Objective (Audited Goalposts)
With data leakage eliminated and baselines corrected, the true optimization goalposts are established:

1. **The Realistic Accuracy Ceiling**: $R^2 \approx 0.70 - 0.71$ (from XGBoost and Multi-Task $L_{2,1}$).
2. **The Economic Floor**: **$12,450.00 per patient** (from standard baselines triggering 12 panels).

**The Ultimate QUBO Goal**: Formulate a Quadratic Unconstrained Binary Optimization (QUBO) model that jointly maximizes predictive performance ($R^2 \approx 0.70$) while adding strong binary group penalties to eliminate entire redundant imaging panels (e.g. dropping Tau PET or ASL MRI), slashing the $12,450 price tag per patient.
