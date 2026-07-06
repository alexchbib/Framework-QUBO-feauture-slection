# ADNI Clinical Trial Pilot Data Pipeline

## 1. Project Objective
The objective of this pipeline is to extract, filter, and construct a highly specific dataset matrix from the `ADNIMERGE2` R package. This matrix acts as a benchmark dataset to test and prototype a multi-endpoint feature selection model.

A key mathematical requirement for this prototype was generating a high-dimensional dataset where the number of features ($P$) significantly exceeds the number of subjects ($n$). Specifically, the target was to rigidly enforce **$n = 40\%$ of $P$**.

## 2. Source Datasets (ADNIMERGE2)
To maximize the feature space and provide distinct, cohesive modalities for the feature selection model, the following core tables were merged:

*   **Demographics (`PTDEMOG`)** & **Genetics (`APOERES`)**
*   **Psychometric Battery (`ADAS`, `MMSE`, `NEUROBAT`)**: Including ADAS-Cog, MMSE, RAVLT, and Trail Making tests.
*   **Functional Assessments (`CDR`, `FAQ`)**: Including Clinical Dementia Rating (CDR-SB) and Functional Assessment Questionnaire.
*   **Structural Neuroimaging (`UCSFFSX7`)**: Over 300 regional brain volume, surface area, and cortical thickness metrics.
*   **Fluid Biomarkers (`UPENNBIOMK_MASTER`)**: Cerebrospinal fluid markers (Abeta, Tau, p-Tau).
*   **PET Imaging (`UCBERKELEYFDG_8mm`, `UCBERKELEY_AMY_6MM`, `UCBERKELEY_TAU_6MM`)**: FDG (metabolism), AV45 (Amyloid), and Tau protein regional standardized uptake value ratios (SUVRs).
*   **Advanced MRI (`DTIROI_MEAN`, `UCSFASLFS_V2`)**: Diffusion Tensor Imaging (white matter tract integrity) and Arterial Spin Labeling (cerebral blood flow).

## 3. Data Engineering Workflow

### A. Baseline Filtering
ADNI utilizes multiple disparate codes to denote a "baseline" visit. The pipeline applies a dynamic baseline filter that uniformly recognizes all known screening/baseline identifiers to ensure exactly one chronological baseline row is extracted per subject.

### B. Cohort Standardization
The pipeline was strictly filtered to include only subjects from the **ADNI2** phase (`ORIGPROT == "ADNI2"`). This isolates patients who underwent modern biomarker testing and prevents variance caused by shifting protocols across the ADNI lifecycle.

### C. Metadata Purging
Administrative metadata columns (e.g., `update_stamp`, `USERDATE`, `VISDATE`, `ID.xxx`) were purged using regex to prevent them from corrupting the feature selection algorithm. The only retained identifiers are **`RID`** (Patient ID) and **`SITEID`** (Clinical Site). `SITEID` is explicitly retained to allow downstream algorithms to control for multi-center batch effects.

### D. Missingness Strategy
1.  **Row Dropping:** A global missingness threshold of **90%** was applied. Any "ghost" subject missing more than 90% of the clinical features was dropped.
2.  **Column Dropping:** Any clinical feature completely empty (100% missing rate) was automatically dropped.
3.  **Imputation Testing:** All other missingness was preserved completely intact so the dataset can benchmark modern imputation handling.

## 4. Algorithmic Test Preparation ($P \gg n$)
To achieve the $P \gg n$ mathematical constraint, the following calculation and random downsampling logic was applied:

1.  Calculated exact feature space after the multi-table merge and metadata purge: **$P = 2,129$**
2.  Calculated target cohort size: **$n = 852$** ($0.40 \times 2,129$)
3.  Randomly sampled the remaining ADNI2 cohort to exactly 852 subjects (`set.seed(123)` applied for reproducibility).

*(Note: The pipeline currently executes Option A: Cross-Sectional baseline modeling. A commented-out block (Option B) is included in the script for transitioning to Longitudinal target extraction).*

## 5. Final Output Matrices

The pipeline produces two final files:

1.  **`adni_pilot_matrix_benchmark_v2.csv`**: The finalized $852 \times 2131$ data matrix.
2.  **`adni_missingness_report_benchmark_v2.csv`**: A domain-mapped metadata dictionary tracking the exact NA percentage for all columns.
