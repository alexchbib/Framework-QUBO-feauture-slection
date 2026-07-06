# extract_adni_longitudinal.R
# This script extracts pristine baseline features (X) and separate longitudinal 
# Month 24 targets (Y) from the ADNIMERGE2 R package for predictive modeling.

library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# Data Directory
data_dir <- "C:/Users/User/Desktop/Framework/ADNIMERGE2/ADNIMERGE2/data"

# 1. Demographics & Genetics
load(file.path(data_dir, "PTDEMOG.rda"))
load(file.path(data_dir, "APOERES.rda"))
# 2. Cognitive Targets
load(file.path(data_dir, "ADAS.rda"))
load(file.path(data_dir, "MMSE.rda"))
load(file.path(data_dir, "NEUROBAT.rda"))
# 3. Functional Targets
load(file.path(data_dir, "CDR.rda"))
load(file.path(data_dir, "FAQ.rda"))
# 4. Structural Biomarker Targets (MRI)
load(file.path(data_dir, "UCSFFSX7.rda"))
# 5. Fluid Biomarkers
load(file.path(data_dir, "UPENNBIOMK_MASTER.rda"))
# 6. PET Imaging
load(file.path(data_dir, "UCBERKELEYFDG_8mm.rda"))
load(file.path(data_dir, "UCBERKELEY_AMY_6MM.rda"))
# 7. Advanced Neuroimaging
load(file.path(data_dir, "DTIROI_MEAN.rda"))
load(file.path(data_dir, "UCSFASLFS_V2.rda"))
load(file.path(data_dir, "UCBERKELEY_TAU_6MM.rda"))

# Baseline Filter
filter_bl <- function(df) {
  baseline_codes <- c("bl", "sc", "scmri", "v01", "v02", "v03", "4_bl", "4_sc", "init")
  if("VISCODE" %in% names(df) && "VISCODE2" %in% names(df)) {
    df %>% filter(VISCODE %in% baseline_codes | VISCODE2 %in% baseline_codes) %>% distinct(RID, .keep_all = TRUE)
  } else if ("VISCODE" %in% names(df)) {
    df %>% filter(VISCODE %in% baseline_codes) %>% distinct(RID, .keep_all = TRUE)
  } else {
    df %>% distinct(RID, .keep_all = TRUE)
  }
}

# Apply baseline filtering for features (X)
ptdemog_bl <- filter_bl(PTDEMOG)
apoe_bl <- filter_bl(APOERES)
adas_bl <- filter_bl(ADAS)
mmse_bl <- filter_bl(MMSE)
neuro_bl <- filter_bl(NEUROBAT)
cdr_bl <- filter_bl(CDR)
faq_bl <- filter_bl(FAQ)
mri_bl <- filter_bl(UCSFFSX7)
biomk_bl <- filter_bl(UPENNBIOMK_MASTER)
fdg_bl <- filter_bl(UCBERKELEYFDG_8mm)
amy_bl <- filter_bl(UCBERKELEY_AMY_6MM)
dti_bl <- filter_bl(DTIROI_MEAN)
asl_bl <- filter_bl(UCSFASLFS_V2)
tau_bl <- filter_bl(UCBERKELEY_TAU_6MM)

# Merge Features
features <- ptdemog_bl %>%
  left_join(apoe_bl, by = "RID", suffix = c("", ".apoe")) %>%
  left_join(adas_bl, by = "RID", suffix = c("", ".adas")) %>%
  left_join(mmse_bl, by = "RID", suffix = c("", ".mmse")) %>%
  left_join(neuro_bl, by = "RID", suffix = c("", ".neuro")) %>%
  left_join(cdr_bl, by = "RID", suffix = c("", ".cdr")) %>%
  left_join(faq_bl, by = "RID", suffix = c("", ".faq")) %>%
  left_join(mri_bl, by = "RID", suffix = c("", ".mri")) %>%
  left_join(biomk_bl, by = "RID", suffix = c("", ".biomk")) %>%
  left_join(fdg_bl, by = "RID", suffix = c("", ".fdg")) %>%
  left_join(amy_bl, by = "RID", suffix = c("", ".amy")) %>%
  left_join(dti_bl, by = "RID", suffix = c("", ".dti")) %>%
  left_join(asl_bl, by = "RID", suffix = c("", ".asl")) %>%
  left_join(tau_bl, by = "RID", suffix = c("", ".tau"))

# Restrict to ADNI2
features <- features %>% filter(ORIGPROT == "ADNI2" | ORIGPROT.adas == "ADNI2")

# Purge Metadata
meta_regex <- "(USERDATE|update_stamp|HAS_QC_ERROR|DD_CRF_VERSION|VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE)"
id_regex <- "^(ID\\.|PTID|Phase|DX|diagnosis)"
features <- features %>%
  select(-matches(meta_regex, ignore.case = TRUE)) %>%
  select(-matches(id_regex, ignore.case = TRUE))

# Drop 100% Missing Features
cols_to_drop <- features %>%
  summarise_all(~ mean(is.na(.))) %>%
  select(where(~ . == 1)) %>%
  names()
features <- features %>% select(-all_of(cols_to_drop))

# ==============================================================================
# LONGITUDINAL EXTRACTION LOGIC
# ==============================================================================
# Extract Longitudinal Targets (Y) at Month 24
target_adas <- ADAS %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_ADAS13 = TOTAL13) %>% distinct(RID, .keep_all = TRUE)
target_cdr <- CDR %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_CDRSB = CDRSB) %>% distinct(RID, .keep_all = TRUE)
target_mmse <- MMSE %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_MMSE = MMSCORE) %>% distinct(RID, .keep_all = TRUE)

# Merge Targets
targets <- target_adas %>%
  full_join(target_cdr, by = "RID") %>%
  full_join(target_mmse, by = "RID")

# "Completer" Inner Join: Keep only subjects with BOTH baseline features AND M24 targets
dataset <- features %>% inner_join(targets, by = "RID")

# Drop rows where all three targets are NA (ghost completers)
target_missing <- rowSums(is.na(dataset[, c("M24_ADAS13", "M24_CDRSB", "M24_MMSE")]))
dataset <- dataset[target_missing < 3, ]

# Drop rows with >90% global missingness strictly on baseline features
feature_cols <- setdiff(names(dataset), c("RID", "SITEID", "M24_ADAS13", "M24_CDRSB", "M24_MMSE"))
row_missingness <- rowMeans(is.na(dataset[, feature_cols]))
dataset <- dataset[row_missingness <= 0.90, ]

# Calculate P and Sample
P <- length(feature_cols) - 2 # Exclude RID and SITEID
target_n <- round(0.40 * P)

cat("==== Longitudinal Sampling Information ====\n")
cat("Cleaned Baseline Features (P):", P, "\n")
cat("Target Subjects (n = 40% of P):", target_n, "\n\n")

set.seed(42) # Reproducibility
if(nrow(dataset) >= target_n) {
  dataset <- dataset %>% sample_n(target_n)
} else {
  cat("Warning: Available Completer patients (", nrow(dataset), ") is less than target n. Sampling with replacement.\n")
  dataset <- dataset %>% sample_n(target_n, replace = TRUE)
}

# Separate Features (X) and Targets (Y) into different matrices matching row-by-row
final_features <- dataset %>% select(all_of(c("RID", "SITEID", feature_cols)))
final_targets <- dataset %>% select(RID, SITEID, M24_ADAS13, M24_CDRSB, M24_MMSE)

# Verification Output
cat("==== Data Matrix Summary ====\n")
cat("Final Subjects (n):", nrow(final_features), "\n")
cat("Final Baseline Features (P):", ncol(final_features) - 2, "\n")
cat("Final Longitudinal Targets (Y):", ncol(final_targets) - 2, "\n\n")

# Write to CSV
write.csv(final_features, "adni_longitudinal_features.csv", row.names = FALSE)
write.csv(final_targets, "adni_longitudinal_targets.csv", row.names = FALSE)

cat("\nOutputs saved: adni_longitudinal_features.csv, adni_longitudinal_targets.csv\n")
