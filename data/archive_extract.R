# extract_adni_matrix.R
# This script extracts, filters, and constructs a specific dataset matrix
# from the ADNIMERGE2 R package for a clinical trial pilot project.

library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# Data Directory
data_dir <- "C:/Users/User/Desktop/Framework/ADNIMERGE2/ADNIMERGE2/data"

# Load required datasets directly from .rda files to bypass installation
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
# 6. PET Imaging (Metabolism and Amyloid)
load(file.path(data_dir, "UCBERKELEYFDG_8mm.rda"))
load(file.path(data_dir, "UCBERKELEY_AMY_6MM.rda"))
# 7. Advanced Neuroimaging (DTI, ASL, Tau)
load(file.path(data_dir, "DTIROI_MEAN.rda"))
load(file.path(data_dir, "UCSFASLFS_V2.rda"))
load(file.path(data_dir, "UCBERKELEY_TAU_6MM.rda"))

# Helper function to filter for baseline and remove duplicates
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

# Apply baseline filtering
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

# Merging datasets by RID
merged_data <- ptdemog_bl %>%
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

# Restrict to ADNI2 cohort for cohesive data
merged_data <- merged_data %>% filter(ORIGPROT == "ADNI2" | ORIGPROT.adas == "ADNI2")

# Purge Administrative Metadata
# Keep only RID and SITEID
meta_regex <- "(USERDATE|update_stamp|HAS_QC_ERROR|DD_CRF_VERSION|VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE)"
id_regex <- "^(ID\\.|PTID|Phase|DX|diagnosis)" # Be careful not to drop diagnostic targets if they are in there, but we have ADAS/CDR

merged_data <- merged_data %>%
  select(-matches(meta_regex, ignore.case = TRUE)) %>%
  select(-matches(id_regex, ignore.case = TRUE))
  
# Drop columns with 100% missingness
cols_to_drop <- merged_data %>%
  summarise_all(~ mean(is.na(.))) %>%
  select(where(~ . == 1)) %>%
  names()
merged_data <- merged_data %>% select(-all_of(cols_to_drop))

# Drop subjects with >90% global missingness
row_missingness <- rowMeans(is.na(merged_data))
merged_data <- merged_data[row_missingness <= 0.90, ]

# ==============================================================================
# OPTION B: Longitudinal Endpoint Extraction (Commented out for future use)
# ==============================================================================
# If your model transitions from cross-sectional (baseline X predicting baseline Y)
# to predicting future longitudinal outcomes, uncomment and adapt the code below.
# 
# m24_endpoints <- ADAS %>% 
#   filter(VISCODE == "m24" | VISCODE2 == "m24") %>%
#   select(RID, ADAS_M24 = TOTAL13) %>% # Example target at 24 months
#   distinct(RID, .keep_all = TRUE)
#
# merged_data <- merged_data %>%
#   inner_join(m24_endpoints, by = "RID") # inner_join drops subjects who didn't reach M24
# ==============================================================================

# Set n = 40% of P
P <- ncol(merged_data) - 2 # Subtract RID and SITEID from the feature count
target_n <- round(0.40 * P)

cat("==== Sampling Information ====\n")
cat("Cleaned Features (P):", P, "\n")
cat("Target Subjects (n = 40% of P):", target_n, "\n\n")

set.seed(123) # Reproducibility
if(nrow(merged_data) >= target_n) {
  merged_data <- merged_data %>% sample_n(target_n)
} else {
  cat("Warning: Available patients (", nrow(merged_data), ") is less than target n. Sampling with replacement.\n")
  merged_data <- merged_data %>% sample_n(target_n, replace = TRUE)
}

# Domain Partitioning Metadata Map
domain_map <- data.frame(
  Feature = colnames(merged_data),
  Domain = case_when(
    colnames(merged_data) %in% colnames(ptdemog_bl) ~ "Demographics",
    colnames(merged_data) %in% colnames(apoe_bl) ~ "Genetics",
    colnames(merged_data) %in% c(colnames(adas_bl), colnames(mmse_bl), colnames(neuro_bl)) ~ "Psychometric Battery",
    colnames(merged_data) %in% c(colnames(cdr_bl), colnames(faq_bl)) ~ "Functional Assessments",
    colnames(merged_data) %in% c(colnames(mri_bl), colnames(fdg_bl), colnames(amy_bl), colnames(dti_bl), colnames(asl_bl), colnames(tau_bl)) ~ "Neuroimaging",
    colnames(merged_data) %in% colnames(biomk_bl) ~ "Fluid Biomarkers",
    TRUE ~ "Other Metadata"
  )
)

# Extract Missingness Rates
missingness_report <- merged_data %>%
  summarise_all(~ mean(is.na(.))) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Missing_Rate") %>%
  left_join(domain_map, by = "Feature")

# Print Summary Verification
cat("==== Data Matrix Summary ====\n")
cat("Total Subjects (Rows):", nrow(merged_data), "\n")
cat("Total Features (Columns):", ncol(merged_data), "\n\n")

cat("==== Domain Feature Counts ====\n")
print(table(domain_map$Domain))

# Write data and metadata to CSV
write.csv(merged_data, "adni_pilot_matrix_benchmark_v2.csv", row.names = FALSE)
write.csv(missingness_report, "adni_missingness_report_benchmark_v2.csv", row.names = FALSE)

cat("\nOutputs saved: adni_pilot_matrix_benchmark_v2.csv, adni_missingness_report_benchmark_v2.csv\n")
