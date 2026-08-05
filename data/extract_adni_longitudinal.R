# extract_adni_longitudinal.R
# Pristine extraction script for ADNI longitudinal features & targets
# Resolves Audit findings A3 (NEUROBAT baseline selection), A4 (Provenance-based panel mapping), 
# A5 (FDG PET pivoting & SUVR ratio), and C3 (Relative path handling).

#packages
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# 1. Path setup (relative to script location)
script_dir <- getwd()
data_dir <- file.path(script_dir, "ADNIMERGE2", "ADNIMERGE2", "data")
if (!dir.exists(data_dir)) {
  data_dir <- file.path(script_dir, "..", "ADNIMERGE2", "ADNIMERGE2", "data")
}

cat("Loading raw R data tables from:", data_dir, "\n")

# Load RDA files
load(file.path(data_dir, "PTDEMOG.rda")) # Demographics
load(file.path(data_dir, "APOERES.rda")) #Age, Sex, Education, APOE ε4 status
load(file.path(data_dir, "ADAS.rda")) #Alzheimer’s Disease Assessment Scale-Cognitive Subscale
load(file.path(data_dir, "MMSE.rda")) #Mini-Mental State Exam
load(file.path(data_dir, "NEUROBAT.rda")) #Neuropsychological Test Battery
load(file.path(data_dir, "CDR.rda")) #Clinical Dementia Rating (CDR)
load(file.path(data_dir, "FAQ.rda")) #Functional Assessment Questionnaire (FAQ)
load(file.path(data_dir, "UCSFFSX7.rda")) #Structural MRI (FreeSurfer)
load(file.path(data_dir, "UPENNBIOMK_MASTER.rda")) #CSF Biomarkers (Lumbar Puncture)
load(file.path(data_dir, "UCBERKELEYFDG_8mm.rda")) #FDG PET Imaging (Cerebral Glucose Metabolism)
load(file.path(data_dir, "UCBERKELEY_AMY_6MM.rda")) #Amyloid PET Imaging (Cerebral Amyloid Deposition)
load(file.path(data_dir, "DTIROI_MEAN.rda")) #DTI MRI (Diffusion Tensor)
load(file.path(data_dir, "UCSFASLFS_V2.rda")) #ASL MRI (Cerebral Blood Flow)
load(file.path(data_dir, "UCBERKELEY_TAU_6MM.rda")) #Tau PET Imaging (Cerebral Tau Deposition)

# Baseline filter prioritizing complete 'bl' over 'sc' 
filter_bl <- function(df) { 
  baseline_codes <- c("bl", "sc", "scmri", "v01", "v02", "v03", "4_bl", "4_sc", "init") #Baseline visit codes
  has_v1 <- "VISCODE" %in% names(df)
  has_v2 <- "VISCODE2" %in% names(df)
  
  if (has_v1 || has_v2) { #Check if there are VISCODE or VISCODE2 columns
    df_sub <- df 
    if (has_v1 && has_v2) { 
      df_sub <- df_sub %>% filter(VISCODE %in% baseline_codes | VISCODE2 %in% baseline_codes)
    } else if (has_v1) {
      df_sub <- df_sub %>% filter(VISCODE %in% baseline_codes)
    } else {
      df_sub <- df_sub %>% filter(VISCODE2 %in% baseline_codes)
    }
    
    df_sub$vis_priority <- 5
    v1_vals <- if(has_v1) df_sub$VISCODE else rep("", nrow(df_sub))
    v2_vals <- if(has_v2) df_sub$VISCODE2 else rep("", nrow(df_sub))
    
    is_bl  <- (!is.na(v1_vals) & v1_vals == "bl") | (!is.na(v2_vals) & v2_vals == "bl")
    is_v01 <- (!is.na(v1_vals) & v1_vals == "v01") | (!is.na(v2_vals) & v2_vals == "v01")
    is_4bl <- (!is.na(v1_vals) & v1_vals == "4_bl") | (!is.na(v2_vals) & v2_vals == "4_bl")
    is_sc  <- (!is.na(v1_vals) & v1_vals == "sc") | (!is.na(v2_vals) & v2_vals == "sc")
    
    df_sub$vis_priority[is_sc]  <- 4
    df_sub$vis_priority[is_4bl] <- 3
    df_sub$vis_priority[is_v01] <- 2
    df_sub$vis_priority[is_bl]  <- 1
    
    res <- df_sub %>%
      arrange(RID, vis_priority) %>%
      distinct(RID, .keep_all = TRUE) %>%
      select(-vis_priority)
    return(res)
  } else {
    return(df %>% distinct(RID, .keep_all = TRUE))
  }
}

# Apply baseline filtering
ptdemog_bl <- filter_bl(PTDEMOG)
apoe_bl    <- filter_bl(APOERES)
adas_bl    <- filter_bl(ADAS)
mmse_bl    <- filter_bl(MMSE)
neuro_bl   <- filter_bl(NEUROBAT) # Retains full baseline psychometric tests!
cdr_bl     <- filter_bl(CDR)
faq_bl     <- filter_bl(FAQ)
mri_bl     <- filter_bl(UCSFFSX7)
biomk_bl   <- filter_bl(UPENNBIOMK_MASTER)
amy_bl     <- filter_bl(UCBERKELEY_AMY_6MM)
dti_bl     <- filter_bl(DTIROI_MEAN)
asl_bl     <- filter_bl(UCSFASLFS_V2)
tau_bl     <- filter_bl(UCBERKELEY_TAU_6MM)

# 2. Fix FDG PET Pivoting & Biomarker Ratio (Fixes A5)
fdg_wide <- UCBERKELEYFDG_8mm %>%
  filter(VISCODE %in% c("bl", "sc", "v01") | VISCODE2 %in% c("bl", "sc", "v01")) %>%
  arrange(RID) %>%
  pivot_wider(
    id_cols = RID,
    names_from = ROINAME,
    values_from = c(MEAN, MAX, STDEV, TOTVOX),
    names_glue = "FDG_{ROINAME}_{.value}"
  ) %>%
  distinct(RID, .keep_all = TRUE)

if ("FDG_MetaROI_MEAN" %in% names(fdg_wide) && "FDG_Top50PonsVermis_MEAN" %in% names(fdg_wide)) {
  fdg_wide <- fdg_wide %>%
    mutate(FDG_SUVR = FDG_MetaROI_MEAN / FDG_Top50PonsVermis_MEAN)
}

# Track provenance of features for panel mapping (Fixes A4)
map_table <- function(df, panel_name) {
  cols <- setdiff(names(df), "RID")
  data.frame(Feature_Name = cols, Panel_Name = panel_name, stringsAsFactors = FALSE)
}

provenance_list <- list(
  map_table(ptdemog_bl, "Demographics & Medical History"),
  map_table(apoe_bl, "Demographics & Medical History"),
  map_table(adas_bl, "ADAS-Cog Assessment"),
  map_table(mmse_bl, "MMSE Assessment"),
  map_table(cdr_bl, "Clinical Dementia Rating (CDR)"),
  map_table(faq_bl, "Functional Assessment Questionnaire (FAQ)"),
  map_table(mri_bl, "Structural MRI (FreeSurfer)"),
  map_table(biomk_bl, "CSF Biomarkers (Lumbar Puncture)"), # Ensures PTAU/PTAU_RAW mapped to CSF!
  map_table(fdg_wide, "FDG PET Imaging"),
  map_table(amy_bl, "Amyloid PET Imaging"),
  map_table(dti_bl, "DTI MRI (Diffusion Tensor)"),
  map_table(asl_bl, "ASL MRI (Arterial Spin Labeling)"),
  map_table(tau_bl, "Tau PET Imaging")
)

# NEUROBAT sub-test table mapping
neuro_cols <- setdiff(names(neuro_bl), "RID")
neuro_map <- data.frame(Feature_Name = neuro_cols, Panel_Name = case_when(
  str_detect(neuro_cols, "^AV|^ANARTERR") ~ "Rey Auditory Verbal Learning Test (RAVLT)",
  str_detect(neuro_cols, "^TRA") ~ "Trail Making Test (TMT)",
  str_detect(neuro_cols, "^BNT") ~ "Boston Naming Test",
  str_detect(neuro_cols, "^CLOCK") ~ "Clock Drawing Test",
  str_detect(neuro_cols, "^COPY") ~ "Copy Drawing Test",
  str_detect(neuro_cols, "^CAT") ~ "Category Fluency Test",
  str_detect(neuro_cols, "^LM|^LIMM|^LDEL") ~ "Logical Memory Test",
  TRUE ~ "Psychometric Battery (Other)"
), stringsAsFactors = FALSE)

provenance_list[[length(provenance_list) + 1]] <- neuro_map
provenance_df <- bind_rows(provenance_list) %>% distinct(Feature_Name, .keep_all = TRUE)

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
  left_join(fdg_wide, by = "RID", suffix = c("", ".fdg")) %>%
  left_join(amy_bl, by = "RID", suffix = c("", ".amy")) %>%
  left_join(dti_bl, by = "RID", suffix = c("", ".dti")) %>%
  left_join(asl_bl, by = "RID", suffix = c("", ".asl")) %>%
  left_join(tau_bl, by = "RID", suffix = c("", ".tau"))

# Restrict to ADNI2 Cohort
features <- features %>% filter(ORIGPROT == "ADNI2" | ORIGPROT.adas == "ADNI2")

# Purge Administrative Metadata
meta_regex <- "(USERDATE|update_stamp|HAS_QC_ERROR|DD_CRF_VERSION|VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE)"
id_regex <- "^(ID\\.|PTID|Phase|DX|diagnosis)"
features <- features %>%
  select(-matches(meta_regex, ignore.case = TRUE)) %>%
  select(-matches(id_regex, ignore.case = TRUE))

# Extract Month 24 Targets
target_adas <- ADAS %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_ADAS13 = TOTAL13) %>% distinct(RID, .keep_all = TRUE)
target_cdr  <- CDR %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_CDRSB = CDRSB) %>% distinct(RID, .keep_all = TRUE)
target_mmse <- MMSE %>% filter(VISCODE == "m24" | VISCODE2 == "m24") %>% select(RID, M24_MMSE = MMSCORE) %>% distinct(RID, .keep_all = TRUE)

targets <- target_adas %>%
  full_join(target_cdr, by = "RID") %>%
  full_join(target_mmse, by = "RID")

# Completer Join
dataset <- features %>% inner_join(targets, by = "RID")

# Drop subjects where all three targets are NA
target_missing <- rowSums(is.na(dataset[, c("M24_ADAS13", "M24_CDRSB", "M24_MMSE")]))
dataset <- dataset[target_missing < 3, ]

# Filter row missingness (<= 90% missing baseline features)
feature_cols <- setdiff(names(dataset), c("RID", "SITEID", "M24_ADAS13", "M24_CDRSB", "M24_MMSE"))
row_missingness <- rowMeans(is.na(dataset[, feature_cols]))
dataset <- dataset[row_missingness <= 0.90, ]

# Distinct unique patients
dataset <- dataset %>% distinct(RID, .keep_all = TRUE)

# Drop 100% missing columns AFTER cohort restriction (Fixes A3)
cols_to_drop <- dataset %>%
  select(all_of(feature_cols)) %>%
  summarise_all(~ mean(is.na(.))) %>%
  select(where(~ . == 1)) %>%
  names()

if (length(cols_to_drop) > 0) {
  dataset <- dataset %>% select(-all_of(cols_to_drop))
}

feature_cols <- setdiff(names(dataset), c("RID", "SITEID", "M24_ADAS13", "M24_CDRSB", "M24_MMSE"))

final_features <- dataset %>% select(all_of(c("RID", "SITEID", feature_cols)))
final_targets  <- dataset %>% select(RID, SITEID, M24_ADAS13, M24_CDRSB, M24_MMSE)

# Filter provenance dataframe for retained features
final_feature_names <- setdiff(names(final_features), c("RID", "SITEID"))
provenance_final <- data.frame(Feature_Name = final_feature_names, stringsAsFactors = FALSE) %>%
  left_join(provenance_df, by = "Feature_Name") %>%
  mutate(Panel_Name = ifelse(is.na(Panel_Name), "Demographics & Medical History", Panel_Name))

# Panel Costs reference (includes FDG PET $2,000)
panel_costs_ref <- data.frame(
  Panel_Name = c(
    "Demographics & Medical History", "ADAS-Cog Assessment", "MMSE Assessment",
    "Clock Drawing Test", "Copy Drawing Test", "Logical Memory Test",
    "Rey Auditory Verbal Learning Test (RAVLT)", "Category Fluency Test",
    "Trail Making Test (TMT)", "Boston Naming Test", "Clinical Dementia Rating (CDR)",
    "Functional Assessment Questionnaire (FAQ)", "CSF Biomarkers (Lumbar Puncture)",
    "DTI MRI (Diffusion Tensor)", "ASL MRI (Arterial Spin Labeling)",
    "FDG PET Imaging", "Tau PET Imaging", "Amyloid PET Imaging",
    "Structural MRI (FreeSurfer)"
  ),
  Cost_USD = c(50, 300, 150, 50, 50, 100, 150, 50, 100, 100, 250, 100, 1000, 1500, 1500, 2000, 3000, 3000, 1500)
)

# Output directories
data_out_dir <- file.path(script_dir, "data")
b1_out_dir <- file.path(script_dir, "benchmark 1 multitask learning")

write.csv(final_features, file.path(data_out_dir, "adni_longitudinal_features.csv"), row.names = FALSE)
write.csv(final_targets, file.path(data_out_dir, "adni_longitudinal_targets.csv"), row.names = FALSE)
write.csv(provenance_final, file.path(b1_out_dir, "feature_to_panel_mapping.csv"), row.names = FALSE)
write.csv(panel_costs_ref, file.path(b1_out_dir, "panel_costs.csv"), row.names = FALSE)

cat("\n==== Extraction Successful ====\n")
cat("Final Subjects (n):", nrow(final_features), "\n")
cat("Final Features (P):", ncol(final_features) - 2, "\n")
cat("Saved features, targets, provenance mapping, and panel costs.\n")
