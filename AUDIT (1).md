# Repository Audit — ADNI Multi-Endpoint Feature Selection Framework

**Date:** 2026-08-03
**Scope:** full repository at commit `f36c742`
**Status:** findings report only — no source files were modified.

---

## 0. Scope & method

**Aim of the project, as understood from the code and docs.** Build a QUBO-based
clinical-trial feature selector that jointly maximises prediction of three Month-24
cognitive endpoints (`ADAS13`, `CDRSB`, `MMSE`) while minimising the financial cost of
the diagnostic panels those features require. Two baselines — Multi-Task `L2,1` and
XGBoost — exist to establish the goalposts: an accuracy ceiling of R² ≈ 0.70 and an
"economic floor" of $12,450 per patient.

**What was examined.** `Project_Summary_Report.md`; `evaluate_costs.py`;
`benchmark 1 multitask learning/mtfl_benchmark.py`; `benchmark 2 xgboost/xgb_benchmark.py`;
`data/extract_adni_longitudinal.R`; `data/extract_adni_matrix.R`;
`data/ADNI_Pipeline_Documentation.md`; `panel_costs.csv`;
`feature_to_panel_mapping.csv`; both `selected_features_benchmark*.csv`; both
`predictive_metrics_benchmark*.txt`; `data/adni_longitudinal_{features,targets}.csv`;
the four `data/adni_pilot_matrix*.csv`; and the `ADNIMERGE2` package documentation for
the source tables.

**What was executed.** `evaluate_costs.py` was re-run against both benchmarks. The
shipped feature/target matrices were analysed directly (stdlib `csv` only — this
environment has no numpy/xgboost, so the benchmark scripts themselves were **not**
re-run).

**Verified vs. inferred.** Every numeric claim below was measured from the shipped
artifacts unless explicitly marked *(inferred)*. Findings about the R extraction are
inferred from the script plus the `ADNIMERGE2` docs, since the `.rda` sources were not
loaded.

---

## 1. What reproduces correctly

These held up under checking and should not be re-litigated:

- **The headline cost figures are exact.** Both benchmarks trigger **12 panels at
  $12,450.00**, reproduced with:
  ```
  python3 evaluate_costs.py \
    --panel_costs "benchmark 1 multitask learning/panel_costs.csv" \
    --mapping_csv "benchmark 1 multitask learning/feature_to_panel_mapping.csv" \
    --features_csv "<benchmark dir>/selected_features_benchmark<N>.csv"
  ```
- **X/Y row alignment is currently correct.** `adni_longitudinal_features.csv` and
  `adni_longitudinal_targets.csv` are both 425 rows, contain identical RID sets, have no
  duplicate RIDs, and are *already in identical order*. The independent
  `x_raw.sort(...)` / `y_raw.sort(...)` in both benchmarks
  (`mtfl_benchmark.py:40-41`, `xgb_benchmark.py:39-40`) therefore aligns them correctly
  today — though it does so without ever asserting it, so a future extraction that
  emits mismatched RID sets would silently mis-pair features and outcomes.
- **The deduplication fix was right.** Removing duplicated patient rows and re-basing
  the ceiling from R² ≈ 0.89 to ≈ 0.70 was the correct call, and the reasoning in
  Phase 3 of the summary report is sound.
- **The panel-mapping table is complete.** All 2,131 columns appear in
  `feature_to_panel_mapping.csv`; only 3 are `UNMAPPED` (`STATUS`, `ROINAME`, `LONIUID`).

---

## 2. Findings that corrupt the central economic claim

These are ordered by impact. The common thread: each one biases feature selection
**toward the expensive imaging panels**, which is the opposite of what the project is
trying to demonstrate.

### A1 — 81 categorical features are silently deleted, almost all from the cheap panels

`safe_float()` (`mtfl_benchmark.py:43-49`, `xgb_benchmark.py:42-48`) ends in a bare
`except: return np.nan`. Any text-valued column becomes all-NaN; mean imputation
(`mtfl_benchmark.py:69-72`) then turns it into a constant-zero column. **81 columns are
destroyed this way**, plus 1 that is genuinely empty (`PTWORKHS`) — 82 dead columns
reaching the models.

| Panel | Voided | Columns |
|---|---|---|
| MMSE Assessment | 24 | every item response: `MMDATE`, `MMYEAR`, `MMMONTH`, `MMDAY`, `MMSEASON`, `MMHOSPIT`, `MMFLOOR`, `MMCITY`, `MMAREA`, `MMSTATE`, `MMTRIALS`… |
| Demographics | 11 | `PTGENDER`, `PTDOB`, `PTETHCAT`, `PTRACCAT`, `PTMARRY`, `PTHAND`, `PTHOME`, `PTPLANG`, `PTTLANG`, `PTNOTRT`, `PTSOURCE` |
| FAQ | 11 | every item: `FAQFINAN`, `FAQFORM`, `FAQSHOP`, `FAQGAME`, `FAQBEVG`, `FAQMEAL`, `FAQEVENT`, `FAQTV`, `FAQREM`, `FAQTRAVL`, `SOURCE.faq` |
| Clock Drawing | 5 | `CLOCKCIRC`, `CLOCKSYM`, `CLOCKNUM`, `CLOCKHAND`, `CLOCKTIME` |
| Copy Drawing | 5 | `COPYCIRC`, `COPYSYM`, `COPYNUM`, `COPYHAND`, `COPYTIME` |
| Logical Memory | 2 | `LMSTORY`, `LDELCUE` |
| CDR | 2 | `CDSOURCE`, `CDVERSION` |
| imaging/admin | 21 | `TRACER*`, `SCANDATE*`, `PROCESSDATE*`, `QC`, `RAWQC`, `FSVER`, `MANUFACTURER`, `BATCH`, … |

The consequence is asymmetric. **The entire MMSE and FAQ item batteries, participant sex,
race, ethnicity, marital status and handedness are deleted**, while the expensive imaging
panels — which are pure numeric — survive **100% intact**. Only `MMSCORE` survives from
MMSE and `PTEDUCAT`/`PTDOBYY` from demographics. A selector cannot choose a cheap panel
whose content has been zeroed out before it ever sees it.

These are decoded factor levels (`PTGENDER` = `"Female"`/`"Male"`), so they are
recoverable — they need ordinal/one-hot encoding, not silent coercion.

### A2 — Feature scaling inflates the most-missing modalities, which are the most expensive

`x_stds` is computed **after** mean imputation (`mtfl_benchmark.py:69-78`). For a column
that is fraction *m* missing, imputing the mean shrinks the standard deviation by
`sqrt(1-m)`, so the surviving observed values are inflated by `1/sqrt(1-m)` after
scaling. Measured median missingness per panel across the 425 subjects:

| Panel | Cost | Median missing | Inflation |
|---|---|---|---|
| ASL MRI | $1,500 | 79% | **×2.20** |
| Tau PET | $3,000 | 71% | **×1.85** |
| DTI MRI | $1,500 | 64% | **×1.67** |
| Amyloid PET | $3,000 | 11% | ×1.06 |
| CSF (Lumbar Puncture) | $1,000 | 9% | ×1.05 |
| Structural MRI | $1,500 | 1% | ×1.00 |
| ADAS / CDR / MMSE / FAQ / Demographics | $50–300 | 0% | ×1.00 |

Feature ranking in Benchmark 1 is the `L2,1` row-norm of coefficients fitted on these
scaled columns (`mtfl_benchmark.py:113-116`), so the three most expensive imaging panels
receive a 1.7–2.2× artificial boost in the selection ranking, purely as a function of how
incomplete they are. This is the single most consequential defect: it biases the
selector toward exactly the panels the project wants to eliminate.

Overall missingness is severe — roughly 1,300 of 2,129 columns are 60–90% missing — so
mean imputation is doing a great deal of work regardless.

### A3 — Six of the 18 panels contain no usable data, so "12 of 18 triggered" is not a result

The summary report presents 12 triggered panels as a modelling outcome. In fact six
panels are essentially empty in the shipped matrix:

| Panel | Representative column | Missing |
|---|---|---|
| RAVLT | `AVTOT1` | 423/425 |
| Trail Making | `TRAASCOR` | 423/425 |
| Boston Naming | `BNTTOTAL` | 423/425 |
| Clock Drawing | `CLOCKSCOR` | 423/425 |
| Category Fluency | `CATANIMSC` | 423/425 |
| Copy Drawing | `COPYSCOR` | 423/425 |

They were never available to be selected. Note that `LIMMTOTAL` (Logical Memory), from
the *same* `NEUROBAT` source table, is only 2/425 missing — so `NEUROBAT` did join.

*(Inferred cause.)* `filter_bl()` (`extract_adni_longitudinal.R:36-45`) filters on a set
of nine baseline codes and then applies `distinct(RID, .keep_all = TRUE)`, which keeps
whichever matching row happens to come first. For `NEUROBAT` this appears to retain the
screening (`sc`) record — which carries Logical Memory but not RAVLT/TMT/BNT/Clock/Copy —
rather than the `bl` record. The fix is to order by visit date and take the earliest
deterministically, then assert one row per RID per source table.

Related ordering issue: the 100%-missing column drop
(`extract_adni_longitudinal.R:89-94`) runs **before** the `inner_join` to targets
(`:110`) and the row filters (`:113-119`), so columns that become empty once the cohort
is narrowed to 425 completers are not caught.

### A4 — Panel regexes collide, and p-tau is billed at $50 instead of $1,000

`panel_costs.csv` assigns features to panels by regex, and the patterns overlap. Two
confirmed misassignments in the shipped mapping:

**`PTAU` and `PTAU_RAW` → "Demographics & Medical History" ($50).** CSF phosphorylated
tau requires a **lumbar puncture ($1,000)**. The Demographics pattern `^PT.*` captures
them before the CSF pattern `…|^PTAU$|^PTAU_RAW$` can. The CSF panel is left with only
`ABETA`, `TAU`, `ABETA_RAW`, `TAU_RAW`, `BATCH`, `KIT`, `STDS`. **Both benchmarks select
`PTAU` and `PTAU_RAW`.**

Today this does not change the total, because `ABETA`/`TAU` trigger the CSF panel anyway.
But it is a direct hazard for the planned QUBO: a cost-constrained solver that drops
`ABETA`/`TAU` would see p-tau as a $50 demographic variable and silently omit the $1,000
lumbar puncture from its objective.

**`TRACER`, `TRACER_SUVR_WARNING`, `TRACER.tau`, `TRACER_SUVR_WARNING.tau` → "Trail
Making Test" ($100).** PET tracer metadata billed as a pencil-and-paper cognitive test,
via `^TRA.*`.

Two further latent hazards:

- `^AV.*` (RAVLT, $150) would capture any `AV45*` amyloid column. None exist in this
  matrix, so it does not currently bite.
- **Tau PET is separated from Amyloid PET only by the `.tau` suffix that `dplyr` adds on
  column-name collision.** The two Berkeley tables share a schema, so 328 of 330 tau
  columns happen to be suffixed and matched by `.*\.tau$`. Any tau column that does *not*
  collide ends in `_SUVR` and is matched by the Amyloid pattern instead. This is a
  $3,000 panel boundary resting on an incidental artifact of the join.

The robust fix is to assign panels from **source-table provenance** at extraction time
rather than by regex over final column names.

### A5 — FDG PET has no panel, and its data mixes target and reference regions

`UCBERKELEYFDG_8mm` is a **long-format** table (7,524 rows, 11 variables) with one row
per ROI: `ROINAME` ∈ {`MetaROI`, `Top50PonsVermis`}, and value columns `MEAN`, `MAX`,
`STDEV`, `TOTVOX`.

`filter_bl()` collapses it with `distinct(RID, .keep_all = TRUE)`, keeping one arbitrary
row per subject. *(Inferred.)* The resulting `MEAN`/`MAX`/`STDEV` columns therefore hold
**the target metabolic region for some subjects and the pons/vermis reference region for
others** — the same column carries two different quantities. The FDG SUVR ratio
(MetaROI ÷ Top50PonsVermis), which is the actual biomarker, is never computed.
`ROINAME` — the column that would reveal which is which — is `UNMAPPED` and is itself
voided by A1.

Separately, **there is no FDG PET panel in `panel_costs.csv` at all**. `MEAN`, `MAX`,
`STDEV`, `TOTVOX` are captured by the Amyloid PET pattern, so a real ~$2,000 modality is
both mis-billed and absent from the economics.

### A6 — Administrative identifiers are used as predictors, then hidden at costing time

`ADMIN_REGEX` (`evaluate_costs.py:6`) exists **only in the cost evaluator**. Neither
selection script applies it — they drop only `RID` and `SITEID`
(`mtfl_benchmark.py:30-32`, `xgb_benchmark.py:29-31`). Consequently 29 administrative
columns are offered to the models as features, and both models select some:

- **Benchmark 1** selects `IMAGEUID`, `KIT`, `STDS`, `LONIUID.asl`, `IMAGEUID.asl`
- **Benchmark 2** selects `SITEID.mmse`, `SITEID.adas`, `SITEID.amy`, `IMAGEUID`, **`ID`**

Site identifiers and a raw database key are being used to predict cognitive decline. That
is a textbook site/batch confound and it inflates the reported R². The cost evaluator
then strips these columns before reporting (`evaluate_costs.py:35-36`), so the published
panel breakdown looks clean while the accuracy figures retain whatever the confound
contributed. One exclusion list should be applied *before* modelling.

Minor: `ADMIN_REGEX` gives the optional suffix group `(\..*)?` to `SITEID`, `IMAGEUID`,
`STATUS`, `VERSION`, `LONIUID`, `SOURCE` but **not** to
`VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE|RID|ID`, so a `RID.x`-style
column would slip through. Also, `panel_costs.get(panel_name, 0.0)`
(`evaluate_costs.py:61`) silently prices an unknown panel at $0 rather than failing.

---

## 3. Methodology and statistics

### B1 — Conclusions rest on ~83 test subjects with no cross-validation

n = 425, one 80/20 split on a hardcoded seed (`mtfl_benchmark.py:56-60`) → 85 test rows,
and after masking missing outcomes **82–84 usable per endpoint** (`ADAS13` 14/425 missing,
`CDRSB` 5/425, `MMSE` 10/425).

At n = 83 the 95% confidence intervals on the two headline numbers are:

| Reported | 95% CI |
|---|---|
| Benchmark 1, `ADAS13` R² = 0.6438 | [0.503, 0.753] |
| Benchmark 2, `ADAS13` R² = 0.7067 | [0.583, 0.800] |

These overlap substantially. **The reported gap between the two baselines is not
statistically distinguishable from noise**, and neither is the "R² ≈ 0.70–0.71 ceiling"
pinned down to the precision the summary report implies. Repeated k-fold or nested CV with
reported intervals is needed before any of these numbers can serve as a goalpost.

### B2 — The two baselines are not compared like-for-like

Benchmark 1's headline figures come from the `L2,1` model **itself**
(`preds_lasso = X_test_scaled.dot(W)`, `mtfl_benchmark.py:143`) — a model over all 2,129
features. Benchmark 2's come from an XGBoost **refit on the 447 selected features**
(`xgb_benchmark.py:126-137`). The genuinely comparable row for Benchmark 1 is "Ridge
Regression (on selected panel)", which scores:

```
ADAS13 R² 0.5192   CDRSB R² 0.5323   MMSE R² 0.0854
```

`MMSE R² = 0.0854` is present in `predictive_metrics_benchmark1.txt` but is not mentioned
in `Project_Summary_Report.md`. On the like-for-like comparison the `L2,1` panel is far
weaker than the summary suggests, and on MMSE it is close to uninformative.

### B3 — Both models train on mean-imputed targets

`Y_train_scaled` (`mtfl_benchmark.py:86`) and `Y_train_imp` (`xgb_benchmark.py:75`)
replace missing outcomes with the column mean. Subjects with no observed M24 score
contribute fabricated labels as training noise. The "no target imputation leak" note is
accurate about *evaluation* only (`mtfl_benchmark.py:175`, `xgb_benchmark.py:147`) — the
claim should be scoped accordingly. Benchmark 1's Ridge stage already does the right
thing (`mtfl_benchmark.py:146-148`); the primary fits should match it.

### B4 — The `L2,1` solver's step size divides by N twice, so "447 features" is an artifact

```python
alpha = LEARNING_RATE / N                                        # :98
grad  = X.T.dot(X.dot(W) - Y) / N                                # :101
W_temp = W - alpha * grad                                        # :102
threshold = alpha * MANUAL_LAMBDA                                # :106
```

The gradient is already normalised by `N`, and `alpha` divides by `N` again. With
N = 340 the effective step is roughly two orders of magnitude below the `1/L` a proximal
gradient method requires, and the per-iteration soft threshold is
`alpha * λ ≈ 7.4e-6`.

The shipped coefficients confirm non-convergence: on standardized data the `L2,1` row
norms in `selected_features_benchmark1.csv` have **max 0.093, median 0.0032**.

The practical consequence is that **the 447 selected features are a product of
`N_ITERS = 2000` early stopping, not the solution to the stated `L2,1` problem at
λ = 0.05.** There is no convergence check, and `selected_indices` is simply
"whatever is still non-zero after 2000 iterations" (`mtfl_benchmark.py:116`). `MANUAL_LAMBDA`
and `ridge_alpha = 10.0` (`:151`) are both hardcoded and never tuned.

### B5 — Benchmark 2's feature budget inherits Benchmark 1's artifact

`TOP_N_FEATURES = 447` (`xgb_benchmark.py:10`) is set to match Benchmark 1 "for a strict
1-to-1 comparison" — so an arbitrary early-stopping count defines the comparison budget
for both methods.

Worth reporting, and currently absent from the summary: **the two methods agree on only
199 of 447 features (45%)**. The report presents identical panel triggering as evidence of
robustness, but at the feature level the two selectors mostly disagree; the shared panel
set is better explained by 447 features being far more than enough to touch every
well-populated panel.

Also, `X_train_imp` is mean-imputed before being handed to XGBoost
(`xgb_benchmark.py:70-71`), which discards XGBoost's native sparsity-aware NaN handling
for no benefit.

### B6 — "$12,450 economic floor" is a ceiling, not a floor

$12,450 is what an **unconstrained** selector costs — it is an upper bound on what the
QUBO should spend, not a lower bound on what is achievable. Calling it a floor inverts the
meaning, and it makes the eventual QUBO result look better than it is.

The measured per-panel breakdown shows how much slack exists:

| Panel | Cost | Features used (B1 / B2) |
|---|---|---|
| MMSE | $150 | 1 / 1 |
| FAQ | $100 | 1 / 1 |
| CSF (Lumbar Puncture) | $1,000 | 4 / 4 |
| DTI MRI | $1,500 | 8 / 11 |
| Logical Memory | $100 | 2 / 4 |

Several expensive panels are triggered by a handful of features, so a trivial greedy
pruner — drop any panel contributing few features, re-score — already captures much of
the available saving. **That greedy/knapsack baseline needs to exist before the QUBO**,
or any improvement the QUBO shows cannot be attributed to the QUBO.

### B7 — Endpoint instruments are charged as input costs

ADAS ($300), CDR ($250) and MMSE ($150) are billed as baseline panels, but those same
instruments *are* the M24 endpoints and must be administered regardless. Whether they are
free (endpoint-mandated) or costed is a modelling decision that should be stated
explicitly; at present it is implicit and inflates the baseline by up to $700.

---

## 4. Documentation, reproducibility, hygiene

- **C1 — The pipeline documentation describes a superseded pipeline.**
  `data/ADNI_Pipeline_Documentation.md` documents n = 852, P = 2,129, output
  `adni_pilot_matrix_benchmark_v2.csv`, and cross-sectional baseline modelling. The work
  actually reported uses the 425-subject longitudinal extraction with M24 targets. The
  doc also states the goal was to "rigidly enforce n = 40% of P" by randomly downsampling
  the cohort to 852 subjects — **deliberately discarding subjects to manufacture P ≫ n is
  not defensible** and should be dropped from the narrative rather than documented as a
  design goal.
- **C2 — Four undocumented data matrices, none marked canonical.**
  `adni_pilot_matrix.csv` (9 rows × 671), `_benchmark` (410 × 1025), `_benchmark_v2`
  (852 × 2131), `_final` (4,381 × 671), plus four missingness reports. `_final` matches
  nothing in the docs and is not used by any script. `data/extract_adni_matrix.R` is dead
  code. `features_list.txt` (2,129 entries) has no stated provenance.
- **C3 — Not runnable as checked in.** `data/extract_adni_longitudinal.R:11` hardcodes
  `C:/Users/User/Desktop/Framework/ADNIMERGE2/ADNIMERGE2/data`. Both benchmarks hardcode
  `DATA_DIR = '../data'`, so they only run from inside their own directory. Directory
  names contain spaces (`benchmark 1 multitask learning`).
- **C4 — No `README.md`, `.gitignore`, `requirements.txt`, or project `LICENSE`.** There
  is no entry point explaining what to run in what order.
- **C5 — A Windows virtualenv is committed.** 1,031 tracked files under
  `benchmark 1 multitask learning/env/`, including 506 `.exe`/`.dll`/`.pyd`/`.pyc`
  binaries. `.git` is **171 MB**.
- **C6 — Participant-level ADNI data is committed.** 217 `.rda` files (90 MB) under
  `ADNIMERGE2/`, plus RID- and SITEID-level derived CSVs (109 MB) under `data/`. The ADNI
  Data Use Agreement restricts redistribution, and the bundled `ADNIMERGE2/LICENSE`
  covers the R package code, not the ADNI data. **Flagged for the author's decision — no
  action taken here.**
- **C7 — Repository name is misspelled:** `Framework-QUBO-feauture-slection` →
  `feature-selection`.

---

## 5. QUBO readiness

**No QUBO code exists in the repository.** The entire stated aim is still ahead. Two
design questions should be settled before implementation:

1. **R² is not a quadratic function of a binary selection vector.** A surrogate objective
   is required — the standard choice is a relevance/redundancy (mRMR-style) form,
   `min_x  -α Σᵢ rᵢxᵢ + β Σᵢ<ⱼ qᵢⱼ xᵢxⱼ`, where `rᵢ` is feature–target association and
   `qᵢⱼ` is feature–feature redundancy. Note that with ~1,300 columns at 60–90%
   missingness, both `r` and `q` will be estimated on small and unequal overlapping
   subsets; that instability needs handling.
2. **Panel costs are group costs, not per-feature costs.** This needs an auxiliary binary
   `p_k` per panel plus linking penalties enforcing `xᵢ ≤ p_k` for every feature `i` in
   panel `k`, and a `Σ_k c_k p_k` term. The linking penalty weight must be large enough
   to be inviolable relative to the accuracy term, which requires a scale analysis of
   `α`, `β`, and the cost coefficients.

Both A2 and A4 feed the QUBO directly: A2 biases `rᵢ` toward the expensive panels, and A4
mis-prices p-tau at $50, so the cost term would be wrong in the direction that most
flatters the result.

---

## 6. Suggested remediation order

Advisory only — none of this was executed.

1. **Data layer** (`data/extract_adni_longitudinal.R`) — deterministic baseline-row
   selection with a one-row-per-RID assertion (A3); pivot long PET tables including FDG
   and compute the SUVR ratio (A5); emit the panel mapping from source-table provenance
   rather than regex (A4); move the 100%-missing column drop after cohort restriction;
   parameterise the data path (C3).
2. **Shared preprocessing module** — the ~50-line load/split/impute block is currently
   duplicated between `mtfl_benchmark.py:19-78` and `xgb_benchmark.py:19-71`. Factor it
   out with typed loading and explicit categorical encoding that fails loudly (A1),
   standard deviations computed on observed values with missing indicators (A2), per-task
   target masking (B3), one shared admin exclusion applied before modelling (A6), and
   repeated k-fold splits (B1).
3. **Benchmarks** — correct the proximal step size, add a convergence check, and select λ
   by CV (B4); align the downstream evaluator so both baselines are refit the same way
   (B2); drive XGBoost's budget from a sweep and let it consume NaN natively (B5).
4. **Economics** — add an FDG PET panel and fix the p-tau assignment (A4, A5); add a
   greedy/knapsack cost-aware baseline as the QUBO's true comparator (B6); state the
   endpoint-instrument cost policy explicitly (B7).
5. **Documentation and hygiene** — rewrite `ADNI_Pipeline_Documentation.md` against the
   pipeline actually in use and restate the summary's figures with confidence intervals
   (C1, B1, B2); add README/`.gitignore`/`requirements.txt` (C4); resolve the committed
   virtualenv and the ADNI data question (C5, C6).
