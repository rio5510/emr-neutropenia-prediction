# Revision 1 (ABE ES26-029): code changes for the reviewer response

`main` keeps the code used for the initial submission. This branch (`revision-1`) holds the changes made for the revision.

## Layout
- `src/fnpred/cohort.py`: cohort, outcome labels, features, attrition (Step 2)
- `src/fnpred/modeling.py`: model development (Step 3)
- `src/fnpred/evaluation.py`: bootstrap CIs, calibration, decision curves, figures (Step 4)
- `src/fnpred/stats.py`: secondary analyses with clustering, Table 1 SMD
- The notebooks add `../src` to `sys.path`, so nothing has to be installed.

## Status
| WP | Comments | Status | Where |
|---|---|---|---|
| WP4 Model development | R1-M3, R2-M5, R2-m1 | Implemented | step3, `modeling.py` |
| WP1 Clustering | R1-M5, R2-M1 | Implemented | step4, `evaluation.cluster_bootstrap`, `stats.clustered_logit` |
| WP2 Calibration / utility | R1-M6, R2-M4 | Implemented | step4, `evaluation.py`, OOF recalibration in `modeling.py` |
| WP6 Temporal validation | R2-m6, R1-M7 | Implemented (`EXPERIMENTS["temporal"]`) | step3 / step4 |
| WP9 Figures | R1-m7 (ROC overlay) | Implemented | step4 |
| WP10 Table 1 | R1-m2 (SMD), R2-m2 (patients with ≥1 event) | Implemented | step4 |
| WP5 Outcome components, no-G-CSF sensitivity | R1-M1, R2-M2, R2-m5 | Implemented (`Event_*` columns, FN 37.5 °C, same-day rule) | step2, `cohort.py` |
| WP3 Missing data / exclusions | R1-M4, R2-M3, R1-m6 | Implemented (attrition with reasons, excluded vs included, partial missingness) | step2, `cohort.py` |
| Baseline labs / vital signs | — | Implemented (`lab_mode="per_item"`: latest WBC and latest ANC taken separately, so ANC is not lost when the latest lab day has no differential; `daysLastWBC` / `daysLastNEUT#` added as features; vital signs likewise with `vital_mode="per_item"` and `daysLast<vital>`; same-day (pre-administration) labs stay as features (`lab_feature_same_day=True`) and the outcome window starts on Day 2 (`event_window_start_day=2`); Day-1 events are kept as `Y_day1_*`) | step2, `cohort.py` |
| WP8 Cancer-code timing, haematological cases | R2-m3, R2-m4 | Implemented (flags only for cancer types with DIAGNOSIS_DATE_2 on or before DO_DATE; undated diagnoses not flagged; counts of haematological dx after solid tumour; administration days on or after that diagnosis excluded, `heme_post_dx_policy="exclude_days"`) | step2, `cohort.py` |

## Kept from the initial code (authors' decision, 2026-11-25)
- Therapeutic / prophylactic G-CSF is classified by drug code (current definition kept).
- The prophylactic G-CSF window is 2 days (`pre_gcsf_days`); the manuscript says 3 days.

## Reproducing the initial submission
`CohortConfig(th_fever=38.5, cancer_flag_mode="ever", lab_mode="same_day", vital_mode="same_day", lab_feature_same_day=True, heme_post_dx_policy="include", event_window_start_day=1, undated_dx_flag=True)` gives the same rows and values as the initial step2 (checked on synthetic data).

## step2 outcome columns
Besides `Event`, step2 should write these 0/1 columns. They are excluded from the features automatically, because any column whose name starts with `Event` is treated as a label.
- `Event_grade4`
- `Event_fn`
- `Event_gcsf`
- `Event_no_gcsf` (= `Event_grade4` OR `Event_fn`)

## Runtime notes (hospital JupyterLab, 64 cores)
- Default parallelism: `n_jobs_search=8` × `n_jobs_model=8`.
- Default search: `n_iter=30` per algorithm, 5-fold patient-level inner CV.
- Bootstrap: `N_BOOT=2000` in step4.
