# Revision 1 (ABE ES26-029): code changes for the reviewer response

`main` keeps the code used for the initial submission. This branch (`revision-1`) holds the changes made for the revision.

## Layout
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
| WP5 Outcome components, no-G-CSF sensitivity | R1-M1, R2-M2, R2-m5 | step3 and step4 ready; **step2 must output `Event_*` columns** | step2 (to do) |
| WP3 Missing data / exclusions | R1-M4, R2-M3, R1-m6 | To do | step2 |
| WP8 Cancer-code timing, haematological cases | R2-m3, R2-m4 | To do | step2 |

## Expected step2 output (for WP5)
Besides `Event`, step2 should write these 0/1 columns. They are excluded from the features automatically, because any column whose name starts with `Event` is treated as a label.
- `Event_grade4`
- `Event_fn`
- `Event_gcsf`
- `Event_no_gcsf` (= `Event_grade4` OR `Event_fn`)

## Runtime notes (hospital JupyterLab, 64 cores)
- Default parallelism: `n_jobs_search=8` × `n_jobs_model=8`.
- Default search: `n_iter=30` per algorithm, 5-fold patient-level inner CV.
- Bootstrap: `N_BOOT=2000` in step4.
