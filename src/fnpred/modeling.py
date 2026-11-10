"""
Model development utilities (Step 3, revision 1).

Changes from the initial-submission pipeline (reviewer comments R1-M3, R2-M5, R2-m1, R2-m6):

* The held-out test set is never used during model development. Early stopping on the
  test set (initial version) has been removed; the number of boosting rounds is tuned
  like any other hyperparameter.
* Hyperparameters are selected by randomized search with patient-level
  StratifiedGroupKFold cross-validation inside the training set only.
* Pre-processing (median imputation, standardization) lives inside an sklearn Pipeline,
  so it is fitted on training folds only.
* The same class-imbalance weight is applied to all three algorithms
  (RF: class_weight={0: 1, 1: w}; XGBoost/LightGBM: scale_pos_weight=w).
* Out-of-fold (OOF) predictions on the training set are stored so that classification
  thresholds can be chosen without looking at the test set.
* Two split modes: patient-level random split ("group", reproduces the original 80:20 split)
  and temporal split ("temporal", R2-m6).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COL = "hs"
DATE_COL = "DO_DATE"
DEFAULT_TARGET = "Event"
# Columns that are labels / outcome components and must never be used as features.
NON_FEATURE_PREFIXES = ("Event", "Y_")
NON_FEATURE_COLS = {ID_COL, DATE_COL, "Valid_FN", "Therapeutic_GCSF_Event"}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    target: str = DEFAULT_TARGET
    split_mode: str = "group"            # "group" | "temporal"
    test_fold_seed: int = 42             # seed of the outer StratifiedGroupKFold (original: 42)
    n_outer_splits: int = 5              # 5 -> 80:20 split
    temporal_cutoff: str = "2023-01-01"  # train: DO_DATE < cutoff, test: DO_DATE >= cutoff
    temporal_drop_overlap: bool = True   # drop test-period records of patients seen in training
    n_inner_splits: int = 5              # patient-level CV inside the training set
    n_iter: int = 30                     # randomized-search iterations per algorithm
    search_seed: int = 2026
    scoring: str = "roc_auc"
    class_weight_mode: str = "sqrt"      # "none" | "sqrt" | "balanced"  (w = 1, sqrt(neg/pos), neg/pos)
    n_jobs_model: int = 8                # threads per model fit
    n_jobs_search: int = 8               # parallel CV fits (8 x 8 = 64 cores)
    algorithms: tuple = ("rf", "xgb", "lgb")
    recalibrate: bool = True             # logistic recalibration fitted on training OOF predictions
    fixed_thresholds: tuple = (0.10, 0.20)   # guideline FN-risk thresholds (ASCO / JSCO)
    target_sensitivity: float = 0.80

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not str(c).startswith(NON_FEATURE_PREFIXES)]


def split_train_test(df: pd.DataFrame, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (is_train, is_test) over the rows of ``df``.

    Rows that are in neither set (temporal mode with ``temporal_drop_overlap``) are excluded.
    """
    y = df[cfg.target].to_numpy()
    groups = df[ID_COL].to_numpy()

    if cfg.split_mode == "group":
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_outer_splits, shuffle=True,
                                    random_state=cfg.test_fold_seed)
        tr_idx, _ = next(sgkf.split(np.zeros(len(df)), y, groups=groups))
        is_train = np.zeros(len(df), bool)
        is_train[tr_idx] = True
        is_test = ~is_train
    elif cfg.split_mode == "temporal":
        dates = pd.to_datetime(df[DATE_COL])
        is_train = (dates < pd.Timestamp(cfg.temporal_cutoff)).to_numpy()
        is_test = ~is_train
        if cfg.temporal_drop_overlap:
            is_test &= ~df[ID_COL].isin(set(df.loc[is_train, ID_COL])).to_numpy()
    else:
        raise ValueError(f"unknown split_mode: {cfg.split_mode}")

    overlap = set(df.loc[is_train, ID_COL]) & set(df.loc[is_test, ID_COL])
    if overlap and not (cfg.split_mode == "temporal" and not cfg.temporal_drop_overlap):
        raise AssertionError(f"{len(overlap)} patients appear in both training and test sets")
    return is_train, is_test


def positive_class_weight(y: np.ndarray, mode: str) -> float:
    neg, pos = np.sum(y == 0), np.sum(y == 1)
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(neg / pos))
    if mode == "balanced":
        return float(neg / pos)
    raise ValueError(mode)


# --------------------------------------------------------------------------------------
# Models and search spaces
# --------------------------------------------------------------------------------------
def _preprocess_steps():
    # Median imputation is fitted on the training folds only; columns without missing values
    # are unaffected. Scaling is kept for compatibility with the original pipeline
    # (tree models are invariant to it).
    return [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]


def build_pipeline(algo: str, w_pos: float, seed: int, n_jobs: int) -> Pipeline:
    if algo == "rf":
        clf = RandomForestClassifier(class_weight={0: 1.0, 1: w_pos}, random_state=seed, n_jobs=n_jobs)
    elif algo == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(scale_pos_weight=w_pos, random_state=seed, n_jobs=n_jobs,
                            eval_metric="logloss", tree_method="hist")
    elif algo == "lgb":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(scale_pos_weight=w_pos, random_state=seed, n_jobs=n_jobs,
                             subsample_freq=1, verbosity=-1)
    else:
        raise ValueError(algo)
    return Pipeline(_preprocess_steps() + [("clf", clf)])


SEARCH_SPACES: dict[str, dict[str, list]] = {
    "rf": {
        "clf__n_estimators": [200, 500, 800],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_leaf": [1, 5, 10, 20],
        "clf__max_features": ["sqrt", 0.2, 0.4],
    },
    "xgb": {
        "clf__n_estimators": [200, 400, 800, 1200],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__min_child_weight": [1, 5, 10],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [1, 5, 10],
    },
    "lgb": {
        "clf__n_estimators": [200, 400, 800, 1200],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__num_leaves": [15, 31, 63],
        "clf__max_depth": [-1, 5, 7, 9],
        "clf__min_child_samples": [5, 20, 50],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0, 1, 5, 10],
    },
}


# --------------------------------------------------------------------------------------
# Recalibration (class weighting shifts predicted probabilities upwards; R1-M6 / R2-M4)
# --------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class LogisticRecalibrator:
    """p_recal = expit(a + b * logit(p)); a, b fitted on training OOF predictions only."""
    a: float = 0.0
    b: float = 1.0

    def fit(self, y, p):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p).reshape(-1, 1), y)
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def transform(self, p):
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * _logit(p))))


# --------------------------------------------------------------------------------------
# Thresholds (chosen on training OOF predictions only)
# --------------------------------------------------------------------------------------
def choose_thresholds(y: np.ndarray, p: np.ndarray, cfg: TrainConfig) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    out = {f"fixed_{t:.2f}": float(t) for t in cfg.fixed_thresholds}
    out["youden_oof"] = float(thr[np.argmax(tpr - fpr)])
    ok = np.where(tpr >= cfg.target_sensitivity)[0]
    out[f"sens{int(cfg.target_sensitivity * 100)}_oof"] = float(thr[ok[0]]) if len(ok) else float("nan")
    out["default_0.50"] = 0.5
    return out


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
@dataclass
class TrainResult:
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    best_params: pd.DataFrame | None = None
    cv_results: dict = field(default_factory=dict)
    oof: pd.DataFrame | None = None
    test_pred: pd.DataFrame | None = None
    thresholds: dict = field(default_factory=dict)
    recalibrators: dict = field(default_factory=dict)
    split: pd.DataFrame | None = None
    class_weight: float = 1.0


def train_all(df: pd.DataFrame, cfg: TrainConfig, verbose: bool = True) -> TrainResult:
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)
    feats = feature_columns(df)
    is_train, is_test = split_train_test(df, cfg)

    X_tr = df.loc[is_train, feats]
    y_tr = df.loc[is_train, cfg.target].to_numpy().astype(int)
    g_tr = df.loc[is_train, ID_COL].to_numpy()
    X_te = df.loc[is_test, feats]
    y_te = df.loc[is_test, cfg.target].to_numpy().astype(int)
    w_pos = positive_class_weight(y_tr, cfg.class_weight_mode)
    if verbose:
        print(f"[split={cfg.split_mode}, target={cfg.target}] "
              f"train: rows={is_train.sum():,} patients={len(set(g_tr)):,} event={y_tr.mean():.2%} | "
              f"test: rows={is_test.sum():,} patients={df.loc[is_test, ID_COL].nunique():,} "
              f"event={y_te.mean():.2%} | features={len(feats)} | w_pos={w_pos:.3f}")

    inner_cv = StratifiedGroupKFold(n_splits=cfg.n_inner_splits, shuffle=True, random_state=cfg.search_seed)
    res = TrainResult(class_weight=w_pos, features=feats)
    rows, oof, test_pred = [], {}, {}
    for algo in cfg.algorithms:
        pipe = build_pipeline(algo, w_pos, cfg.search_seed, cfg.n_jobs_model)
        search = RandomizedSearchCV(pipe, SEARCH_SPACES[algo], n_iter=cfg.n_iter, scoring=cfg.scoring,
                                    cv=inner_cv, refit=True, random_state=cfg.search_seed,
                                    n_jobs=cfg.n_jobs_search, error_score="raise")
        search.fit(X_tr, y_tr, groups=g_tr)
        best = search.best_estimator_
        # OOF predictions of the selected configuration (used for threshold selection only)
        oof[algo] = cross_val_predict(clone(best), X_tr, y_tr, groups=g_tr, cv=inner_cv,
                                      method="predict_proba", n_jobs=cfg.n_jobs_search)[:, 1]
        test_pred[algo] = best.predict_proba(X_te)[:, 1]
        res.models[algo] = best
        res.cv_results[algo] = pd.DataFrame(search.cv_results_)
        res.thresholds[algo] = choose_thresholds(y_tr, oof[algo], cfg)
        if cfg.recalibrate:
            cal = LogisticRecalibrator().fit(y_tr, oof[algo])
            res.recalibrators[algo] = cal
            oof[f"{algo}_recal"] = cal.transform(oof[algo])
            test_pred[f"{algo}_recal"] = cal.transform(test_pred[algo])
            res.thresholds[f"{algo}_recal"] = choose_thresholds(y_tr, oof[f"{algo}_recal"], cfg)
        rows.append({"algorithm": algo, "cv_auroc_mean": search.best_score_,
                     "cv_auroc_sd": float(search.cv_results_["std_test_score"][search.best_index_]),
                     **{k.replace("clf__", ""): v for k, v in search.best_params_.items()}})
        if verbose:
            print(f"  {algo}: inner-CV AUROC={search.best_score_:.4f}  best={search.best_params_}")

    res.best_params = pd.DataFrame(rows)
    keys_tr = df.loc[is_train, [ID_COL, DATE_COL, cfg.target]].reset_index(drop=True)
    res.oof = pd.concat([keys_tr, pd.DataFrame(oof)], axis=1)
    keys_te = df.loc[is_test, [ID_COL, DATE_COL, cfg.target]].reset_index(drop=True)
    res.test_pred = pd.concat([keys_te, pd.DataFrame(test_pred)], axis=1)
    res.split = df[[ID_COL, DATE_COL]].assign(
        set=np.where(is_train, "train", np.where(is_test, "test", "excluded")))
    return res


def save_result(res: TrainResult, cfg: TrainConfig, out_dir: str) -> None:
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    for algo, m in res.models.items():
        joblib.dump(m, os.path.join(out_dir, f"model_{algo}.joblib"))
    res.best_params.to_csv(os.path.join(out_dir, "best_params.csv"), index=False)
    for algo, cvr in res.cv_results.items():
        cvr.to_csv(os.path.join(out_dir, f"cv_results_{algo}.csv"), index=False)
    res.oof.to_parquet(os.path.join(out_dir, "oof_pred_train.parquet"))
    res.test_pred.to_parquet(os.path.join(out_dir, "pred_test.parquet"))
    res.split.to_parquet(os.path.join(out_dir, "split.parquet"))
    with open(os.path.join(out_dir, "features.json"), "w") as f:
        json.dump(res.features, f, indent=0, ensure_ascii=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(res.thresholds, f, indent=2)
    with open(os.path.join(out_dir, "recalibrators.json"), "w") as f:
        json.dump({k: asdict(v) for k, v in res.recalibrators.items()}, f, indent=2)
    with open(os.path.join(out_dir, "search_space.json"), "w") as f:
        json.dump(SEARCH_SPACES, f, indent=2, default=str)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())
    with open(os.path.join(out_dir, "class_weight.json"), "w") as f:
        json.dump({"mode": cfg.class_weight_mode, "w_pos": res.class_weight}, f, indent=2)
