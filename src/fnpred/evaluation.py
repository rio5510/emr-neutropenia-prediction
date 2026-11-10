"""
Evaluation utilities (Step 4, revision 1).

Reviewer comments addressed:
* R1-M5 / R2-M1 : 95% CIs from a patient-level cluster bootstrap (patients are the resampling unit).
* R1-M6 / R2-M4 : calibration (decile plot + LOWESS, calibration intercept and slope),
                  threshold-based sensitivity / specificity / PPV / NPV, decision curve analysis.
* R1-M3(c) / R2-M5 : threshold-dependent metrics are reported for explicitly stated thresholds.
* R1-m7 : ROC curves of all models overlaid in one panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

EPS = 1e-12
LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lgb": "LightGBM",
          "rf_recal": "Random Forest (recalibrated)", "xgb_recal": "XGBoost (recalibrated)",
          "lgb_recal": "LightGBM (recalibrated)"}
STYLES = {"rf": dict(color="#1f77b4", ls="-"), "xgb": dict(color="#d62728", ls="--"),
          "lgb": dict(color="#2ca02c", ls="-.")}
STYLES.update({f"{k}_recal": v for k, v in list(STYLES.items())})


# --------------------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope.

    Ideal values: intercept 0, slope 1.
    """
    import statsmodels.api as sm
    lp = _logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(intercept), float(slope)


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    sens = tp / (tp + fn + EPS); spec = tn / (tn + fp + EPS)
    ppv = tp / (tp + fp + EPS); npv = tn / (tn + fn + EPS)
    return {"threshold": t, "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "accuracy": (tp + tn) / len(y), "f1": 2 * ppv * sens / (ppv + sens + EPS),
            "flagged_rate": pred.mean(), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def point_metrics(y: np.ndarray, p: np.ndarray, with_calibration: bool = True) -> dict[str, float]:
    out = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
           "brier": brier_score_loss(y, p), "prevalence": y.mean(), "mean_pred": p.mean()}
    if with_calibration:
        out["cal_intercept"], out["cal_slope"] = calibration_intercept_slope(y, p)
    return out


# --------------------------------------------------------------------------------------
# Patient-level cluster bootstrap
# --------------------------------------------------------------------------------------
_CI_THRESHOLD_KEYS = ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      thresholds: dict[str, float] | None = None, n_boot: int = 2000,
                      seed: int = 2026, alpha: float = 0.05, with_calibration: bool = True) -> pd.DataFrame:
    """Percentile CIs, resampling patients (not administration days) with replacement."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(codes))]
    rows_of = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]

    def stats(idx):
        yy, pp = y[idx], p[idx]
        d = point_metrics(yy, pp, with_calibration)
        for name, t in (thresholds or {}).items():
            tm = threshold_metrics(yy, pp, t)
            d.update({f"{k}@{name}": tm[k] for k in _CI_THRESHOLD_KEYS})
        return d

    est = stats(np.arange(len(y)))
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        boots.append(stats(idx))
    B = pd.DataFrame(boots)
    return pd.DataFrame({"estimate": pd.Series(est), "ci_low": B.quantile(alpha / 2),
                         "ci_high": B.quantile(1 - alpha / 2), "n_boot": len(B)})


def fmt_ci(row, digits: int = 3) -> str:
    return f"{row['estimate']:.{digits}f} ({row['ci_low']:.{digits}f}–{row['ci_high']:.{digits}f})"


# --------------------------------------------------------------------------------------
# Decision curve analysis
# --------------------------------------------------------------------------------------
def decision_curve(y: np.ndarray, p: np.ndarray, thresholds=np.arange(0.01, 0.51, 0.01)) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        w = t / (1 - t)
        rows.append({"threshold": t, "nb_model": tp / n - fp / n * w,
                     "nb_treat_all": prev - (1 - prev) * w, "nb_treat_none": 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_roc_overlay(y, preds: dict, ci: dict | None = None, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    for algo, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        lab = LABELS.get(algo, algo)
        if ci is not None:
            lab += f": AUROC {fmt_ci(ci[algo].loc['auroc'])}"
        ax.plot(fpr, tpr, lw=1.6, label=lab, **STYLES.get(algo, {}))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return ax


def plot_calibration(y, preds: dict, n_bins: int = 10, max_p: float | None = None):
    """Decile calibration points + LOWESS curve, with a predicted-risk histogram below."""
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = np.asarray(y).astype(int)
    fig = plt.figure(figsize=(5.5, 6.5))
    ax = fig.add_axes([0.15, 0.32, 0.8, 0.62])
    axh = fig.add_axes([0.15, 0.08, 0.8, 0.17], sharex=ax)
    hi = max_p or float(min(1.0, np.quantile(np.concatenate(list(preds.values())), 0.995) * 1.05))
    for algo, p in preds.items():
        st = STYLES.get(algo, {})
        q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
        g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").mean()
        ax.plot(g["p"], g["y"], "o", ms=4, color=st.get("color"))
        sm = lowess(y, p, frac=0.3, it=0, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], lw=1.5, label=LABELS.get(algo, algo), **st)
        axh.hist(p, bins=50, range=(0, hi), histtype="step", color=st.get("color"))
    ax.plot([0, hi], [0, hi], color="grey", ls=":", lw=1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_ylabel("Observed event proportion"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    axh.set_xlabel("Predicted probability"); axh.set_ylabel("Count"); axh.set_yticks([])
    plt.setp(ax.get_xticklabels(), visible=False)
    return fig


def plot_decision_curve(y, preds: dict, thresholds=np.arange(0.01, 0.51, 0.01), ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    base = None
    for algo, p in preds.items():
        d = decision_curve(y, p, thresholds)
        ax.plot(d["threshold"], d["nb_model"], lw=1.6, label=LABELS.get(algo, algo), **STYLES.get(algo, {}))
        base = d
    ax.plot(base["threshold"], base["nb_treat_all"], color="black", lw=1, label="Treat all")
    ax.axhline(0, color="grey", lw=1, label="Treat none")
    ax.set_ylim(-0.02, max(float(np.mean(y)) * 1.1, 0.01))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.legend(fontsize=8, frameon=False)
    return ax
