"""
Secondary (exploratory) association analyses accounting for within-patient clustering
(reviewer comments R1-M5, R2-M1, R1-m4).

Each univariate logistic regression is fitted three ways:
* naive GLM (independence; as in the initial submission, for comparison);
* GLM with cluster-robust (patient-level) standard errors;
* GEE with an exchangeable working correlation (patient as cluster).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def du_bois_bsa(height_cm, weight_kg):
    """Body surface area (m^2), Du Bois & Du Bois (1916)."""
    return 0.007184 * np.power(height_cm, 0.725) * np.power(weight_kg, 0.425)


def clustered_logit(df: pd.DataFrame, y: str, x: str, group: str, per: float = 1.0) -> pd.DataFrame:
    """Odds ratio per ``per`` units of ``x`` with 95% CI and p-value under three variance models."""
    import statsmodels.api as sm
    d = df[[y, x, group]].dropna().sort_values(group).reset_index(drop=True)
    X = sm.add_constant(d[x] / per)
    Y = d[y].astype(float)
    g = pd.factorize(d[group])[0]
    fits = {
        "naive": sm.GLM(Y, X, family=sm.families.Binomial()).fit(),
        "cluster_robust": sm.GLM(Y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": g}),
        "gee_exchangeable": sm.GEE(Y, X, groups=g, family=sm.families.Binomial(),
                                   cov_struct=sm.cov_struct.Exchangeable()).fit(),
    }
    rows = []
    for name, f in fits.items():
        b = f.params.iloc[1]
        lo, hi = f.conf_int().iloc[1]
        rows.append({"variable": x, "per": per, "model": name, "OR": np.exp(b),
                     "ci_low": np.exp(lo), "ci_high": np.exp(hi), "p": f.pvalues.iloc[1],
                     "n_obs": len(d), "n_patients": d[group].nunique()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Table 1: training vs. test comparison with standardized mean differences (R1-m2)
# --------------------------------------------------------------------------------------
def smd_continuous(a, b) -> float:
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def smd_binary(a, b) -> float:
    p1, p2 = pd.Series(a).dropna().mean(), pd.Series(b).dropna().mean()
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / sd) if sd > 0 else 0.0


def table1_smd(train: pd.DataFrame, test: pd.DataFrame, continuous: list[str], binary: list[str]) -> pd.DataFrame:
    """Mean ± SD / n (%) by set, with absolute SMD (< 0.1 is conventionally taken as negligible)."""
    rows = []
    for c in continuous:
        rows.append({"variable": c,
                     "train": f"{train[c].mean():.1f} ± {train[c].std():.1f}",
                     "test": f"{test[c].mean():.1f} ± {test[c].std():.1f}",
                     "abs_SMD": abs(smd_continuous(train[c], test[c]))})
    for c in binary:
        rows.append({"variable": c,
                     "train": f"{int(train[c].sum()):,} ({train[c].mean():.1%})",
                     "test": f"{int(test[c].sum()):,} ({test[c].mean():.1%})",
                     "abs_SMD": abs(smd_binary(train[c], test[c]))})
    return pd.DataFrame(rows)
