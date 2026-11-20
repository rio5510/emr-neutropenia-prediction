"""
Cohort construction, outcome labelling and feature generation (Step 2, revision 1).

Changes from the initial submission (reviewer comments in brackets):

* Outcome components are kept as separate columns, and a composite without therapeutic
  G-CSF is added for the sensitivity analysis [R1-M1, R2-M2]:
  ``Event_grade4``, ``Event_fn``, ``Event_gcsf``, ``Event_no_gcsf``.
  ``Event_fn_385`` keeps the FN flag with a 38.5 degC threshold for reference.
* The FN definition is explicit [R2-m5]: ANC < 1.0 x10^9/L and body temperature >= ``th_fever``
  recorded on the same calendar day (daily minimum ANC, daily maximum temperature).
* Every exclusion step is recorded (patients / administration days / reason), so excluded and
  included patients can be compared [R1-M4, R2-M3]. Partial missingness that is imputed later
  (step 3) is reported separately from exclusions [R1-m6].
* Cancer-type flags are built *as of each administration date* (diagnoses registered on or
  before DO_DATE) instead of from the patient's whole history [R2-m3, R2-m4].
* Haematological diagnoses made after a solid-tumour diagnosis are counted, and the handling of
  administration days after such a diagnosis is configurable [R2-m4].

Behaviour intentionally unchanged (pending the authors' decision):
the event window starts on Day 1 (``event_window_start_day``), G-CSF agents are classified as
therapeutic / prophylactic by drug code, and the prophylactic G-CSF feature sums doses given on
the 2 days before DO_DATE (``pre_gcsf_days``; the manuscript says 3 days).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

FILE_NAMES = {
    "exam": "step1_all_sample_rslt.parquet",
    "inj": "step1_all_injection.parquet",
    "drug": "step1_all_drug_taken.parquet",
    "vital": "step1_all_vital.parquet",
    "soma": "step1_all_somatometry.parquet",
    "cancer": "step1_all_cancer.parquet",
}


@dataclass
class CohortConfig:
    gcsf_therapeutic_codes: tuple = ("3399405", "3399406", "3399408")
    gcsf_prophylactic_codes: tuple = ("3399410",)
    exam_items: dict = field(default_factory=lambda: {"  NEUT#": "NEUT", "  WBC": "WBC"})
    heme_pattern: str = r"^C(42|8[1-9]|9[0-6])$"
    th_neut_severe: float = 0.5           # Grade 4 (x10^9/L)
    th_neut_fn: float = 1.0
    th_fever: float = 37.5                # manuscript / JSCO guideline (initial code used 38.5)
    event_window_start_day: int = 1       # Day 1 = DO_DATE (unchanged; see module docstring)
    event_window_end_day: int = 28
    pre_gcsf_days: int = 2                # DO_DATE-2 .. DO_DATE-1 (reproduces the initial code)
    cancer_flag_mode: str = "as_of_date"  # "as_of_date" (revision) | "ever" (initial submission)
    heme_post_dx_policy: str = "include"  # "include" | "exclude_days"
    vital_cols: tuple = ("TEMPR", "PULSE", "RESP", "BPH", "BPL")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# Loading / basic preparation
# --------------------------------------------------------------------------------------
def load_step1(input_dir: str) -> dict[str, pd.DataFrame]:
    """Read the step 1 outputs and fix column types (numerics -> float64, dates -> datetime64).

    Safe to run on files written before step 1 applied ``normalize_types`` (e.g. 2026 extract).
    """
    from .dtypes import normalize_types
    dfs = {}
    for k, fn in FILE_NAMES.items():
        dfs[k] = normalize_types(pd.read_parquet(os.path.join(input_dir, fn)), name=fn)
        print(f"loaded {k:6s} {len(dfs[k]):>12,} rows  <- {fn}")
    return dfs


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def prepare_drugs(dfs, cfg: CohortConfig):
    def _clean(df):
        df = df[df["DRUG_PRICE_LIST_CD"].astype(str).str.strip() != ""].copy()
        df["DRUG_CD"] = df["DRUG_PRICE_LIST_CD"].astype(str).str.slice(0, 7)
        df["DO_DATE"] = _to_date(df["QUERY_DATE"])
        return df.drop(columns=["QUERY_DATE"])

    inj, drug = _clean(dfs["inj"]), _clean(dfs["drug"])
    gcsf_t = inj.loc[inj["DRUG_CD"].isin(cfg.gcsf_therapeutic_codes), ["hs", "DO_DATE", "EXEC_DOSE"]]
    gcsf_p = inj.loc[inj["DRUG_CD"].isin(cfg.gcsf_prophylactic_codes), ["hs", "DO_DATE", "EXEC_DOSE"]]
    chemo = pd.concat([drug, inj])
    chemo = chemo[chemo["ANTI_CANCER_FLG"] == "＋"].copy()
    return (chemo, gcsf_t.rename(columns={"DO_DATE": "GCSF_DATE"}),
            gcsf_p.rename(columns={"DO_DATE": "GLAS_DATE"}))


def prepare_cancer(dfs, cfg: CohortConfig):
    c = dfs["cancer"].copy()
    c["DX_DATE"] = _to_date(c["DIAGNOSIS_DATE_2"])
    c["DX3"] = c["DIAGNOSIS_CD"].astype(str).str.slice(0, 3)
    c["IS_HEME"] = c["DX3"].str.match(cfg.heme_pattern, na=False)
    first = c.groupby("hs")["DX_DATE"].transform("min")
    heme_first = set(c.loc[(c["DX_DATE"] == first) & c["IS_HEME"], "hs"])   # as in the initial code
    return c, heme_first


def daily_exam(dfs, cfg: CohortConfig) -> pd.DataFrame:
    e = dfs["exam"]
    e = e[e["EXAM_ITEM"].isin(list(cfg.exam_items))].copy()
    e["DATE"] = _to_date(e["QUERY_DATE"])
    e["ITEM"] = e["EXAM_ITEM"].map(cfg.exam_items)
    w = e.pivot_table(index=["hs", "DATE"], columns="ITEM", values="RESULT_NUM", aggfunc="min").reset_index()
    for c in cfg.exam_items.values():
        if c not in w:
            w[c] = np.nan
    return w


def daily_vital(dfs) -> pd.DataFrame:
    v = dfs["vital"].copy()
    v["DATE"] = _to_date(v["QUERY_DATE"])
    return v


# --------------------------------------------------------------------------------------
# Outcome labels
# --------------------------------------------------------------------------------------
def make_labels(base, exam_d, vital, gcsf_t, cfg: CohortConfig) -> pd.DataFrame:
    """Composite outcome and its components for each (hs, DO_DATE) in ``base``."""
    bt = vital.groupby(["hs", "DATE"])["TEMPR"].max().rename("maxBT").reset_index()
    daily = exam_d.merge(bt, on=["hs", "DATE"], how="outer")
    daily["g4"] = daily["NEUT"] < cfg.th_neut_severe
    daily["fn"] = (daily["NEUT"] < cfg.th_neut_fn) & (daily["maxBT"] >= cfg.th_fever)
    daily["fn385"] = (daily["NEUT"] < cfg.th_neut_fn) & (daily["maxBT"] >= 38.5)
    daily["lab"] = daily["NEUT"].notna() | daily["WBC"].notna()

    m = base.merge(daily, on="hs", how="inner")
    day = (m["DATE"] - m["DO_DATE"]).dt.days + 1
    m = m[(day >= cfg.event_window_start_day) & (day <= cfg.event_window_end_day)]
    lab = m.groupby(["hs", "DO_DATE"])[["g4", "fn", "fn385", "lab"]].sum()

    g = base.merge(gcsf_t, on="hs", how="inner")
    gday = (g["GCSF_DATE"] - g["DO_DATE"]).dt.days + 1
    g = g[(gday >= cfg.event_window_start_day) & (gday <= cfg.event_window_end_day) & (g["EXEC_DOSE"] > 0)]
    gc = g.groupby(["hs", "DO_DATE"]).size().rename("gcsf")

    out = base.set_index(["hs", "DO_DATE"]).join(lab).join(gc).fillna(0)
    res = pd.DataFrame(index=out.index)
    res["Event_grade4"] = (out["g4"] > 0).astype(int)
    res["Event_fn"] = (out["fn"] > 0).astype(int)
    res["Event_fn_385"] = (out["fn385"] > 0).astype(int)
    res["Event_gcsf"] = (out["gcsf"] > 0).astype(int)
    res["Event_no_gcsf"] = ((res["Event_grade4"] + res["Event_fn"]) > 0).astype(int)
    res["Event"] = ((res["Event_no_gcsf"] + res["Event_gcsf"]) > 0).astype(int)
    res["Y_fu_lab_days"] = out["lab"].astype(int)   # days with ANC/WBC in the window (not a feature)
    return res.reset_index()


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def _asof(base: pd.DataFrame, src: pd.DataFrame, date_col: str, cols: list[str]) -> pd.DataFrame:
    """Latest record on or before DO_DATE (same-day records included, as in the initial code)."""
    src = src.dropna(subset=[date_col]).sort_values(date_col)
    out = pd.merge_asof(base.sort_values("DO_DATE"), src[["hs", date_col] + list(cols)],
                        left_on="DO_DATE", right_on=date_col, by="hs",
                        direction="backward", allow_exact_matches=True)
    return out.drop(columns=[date_col])


def cancer_flags(base, cancer, cfg: CohortConfig) -> pd.DataFrame:
    c = cancer[["hs", "DX3", "DX_DATE"]].drop_duplicates()
    m = base.merge(c, on="hs", how="inner")
    if cfg.cancer_flag_mode == "as_of_date":
        # Diagnoses registered on or before the administration date; undated diagnoses are kept.
        m = m[m["DX_DATE"].isna() | (m["DX_DATE"] <= m["DO_DATE"])]
    elif cfg.cancer_flag_mode != "ever":
        raise ValueError(cfg.cancer_flag_mode)
    w = pd.crosstab([m["hs"], m["DO_DATE"]], m["DX3"]).clip(upper=1)
    w.columns = [f"DIAGNOSIS_{c}" for c in w.columns]
    return w.reset_index()


def make_features(base, dfs, chemo, gcsf_p, exam_d, vital, cancer, cfg: CohortConfig) -> pd.DataFrame:
    # Laboratory: latest date with WBC or ANC (daily minimum)
    ex = exam_d.rename(columns={"NEUT": "lastNEUT#", "WBC": "lastWBC", "DATE": "EX_DATE"}).assign(_has_lab=1)
    f = _asof(base, ex, "EX_DATE", ["lastNEUT#", "lastWBC", "_has_lab"])
    # Age from the chemotherapy record (first value if several on the same day)
    f = f.merge(chemo.groupby(["hs", "DO_DATE"])["AGE"].first().reset_index(), on=["hs", "DO_DATE"], how="left")
    # Somatometry: forward-filled within patient, daily median, latest date
    s = dfs["soma"].copy()
    s["SOMA_DATE"] = _to_date(s["SOMA_DATE"])
    s = s.sort_values(["hs", "SOMA_DATE"])
    s[["MS_HEIGHT", "MS_WEIGHT"]] = s.groupby("hs")[["MS_HEIGHT", "MS_WEIGHT"]].ffill()
    s = s.groupby(["hs", "SOMA_DATE"])[["MS_HEIGHT", "MS_WEIGHT"]].median().reset_index().assign(_has_soma=1)
    f = _asof(f, s, "SOMA_DATE", ["MS_HEIGHT", "MS_WEIGHT", "_has_soma"])
    # Vital signs: latest date, median of that day's records
    vcols = [c for c in cfg.vital_cols if c in vital.columns]
    v = (vital.groupby(["hs", "DATE"])[vcols].median().reset_index()
         .rename(columns={"DATE": "V_DATE"}).assign(_has_vital=1))
    f = _asof(f, v, "V_DATE", vcols + ["_has_vital"])
    # Chemotherapy doses on DO_DATE, by drug code (wide)
    dose = chemo.groupby(["hs", "DO_DATE", "DRUG_CD"])["EXEC_DOSE"].sum().unstack(fill_value=0)
    dose.columns = [str(c) for c in dose.columns]
    f = f.merge(dose.reset_index(), on=["hs", "DO_DATE"], how="left")
    # Prophylactic G-CSF given 1..pre_gcsf_days days before DO_DATE
    gp = base.merge(gcsf_p, on="hs", how="inner")
    dd = (gp["DO_DATE"] - gp["GLAS_DATE"]).dt.days
    gp = gp[(dd >= 1) & (dd <= cfg.pre_gcsf_days)].groupby(["hs", "DO_DATE"])["EXEC_DOSE"].sum()
    f = f.merge(gp.rename("TOTAL_GLAS_PREDOSE").reset_index(), on=["hs", "DO_DATE"], how="left")
    f["TOTAL_GLAS_PREDOSE"] = f["TOTAL_GLAS_PREDOSE"].fillna(0)
    # Sex (M: 0, F: 1)
    sex = dfs["exam"][["hs", "PT_SEX"]].dropna().drop_duplicates("hs")
    f = f.merge(sex.assign(PT_SEX=sex["PT_SEX"].map({"M": 0, "F": 1})), on="hs", how="left")
    # Cancer-type flags
    f = f.merge(cancer_flags(base, cancer, cfg), on=["hs", "DO_DATE"], how="left")
    dx_cols = [c for c in f.columns if c.startswith("DIAGNOSIS_")]
    f[dx_cols] = f[dx_cols].fillna(0).astype(int)
    return f


# --------------------------------------------------------------------------------------
# Haematological diagnoses after a solid tumour (R2-m4)
# --------------------------------------------------------------------------------------
def heme_after_solid(d, cancer, chemo, heme_first) -> tuple[pd.Series, dict]:
    h = cancer[cancer["IS_HEME"] & ~cancer["hs"].isin(heme_first)]
    first_heme = h.groupby("hs")["DX_DATE"].min()
    after = d["hs"].map(first_heme)
    after = after.notna() & (after <= d["DO_DATE"])
    days_after = d.loc[after, ["hs", "DO_DATE"]]
    top = (chemo.merge(days_after, on=["hs", "DO_DATE"])
           .groupby(["DRUG_CD", "COMPONENT_NAME"], dropna=False)
           .agg(administrations=("hs", "size"), patients=("hs", "nunique"))
           .sort_values("administrations", ascending=False).head(20).reset_index())
    info = {"patients_solid_then_heme": int(first_heme.size),
            "heme_codes": {k: int(v) for k, v in h["DX3"].value_counts().items()},
            "days_on_or_after_heme_dx": int(after.sum()),
            "patients_with_days_after_heme_dx": int(days_after["hs"].nunique()),
            "top_drugs_after_heme_dx": top}
    return after, info


# --------------------------------------------------------------------------------------
# Main builder with attrition bookkeeping
# --------------------------------------------------------------------------------------
@dataclass
class CohortResult:
    data: pd.DataFrame
    attrition: pd.DataFrame
    all_days: pd.DataFrame      # every chemotherapy day with labels, basic covariates and exclusion reason
    missingness: pd.DataFrame
    heme: dict


def build_cohort(dfs: dict, cfg: CohortConfig, verbose: bool = True) -> CohortResult:
    chemo, gcsf_t, gcsf_p = prepare_drugs(dfs, cfg)
    cancer, heme_first = prepare_cancer(dfs, cfg)
    exam_d, vital = daily_exam(dfs, cfg), daily_vital(dfs)

    days = chemo[["hs", "DO_DATE"]].dropna().drop_duplicates().reset_index(drop=True)
    labels = make_labels(days, exam_d, vital, gcsf_t, cfg)
    feats = make_features(days, dfs, chemo, gcsf_p, exam_d, vital, cancer, cfg)
    d = days.merge(labels, on=["hs", "DO_DATE"]).merge(feats, on=["hs", "DO_DATE"])
    after_heme, heme_info = heme_after_solid(d, cancer, chemo, heme_first)

    # Sequential exclusions: each day gets the first criterion it meets
    rules = [
        ("not_in_cancer_registry", ~d["hs"].isin(set(cancer["hs"]))),
        ("haematological_malignancy_first_diagnosis", d["hs"].isin(heme_first)),
        ("no_lab_or_vital_records_at_all", ~d["hs"].isin(set(exam_d["hs"]) | set(vital["hs"]))),
        ("no_baseline_lab_WBC_ANC", d["_has_lab"].isna()),
        ("no_baseline_somatometry", d["_has_soma"].isna()),
        ("no_baseline_vital_signs", d["_has_vital"].isna()),
        ("sex_unknown", d["PT_SEX"].isna()),
        ("age_unknown", d["AGE"].isna()),
    ]
    if cfg.heme_post_dx_policy == "exclude_days":
        rules.append(("on_or_after_haematological_diagnosis", after_heme))
    elif cfg.heme_post_dx_policy != "include":
        raise ValueError(cfg.heme_post_dx_policy)

    reason = pd.Series(None, index=d.index, dtype="object")
    att = [{"step": "chemotherapy administration days (anticancer drug flag)", "excluded_days": 0,
            "excluded_patients_entirely": 0, "patients": d["hs"].nunique(), "days": len(d)}]
    for name, mask in rules:
        prev_pts = set(d.loc[reason.isna(), "hs"])
        mask = mask.fillna(False).astype(bool) & reason.isna()
        reason[mask] = name
        keep = reason.isna()
        now_pts = set(d.loc[keep, "hs"])
        att.append({"step": f"exclude: {name}", "excluded_days": int(mask.sum()),
                    "excluded_patients_entirely": len(prev_pts - now_pts),
                    "patients": len(now_pts), "days": int(keep.sum())})
    attrition = pd.DataFrame(att)
    d["exclusion_reason"] = reason
    d["after_heme_dx"] = after_heme.astype(int)

    data = d[reason.isna()].drop(columns=["exclusion_reason", "_has_lab", "_has_soma", "_has_vital"])
    if cfg.heme_post_dx_policy == "exclude_days":
        data = data.drop(columns=["after_heme_dx"])
    else:   # keep as a label-type column (excluded from features by the "Y_" prefix)
        data = data.rename(columns={"after_heme_dx": "Y_after_heme_dx"})
    dx = [c for c in data.columns if c.startswith("DIAGNOSIS_")]
    data = data.drop(columns=[c for c in dx if data[c].sum() == 0])   # flags never present in the cohort
    drug_cols = [c for c in data.columns if str(c).isdigit()]
    data[drug_cols] = data[drug_cols].fillna(0)
    data = data.sort_values(["hs", "DO_DATE"]).reset_index(drop=True)

    miss = data.drop(columns=["hs", "DO_DATE"]).isna().sum()
    missingness = (pd.DataFrame({"n_missing": miss, "pct_missing": miss / len(data)})
                   .query("n_missing > 0").sort_values("n_missing", ascending=False))
    keep_cols = ["hs", "DO_DATE", "exclusion_reason", "after_heme_dx", "AGE", "PT_SEX",
                 "MS_HEIGHT", "MS_WEIGHT", "lastWBC", "lastNEUT#", "Y_fu_lab_days"] + \
                [c for c in d.columns if c.startswith("Event")]
    all_days = d[keep_cols].copy()
    if verbose:
        print(attrition.to_string(index=False))
        print(f"\nfinal: patients={data['hs'].nunique():,} days={len(data):,} event={data['Event'].mean():.2%}")
        print("partially missing (imputed in step 3):", missingness["n_missing"].to_dict())
    return CohortResult(data, attrition, all_days, missingness, heme_info)


def compare_excluded_included(all_days: pd.DataFrame) -> pd.DataFrame:
    """Patient-level comparison: patients with >=1 included day vs patients entirely excluded (R1-M4, R2-M3)."""
    from .stats import table1_smd
    a = all_days.copy()
    a["included"] = a["exclusion_reason"].isna()
    pt = a.groupby("hs").agg(included=("included", "max"), AGE=("AGE", "first"), Female=("PT_SEX", "first"),
                             n_days=("DO_DATE", "size"), any_event=("Event", "max"),
                             any_event_no_gcsf=("Event_no_gcsf", "max"),
                             fu_lab_days_mean=("Y_fu_lab_days", "mean")).reset_index()
    inc, exc = pt[pt["included"]], pt[~pt["included"]]
    t = table1_smd(inc, exc, ["AGE", "n_days", "fu_lab_days_mean"], ["Female", "any_event", "any_event_no_gcsf"])
    t = t.rename(columns={"train": "included", "test": "excluded"})
    t.loc[len(t)] = {"variable": "patients", "included": f"{len(inc):,}", "excluded": f"{len(exc):,}", "abs_SMD": np.nan}
    return t


def save_cohort(res: CohortResult, cfg: CohortConfig, out_dir: str, name: str = "step2_all_tagt_feat") -> None:
    os.makedirs(out_dir, exist_ok=True)
    res.data.to_parquet(os.path.join(out_dir, name))
    res.attrition.to_csv(os.path.join(out_dir, "attrition.csv"), index=False)
    res.all_days.to_parquet(os.path.join(out_dir, "all_chemo_days.parquet"))
    res.missingness.to_csv(os.path.join(out_dir, "missingness_included.csv"))
    res.heme["top_drugs_after_heme_dx"].to_csv(os.path.join(out_dir, "heme_after_solid_top_drugs.csv"), index=False)
    with open(os.path.join(out_dir, "heme_after_solid_summary.json"), "w") as f:
        json.dump({k: v for k, v in res.heme.items() if k != "top_drugs_after_heme_dx"}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "cohort_config.json"), "w") as f:
        f.write(cfg.to_json())
