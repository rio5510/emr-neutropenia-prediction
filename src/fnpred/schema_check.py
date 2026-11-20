"""
Diagnose differences between two DWH extracts used by step 1
(e.g. ``p1_pts_parse_whole_spilt_2025`` vs ``p1_pts_parse_whole_spilt_2026``).

Only Parquet metadata is read for the schema comparison, so it is fast even for large extracts.
Values are read only for the columns that step 1 / step 2 depend on, and only a sample of rows.

Typical causes of the errors seen after switching extracts:
* a column changes Arrow type between extracts (string -> int64, timestamp -> string, ...);
* a column has *different types across batch files* (d00, d01, ...) within one extract.
  ``pd.concat`` then yields an ``object`` column, and ``to_parquet`` fails
  ("ArrowTypeError: Expected bytes, got a 'int' object");
* timestamp unit / time zone / INT96 storage changes (ns vs us vs ms, tz-aware vs naive);
* value conventions change ('0' vs 0 for STOP_FLG, full-width '＋' vs '+', leading spaces in EXAM_ITEM,
  date strings 'YYYYMMDD' vs 'YYYY-MM-DD', leading zeros lost in codes);
* a column is missing or renamed, or a batch has an all-null column typed ``null``.
"""
from __future__ import annotations

import glob
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Tables read by the step 1 notebooks and the columns they select
TABLES: dict[str, list[str]] = {
    "CANCER_REGISTRATION": ["hs", "PT_ID", "DIAGNOSIS_CD", "DIAGNOSIS_TEXT", "UICC_STAGE_BEFORE",
                            "MAIN5_STAGE_BEFORE", "UICC_STAGE_AFTER", "ICD_O_3_CD", "ICD_O_3_TEXT",
                            "DIAGNOSIS_DATE_1", "DIAGNOSIS_DATE_2", "AGE"],
    "MDV_DRUG_INSTRUCTION": ["hs", "AGE", "QUERY_DATE", "COMPONENT_NAME", "DRUG_PRICE_LIST_CD",
                             "ORDER_DOSE", "ORDER_UNIT", "ANTI_CANCER_FLG", "STOP_FLG"],
    "MDV_INJECTION": ["hs", "AGE", "QUERY_DATE", "COMPONENT_NAME", "DRUG_PRICE_LIST_CD",
                      "EXEC_DOSE", "EXEC_UNIT", "ANTI_CANCER_FLG"],
    "MDV_SAMPLE_RSLT": ["hs", "PT_SEX", "AGE", "QUERY_DATE", "EXAM_ITEM_CD", "EXAM_ITEM",
                        "RESULT_NUM", "RESULT_UNIT"],
    "MDV_SOMATOMETRY": ["hs", "AGE", "SOMA_DATE", "MS_HEIGHT", "MS_WEIGHT", "MS_BSA"],
    "MDV_VITAL": ["hs", "AGE", "QUERY_DATE", "TEMPR", "PULSE", "HEART_RATE", "RESP", "BPH", "BPL"],
}

# Value conventions that the downstream code relies on: column -> (description, check function)
def _is_num(s):
    return pd.to_numeric(s, errors="coerce").notna()


def _is_date(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.notna()
    return pd.to_datetime(s.astype("string"), errors="coerce", format="mixed").notna()


EXPECTATIONS: dict[str, dict[str, tuple]] = {
    "MDV_DRUG_INSTRUCTION": {
        "STOP_FLG": ("== '0' (string) is used to keep active orders", lambda s: s.astype("string") == "0",
                     lambda s: s == "0"),
        "ANTI_CANCER_FLG": ("== '＋' (full-width plus) marks anticancer drugs",
                            lambda s: s.astype("string") == "＋", lambda s: s == "＋"),
        "DRUG_PRICE_LIST_CD": ("string; first 7 characters = drug code",
                               lambda s: s.astype("string").str.len() >= 7, None),
        "QUERY_DATE": ("date / timestamp", _is_date, None),
        "ORDER_DOSE": ("numeric", _is_num, None),
    },
    "MDV_INJECTION": {
        "ANTI_CANCER_FLG": ("== '＋' (full-width plus) marks anticancer drugs",
                            lambda s: s.astype("string") == "＋", lambda s: s == "＋"),
        "DRUG_PRICE_LIST_CD": ("string; first 7 characters = drug code",
                               lambda s: s.astype("string").str.len() >= 7, None),
        "QUERY_DATE": ("date / timestamp", _is_date, None),
        "EXEC_DOSE": ("numeric", _is_num, None),
    },
    "MDV_SAMPLE_RSLT": {
        "EXAM_ITEM": ("contains '  NEUT#' / '  WBC' (two leading spaces)",
                      lambda s: s.astype("string").isin(["  NEUT#", "  WBC"]), None),
        "RESULT_NUM": ("numeric", _is_num, None),
        "PT_SEX": ("'M' / 'F'", lambda s: s.astype("string").isin(["M", "F"]), lambda s: s.isin(["M", "F"])),
        "QUERY_DATE": ("date / timestamp", _is_date, None),
    },
    "MDV_SOMATOMETRY": {
        "SOMA_DATE": ("date / timestamp", _is_date, None),
        "MS_HEIGHT": ("numeric", _is_num, None),
        "MS_WEIGHT": ("numeric", _is_num, None),
    },
    "MDV_VITAL": {
        "QUERY_DATE": ("date / timestamp", _is_date, None),
        "TEMPR": ("numeric", _is_num, None),
    },
    "CANCER_REGISTRATION": {
        "DIAGNOSIS_CD": ("ICD-10 'Cxx...' string", lambda s: s.astype("string").str.match(r"^[CD]\d\d"), None),
        "DIAGNOSIS_DATE_2": ("parseable date", _is_date, None),
    },
}


# --------------------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------------------
def find_files(base_dir: str, table: str) -> dict[str, str]:
    """{batch_id: path} for ``<base_dir>/dNN/df_dNN_t<TABLE>_*.parquet`` (same rule as step 1)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(base_dir, "d*"))):
        b = os.path.basename(d)
        if not re.match(r"^d\d+$", b):
            continue
        hits = sorted(glob.glob(os.path.join(d, f"df_{b}_t{table}_*.parquet")))
        if hits:
            out[b] = hits[0]
    return out


def discover_extracts(root: str, pattern: str = "p1_pts_parse_whole_spilt_*") -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(root, pattern)) if os.path.isdir(p))


# --------------------------------------------------------------------------------------
# Schema (metadata only)
# --------------------------------------------------------------------------------------
def _type_str(t: pa.DataType) -> str:
    return str(t)


def file_schema(path: str) -> dict:
    meta = pq.read_metadata(path)
    sch = pq.read_schema(path)
    phys = {}
    for i in range(meta.num_columns):
        c = meta.schema.column(i)
        phys[c.name] = (c.physical_type, str(c.logical_type))
    return {"rows": meta.num_rows, "types": {f.name: _type_str(f.type) for f in sch},
            "physical": phys, "pandas_meta": bool(sch.metadata and b"pandas" in sch.metadata)}


def schema_table(base_dir: str, table: str) -> pd.DataFrame:
    """Long table: batch, column, arrow type, physical type, rows."""
    rows = []
    for b, p in find_files(base_dir, table).items():
        try:
            s = file_schema(p)
        except Exception as e:  # unreadable file
            rows.append({"batch": b, "column": "<ERROR>", "arrow_type": repr(e), "physical": "", "rows": None})
            continue
        for col, t in s["types"].items():
            rows.append({"batch": b, "column": col, "arrow_type": t,
                         "physical": "/".join(s["physical"].get(col, ("", ""))), "rows": s["rows"]})
    return pd.DataFrame(rows)


def summarize_types(st: pd.DataFrame) -> pd.DataFrame:
    """Per column: the Arrow types seen across batches, with batch counts."""
    rows = []
    if not st.empty:
        for col, g in st.groupby("column", sort=True):
            cnt = dict(Counter(g["arrow_type"]))
            rows.append({"column": col, "types": " | ".join(f"{t} x{n}" for t, n in sorted(cnt.items())),
                         "n_types": len(cnt), "n_batches": g["batch"].nunique()})
    return pd.DataFrame(rows, columns=["column", "types", "n_types", "n_batches"])


def compare_schemas(old_dir, new_dir, tables=TABLES) -> dict[str, pd.DataFrame]:
    """For each table, a column-level comparison of old vs new Arrow types.

    ``old_dir`` / ``new_dir`` may be a path or a dict {table: path} (step 1 uses a different
    folder name for CANCER_REGISTRATION: ``..._2025_`` with a trailing underscore).
    """
    out = {}
    for t, used in tables.items():
        od = old_dir[t] if isinstance(old_dir, dict) else old_dir
        nd = new_dir[t] if isinstance(new_dir, dict) else new_dir
        so, sn = schema_table(od, t), schema_table(nd, t)
        ao, an = summarize_types(so), summarize_types(sn)
        m = ao.merge(an, on="column", how="outer", suffixes=("_old", "_new"))
        m["used_in_step1"] = m["column"].isin(used)
        m["status"] = "same"
        m.loc[m["types_old"].isna(), "status"] = "added in new"
        m.loc[m["types_new"].isna(), "status"] = "missing in new"
        both = m["types_old"].notna() & m["types_new"].notna()
        def _set(x):
            return {t.rsplit(" x", 1)[0] for t in str(x).split(" | ")}
        diff = both & pd.Series([_set(o) != _set(n) for o, n in zip(m["types_old"], m["types_new"])], index=m.index)
        m.loc[diff, "status"] = "type changed"
        m.loc[both & (m["n_types_new"] > 1), "status"] = "mixed types across new batches"
        m.loc[both & (m["n_types_new"] > 1) & (m["n_types_old"] > 1), "status"] = "mixed in both"
        for c in [c for c in used if c not in set(m["column"])]:
            m.loc[len(m)] = {"column": c, "used_in_step1": True, "status": "missing in both"}
        m["n_files_old"], m["n_files_new"] = so["batch"].nunique() if len(so) else 0, sn["batch"].nunique() if len(sn) else 0
        order = {"missing in new": 0, "missing in both": 0, "type changed": 1, "mixed types across new batches": 2,
                 "mixed in both": 3, "added in new": 4, "same": 5}
        m["_o"] = m["status"].map(order)
        out[t] = m.sort_values(["used_in_step1", "_o", "column"], ascending=[False, True, True]).drop(columns="_o").reset_index(drop=True)
    return out


def issues_overview(cmp: dict[str, pd.DataFrame], only_used: bool = True) -> pd.DataFrame:
    rows = []
    for t, m in cmp.items():
        no, nn = int(m["n_files_old"].max() or 0), int(m["n_files_new"].max() or 0)
        if no == 0 or nn == 0:
            rows.append({"table": t, "column": "<files>", "status": "no batch files found",
                         "old": f"{no} files", "new": f"{nn} files"})
            continue
        if no != nn:
            rows.append({"table": t, "column": "<files>", "status": "different number of batch files",
                         "old": f"{no} files", "new": f"{nn} files"})
        mm = m[m["status"] != "same"]
        if only_used:
            mm = mm[mm["used_in_step1"]]
        for _, r in mm.iterrows():
            rows.append({"table": t, "column": r["column"], "status": r["status"],
                         "old": r.get("types_old"), "new": r.get("types_new")})
    return pd.DataFrame(rows)


def batches_by_type(base_dir: str, table: str, column: str) -> pd.DataFrame:
    """Which batch files have which type for one column (to locate the odd batches)."""
    st = schema_table(base_dir, table)
    st = st[st["column"] == column]
    return st.groupby("arrow_type")["batch"].apply(lambda s: ", ".join(sorted(s))).reset_index()


# --------------------------------------------------------------------------------------
# Values (sampled, only selected columns)
# --------------------------------------------------------------------------------------
def read_sample(path: str, columns: list[str], n_rows: int = 20000, **kw) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = [c for c in columns if c in pf.schema_arrow.names]
    if not cols:
        return pd.DataFrame()
    it = pf.iter_batches(batch_size=n_rows, columns=cols)
    try:
        b = next(it)
    except StopIteration:
        return pd.DataFrame(columns=cols)
    try:
        return b.to_pandas(**kw)
    except Exception:
        # e.g. timestamps out of the ns range: keep them as Python objects
        return b.to_pandas(timestamp_as_object=True)


def value_profile(base_dir: str, table: str, columns: list[str] | None = None,
                  n_rows: int = 20000, max_batches: int | None = None) -> pd.DataFrame:
    """Per column: pandas dtype after reading, Python types of the values, null %, examples."""
    columns = columns or TABLES[table]
    files = find_files(base_dir, table)
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    acc = defaultdict(lambda: {"dtypes": Counter(), "pytypes": Counter(), "n": 0, "null": 0, "ex": Counter()})
    for b, p in files.items():
        df = read_sample(p, columns, n_rows)
        for c in df.columns:
            s = df[c]
            a = acc[c]
            a["dtypes"][str(s.dtype)] += 1
            a["n"] += len(s)
            a["null"] += int(s.isna().sum())
            nn = s.dropna()
            a["pytypes"].update(type(v).__name__ for v in nn.head(2000))
            a["ex"].update(repr(v) for v in nn.head(2000))
    rows = []
    for c, a in acc.items():
        rows.append({"column": c, "pandas_dtypes": dict(a["dtypes"]), "python_types": dict(a["pytypes"]),
                     "null_pct": round(100 * a["null"] / max(a["n"], 1), 2),
                     "top_values": [v for v, _ in a["ex"].most_common(8)]})
    return pd.DataFrame(rows)


def compare_values(old_dir: str, new_dir: str, table: str, columns: list[str] | None = None, **kw) -> pd.DataFrame:
    o = value_profile(old_dir, table, columns, **kw).add_suffix("_old").rename(columns={"column_old": "column"})
    n = value_profile(new_dir, table, columns, **kw).add_suffix("_new").rename(columns={"column_new": "column"})
    m = o.merge(n, on="column", how="outer")
    cols = ["column"] + [f"{k}_{v}" for k in ["pandas_dtypes", "python_types", "null_pct", "top_values"] for v in ["old", "new"]]
    return m[cols]


def check_expectations(base_dir: str, table: str, n_rows: int = 20000, max_batches: int | None = None) -> pd.DataFrame:
    """Share of sampled values meeting each convention used by step 1/2.

    ``pct_ok_as_read`` evaluates the comparison exactly as the code does (e.g. ``s == '0'``);
    a drop to ~0% while ``pct_ok_as_string`` stays high means only the type changed.
    """
    exp = EXPECTATIONS.get(table, {})
    files = find_files(base_dir, table)
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    tot = defaultdict(lambda: [0, 0, 0])  # n, ok_string, ok_as_read
    for b, p in files.items():
        df = read_sample(p, list(exp), n_rows)
        for c, (desc, f_str, f_raw) in exp.items():
            if c not in df:
                continue
            s = df[c].dropna()
            tot[c][0] += len(s)
            try:
                tot[c][1] += int(f_str(s).fillna(False).sum())
            except Exception:
                pass
            if f_raw is not None:
                try:
                    tot[c][2] += int(pd.Series(f_raw(s)).fillna(False).sum())
                except Exception:
                    pass
    rows = []
    for c, (desc, _, f_raw) in exp.items():
        n, ok_s, ok_r = tot[c]
        rows.append({"column": c, "expectation": desc, "n_non_null_sampled": n,
                     "pct_ok_as_string": round(100 * ok_s / n, 2) if n else None,
                     "pct_ok_as_read": (round(100 * ok_r / n, 2) if n else None) if f_raw else None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Reproduce the step 1 concatenation to find the failing column
# --------------------------------------------------------------------------------------
def try_step1_concat(base_dir: str, table: str, columns: list[str] | None = None,
                     max_batches: int | None = None, out_path: str | None = None) -> pd.DataFrame:
    """Read every batch like step 1, concatenate, and try to write Parquet column by column.

    Returns a table of columns that become ``object`` after concat (with the mix of Python types)
    and whether writing that column to Parquet fails.
    """
    import io
    columns = columns or TABLES[table]
    files = find_files(base_dir, table)
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    parts, per_batch = [], []
    for b, p in files.items():
        cols = [c for c in columns if c in pq.ParquetFile(p).schema_arrow.names]
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception as e:
            per_batch.append({"batch": b, "column": "<read_parquet>", "error": repr(e)[:300]})
            continue
        parts.append(df)
        per_batch += [{"batch": b, "column": c, "dtype": str(df[c].dtype)} for c in df.columns]
    rep = pd.DataFrame(per_batch)
    if not parts:
        return rep
    allx = pd.concat(parts, ignore_index=True)
    rows = []
    for c in allx.columns:
        s = allx[c]
        err = None
        try:
            pa.Table.from_pandas(allx[[c]], preserve_index=False)
            buf = io.BytesIO(); allx[[c]].to_parquet(buf)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        dts = rep[rep["column"] == c]["dtype"].value_counts().to_dict() if "dtype" in rep else {}
        rows.append({"column": c, "dtype_after_concat": str(s.dtype), "batch_dtypes": dts,
                     "python_types": dict(Counter(type(v).__name__ for v in s.dropna().head(200000))),
                     "to_parquet_error": err})
    res = pd.DataFrame(rows)
    read_errors = rep[rep.get("column", pd.Series(dtype=str)) == "<read_parquet>"] if "error" in rep else pd.DataFrame()
    if len(read_errors):
        print("read_parquet errors:\n", read_errors.to_string(index=False))
    return res.sort_values("to_parquet_error", na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Optional: harmonise types before concatenation (proposal for step 1)
# --------------------------------------------------------------------------------------
STRING_COLS = {"hs", "PT_ID", "DIAGNOSIS_CD", "DRUG_PRICE_LIST_CD", "ANTI_CANCER_FLG", "STOP_FLG", "PT_SEX",
               "EXAM_ITEM_CD", "EXAM_ITEM", "COMPONENT_NAME", "ORDER_UNIT", "EXEC_UNIT", "RESULT_UNIT",
               "ICD_O_3_CD", "DIAGNOSIS_TEXT", "ICD_O_3_TEXT", "UICC_STAGE_BEFORE", "UICC_STAGE_AFTER",
               "MAIN5_STAGE_BEFORE", "DIAGNOSIS_DATE_1", "DIAGNOSIS_DATE_2"}
NUMERIC_COLS = {"AGE", "ORDER_DOSE", "EXEC_DOSE", "RESULT_NUM", "MS_HEIGHT", "MS_WEIGHT", "MS_BSA",
                "TEMPR", "PULSE", "HEART_RATE", "RESP", "BPH", "BPL"}
DATETIME_COLS = {"QUERY_DATE", "SOMA_DATE"}


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to one fixed type so that every batch / extract concatenates cleanly.

    Strings are stripped of trailing '.0' produced by float-typed codes (e.g. 4240403.0)
    but internal spaces (e.g. '  NEUT#') are preserved.
    """
    df = df.copy()
    for c in df.columns:
        if c in STRING_COLS:
            s = df[c]
            if pd.api.types.is_float_dtype(s):
                s = s.map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
            df[c] = s.astype("string")
        elif c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif c in DATETIME_COLS:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
            df[c] = pd.to_datetime(s, errors="coerce", format="mixed" if s.dtype == object else None).astype("datetime64[ns]")
    return df


# --------------------------------------------------------------------------------------
# Pinpointing "TypeError: '<=' not supported between instances of 'str' and 'float'"
# --------------------------------------------------------------------------------------
def mixed_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns whose non-null values mix Python types (e.g. str and float).

    min()/max()/sort on such a column raises "'<=' not supported between 'str' and 'float'".
    Note: NaN itself is a float, so an object column of strings + NaN is fine for most
    operations, but str + real numbers (or numeric strings + floats) is not.
    """
    rows = []
    for c in df.columns:
        s = df[c]
        if s.dtype != object:
            continue
        nn = s.dropna()
        types = Counter(type(v).__name__ for v in nn)
        if len(types) > 1:
            ex = {t: repr(next(v for v in nn if type(v).__name__ == t)) for t in types}
            rows.append({"column": c, "python_types": dict(types), "examples": ex})
    return pd.DataFrame(rows, columns=["column", "python_types", "examples"])


def non_numeric_values(base_dir: str, table: str, column: str, top: int = 30,
                       max_batches: int | None = None) -> pd.DataFrame:
    """Values of a supposedly numeric column that ``pd.to_numeric`` cannot parse (e.g. '<0.1', '検体不良')."""
    files = find_files(base_dir, table)
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    cnt, n_total, types = Counter(), 0, Counter()
    for b, p in files.items():
        s = pd.read_parquet(p, columns=[column])[column].dropna()
        n_total += len(s)
        types.update(type(v).__name__ for v in s.head(5000))
        bad = s[pd.to_numeric(s, errors="coerce").isna()]
        cnt.update(bad.astype(str))
    out = pd.DataFrame(cnt.most_common(top), columns=["value", "count"])
    out.attrs.update({"n_non_null": n_total, "n_non_numeric": sum(cnt.values()), "python_types": dict(types)})
    return out


def check_step2_inputs(step1_out_dir: str) -> pd.DataFrame:
    """Check the step1_all_*.parquet files that step 2 reads, for the columns step 2 aggregates."""
    targets = {
        "step1_all_sample_rslt.parquet": ["RESULT_NUM", "QUERY_DATE", "EXAM_ITEM", "PT_SEX", "hs"],
        "step1_all_vital.parquet": ["TEMPR", "PULSE", "RESP", "BPH", "BPL", "QUERY_DATE", "hs"],
        "step1_all_somatometry.parquet": ["MS_HEIGHT", "MS_WEIGHT", "SOMA_DATE", "hs"],
        "step1_all_injection.parquet": ["EXEC_DOSE", "QUERY_DATE", "DRUG_PRICE_LIST_CD", "ANTI_CANCER_FLG", "AGE", "hs"],
        "step1_all_drug_taken.parquet": ["EXEC_DOSE", "QUERY_DATE", "DRUG_PRICE_LIST_CD", "ANTI_CANCER_FLG", "AGE", "hs"],
        "step1_all_cancer.parquet": ["DIAGNOSIS_DATE_2", "DIAGNOSIS_CD", "hs"],
    }
    rows = []
    for fn, cols in targets.items():
        p = os.path.join(step1_out_dir, fn)
        if not os.path.exists(p):
            rows.append({"file": fn, "column": "<missing file>"}); continue
        sch = pq.read_schema(p)
        df = pd.read_parquet(p, columns=[c for c in cols if c in sch.names])
        mixed = set(mixed_object_columns(df)["column"]) if len(df) else set()
        for c in cols:
            if c not in df:
                rows.append({"file": fn, "column": c, "arrow_type": "<missing>"}); continue
            s = df[c]
            num_ok = pd.to_numeric(s, errors="coerce").notna().mean() if s.dtype == object else None
            rows.append({"file": fn, "column": c, "arrow_type": str(sch.field(c).type), "pandas_dtype": str(s.dtype),
                         "python_types": dict(Counter(type(v).__name__ for v in s.dropna().head(20000))),
                         "mixed_types": c in mixed, "pct_numeric_parseable": None if num_ok is None else round(100 * num_ok, 2),
                         "null_pct": round(100 * s.isna().mean(), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Units / value scale (a type change may come with a unit change)
# --------------------------------------------------------------------------------------
def lab_unit_summary(base_dir: str, items=("  NEUT#", "  WBC"), max_batches: int | None = None) -> pd.DataFrame:
    """Unit strings and value distribution of ANC / WBC (thresholds 0.5 / 1.0 assume 10^3/uL)."""
    files = find_files(base_dir, "MDV_SAMPLE_RSLT")
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    parts = []
    for p in files.values():
        d = pd.read_parquet(p, columns=["EXAM_ITEM", "RESULT_UNIT", "RESULT_NUM"])
        d = d[d["EXAM_ITEM"].astype(str).isin(items)]
        d["RESULT_NUM"] = pd.to_numeric(d["RESULT_NUM"], errors="coerce")
        parts.append(d)
    d = pd.concat(parts) if parts else pd.DataFrame(columns=["EXAM_ITEM", "RESULT_UNIT", "RESULT_NUM"])
    g = d.groupby(["EXAM_ITEM", "RESULT_UNIT"], dropna=False)["RESULT_NUM"]
    return g.describe(percentiles=[.05, .5, .95]).reset_index()


def dose_unit_summary(base_dir: str, table: str = "MDV_INJECTION", unit_col: str = "EXEC_UNIT",
                      top: int = 15, max_batches: int | None = None) -> pd.Series:
    files = find_files(base_dir, table)
    if max_batches:
        files = dict(list(files.items())[:max_batches])
    cnt = Counter()
    for p in files.values():
        cols = [c for c in [unit_col, "ANTI_CANCER_FLG"] if c in pq.ParquetFile(p).schema_arrow.names]
        d = pd.read_parquet(p, columns=cols)
        if "ANTI_CANCER_FLG" in d:
            d = d[d["ANTI_CANCER_FLG"] == "＋"]
        if unit_col in d:
            cnt.update(d[unit_col].astype(str))
        else:
            cnt.update({f"<{unit_col} missing>": len(d)})
    return pd.Series(dict(cnt.most_common(top)), name="count")
