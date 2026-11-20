"""
Fixed column types for the step 1 outputs.

The 2026 DWH extract delivers numeric columns (RESULT_NUM, TEMPR, MS_HEIGHT, MS_WEIGHT, EXEC_DOSE, ...)
as strings and date columns (QUERY_DATE, SOMA_DATE) as ``date32[day]`` (Python ``datetime.date``).
The values themselves are unchanged (step 0, sections 4 and 7). ``normalize_types`` restores the
types that the downstream code expects, and stops (``strict=True``) instead of silently turning
unparseable values into NaN.
"""
from __future__ import annotations

import pandas as pd

NUMERIC_COLS = ("AGE", "RESULT_NUM", "TEMPR", "PULSE", "HEART_RATE", "RESP", "BPH", "BPL",
                "MS_HEIGHT", "MS_WEIGHT", "MS_BSA", "EXEC_DOSE", "ORDER_DOSE")
DATETIME_COLS = ("QUERY_DATE", "SOMA_DATE")
STRING_COLS = ("hs", "PT_ID", "PT_SEX", "EXAM_ITEM_CD", "EXAM_ITEM", "RESULT_UNIT", "COMPONENT_NAME",
               "DRUG_PRICE_LIST_CD", "ANTI_CANCER_FLG", "STOP_FLG", "EXEC_UNIT", "ORDER_UNIT",
               "DIAGNOSIS_CD", "DIAGNOSIS_TEXT", "ICD_O_3_CD", "ICD_O_3_TEXT", "UICC_STAGE_BEFORE",
               "UICC_STAGE_AFTER", "MAIN5_STAGE_BEFORE", "DIAGNOSIS_DATE_1", "DIAGNOSIS_DATE_2")


class TypeNormalizationError(ValueError):
    pass


def _blank_to_na(s: pd.Series) -> pd.Series:
    """Empty / whitespace-only strings are treated as missing for numeric and date columns."""
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype("string").str.strip()
        s = s.mask(s == "")
    return s


def normalize_types(df: pd.DataFrame, name: str = "", strict: bool = True,
                    max_examples: int = 10) -> pd.DataFrame:
    """Cast known columns to fixed types: numeric -> float64, dates -> datetime64[ns], codes -> string.

    With ``strict=True`` a ``TypeNormalizationError`` is raised when a non-missing value cannot be
    converted, listing the offending values; with ``strict=False`` they become NaN / NaT and are reported.
    """
    df = df.copy()
    problems = {}
    for c in df.columns:
        if c in NUMERIC_COLS:
            raw = _blank_to_na(df[c])
            num = pd.to_numeric(raw, errors="coerce")
            bad = raw[raw.notna() & num.isna()]
            if len(bad):
                problems[c] = bad.astype(str).value_counts().head(max_examples).to_dict()
            df[c] = num.astype("float64")
        elif c in DATETIME_COLS:
            raw = df[c]
            if not pd.api.types.is_datetime64_any_dtype(raw):
                nn = raw.dropna()
                if len(nn) and nn.map(lambda v: isinstance(v, str)).all():
                    raw = _blank_to_na(raw)          # date strings
                # datetime.date objects (date32) are handled by pd.to_datetime directly
            conv = pd.to_datetime(raw, errors="coerce")
            if getattr(conv.dt, "tz", None) is not None:
                conv = conv.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
            bad = raw[pd.Series(raw).notna() & conv.isna()]
            if len(bad):
                problems[c] = bad.astype(str).value_counts().head(max_examples).to_dict()
            df[c] = conv.astype("datetime64[ns]")
        elif c in STRING_COLS:
            # keep internal / leading spaces (e.g. '  NEUT#'); only the dtype is fixed
            s = df[c]
            if pd.api.types.is_float_dtype(s):   # codes read as float (e.g. 4240403.0)
                s = s.map(lambda v: pd.NA if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
            df[c] = s.astype("string").astype(object).where(s.notna(), None)
    if problems:
        msg = f"[{name}] values that cannot be converted: {problems}"
        if strict:
            raise TypeNormalizationError(msg)
        print("WARNING", msg)
    return df


def describe_types(df: pd.DataFrame) -> pd.DataFrame:
    """dtype and null share per column (quick check after normalization)."""
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "null_pct": (df.isna().mean() * 100).round(2)})
