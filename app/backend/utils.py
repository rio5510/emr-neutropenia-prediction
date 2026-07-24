"""
FN Risk Prediction — Feature Engineering Utilities
====================================================
This module provides the ``map_to_288_features`` function, which converts a
raw patient data dictionary into the 288-dimensional feature vector expected
by the trained LightGBM model.

Feature layout (based on ``feature_mapping.json``)
---------------------------------------------------
+------------------+------------------+---------------------------------------------------+
| Index range      | Count            | Description                                       |
+==================+==================+===================================================+
| Column_0 – 9    | 10               | Continuous clinical variables (vital signs,       |
|                  |                  | haematology); missing values imputed with         |
|                  |                  | training-set medians (see ``MEDIAN_VALUES``).     |
+------------------+------------------+---------------------------------------------------+
| Column_10 – 217  | 208              | Chemotherapy drug doses, keyed by YJ              |
|                  |                  | pharmaceutical classification code.               |
+------------------+------------------+---------------------------------------------------+
| Column_218       | 1                | Prophylactic G-CSF total dose (μg).               |
+------------------+------------------+---------------------------------------------------+
| Column_219       | 1                | Biological sex (0 = female, 1 = male).            |
+------------------+------------------+---------------------------------------------------+
| Column_220 – 287 | 68               | ICD-10 Chapter C diagnosis flags (binary).        |
+------------------+------------------+---------------------------------------------------+
| **Total**        | **288**          |                                                   |
+------------------+------------------+---------------------------------------------------+

Notes
-----
- Columns 218 and 219 are shifted by one relative to earlier model versions
  because a dummy drug feature was removed during model refinement.
- Column_220 – 287 now include ``C80`` (malignant neoplasm, primary site
  unknown), accounting for all 68 ICD-10 Chapter C codes supported by the
  model.
"""

import numpy as np

# ---------------------------------------------------------------------------
# YJ pharmaceutical classification codes
# Ordered list corresponding to Column_10 – Column_217 of the feature matrix.
# Each element is a 7-digit YJ code identifying a specific chemotherapy or
# supportive-care agent included in the training dataset.
# ---------------------------------------------------------------------------
YJ_CODE_ORDER = [
    "1190402", "1214405", "1317715", "1319401", "2399402", "2399403", "2399404",
    "2454002", "2454405", "2478002", "2499406", "2499407", "2499412", "3919401",
    "3929407", "3929410", "3999423", "3999429", "3999445", "3999462", "4211002",
    "4211401", "4211402", "4212400", "4213400", "4219002", "4219003", "4219004",
    "4219400", "4219401", "4219402", "4219403", "4219404", "4219405", "4219406",
    "4221001", "4222001", "4222400", "4223004", "4223005", "4223401", "4223701",
    "4224401", "4224403", "4229001", "4229100", "4229101", "4229400", "4229401",
    "4229402", "4229403", "4231400", "4233400", "4234400", "4234402", "4235400",
    "4235401", "4235402", "4235403", "4235404", "4235405", "4235406", "4239400",
    "4239401", "4240001", "4240400", "4240401", "4240402", "4240403", "4240404",
    "4240405", "4240406", "4240407", "4240408", "4240409", "4240410", "4291002",
    "4291003", "4291005", "4291006", "4291007", "4291009", "4291010", "4291011",
    "4291012", "4291013", "4291015", "4291016", "4291017", "4291018", "4291019",
    "4291020", "4291021", "4291023", "4291024", "4291026", "4291027", "4291028",
    "4291029", "4291030", "4291031", "4291032", "4291033", "4291034", "4291036",
    "4291037", "4291038", "4291039", "4291040", "4291043", "4291044", "4291045",
    "4291046", "4291047", "4291048", "4291049", "4291051", "4291052", "4291053",
    "4291054", "4291055", "4291057", "4291058", "4291059", "4291062", "4291063",
    "4291064", "4291065", "4291066", "4291068", "4291069", "4291070", "4291075",
    "4291076", "4291077", "4291400", "4291401", "4291402", "4291403", "4291405",
    "4291406", "4291407", "4291408", "4291409", "4291410", "4291412", "4291413",
    "4291415", "4291416", "4291417", "4291419", "4291420", "4291421", "4291422",
    "4291424", "4291425", "4291426", "4291427", "4291428", "4291429", "4291430",
    "4291431", "4291433", "4291434", "4291435", "4291436", "4291437", "4291438",
    "4291439", "4291441", "4291442", "4291443", "4291444", "4291445", "4291446",
    "4291448", "4291449", "4291450", "4291452", "4291454", "4291455", "4291457",
    "4291459", "4291464", "4291465", "4291466", "4291468", "4291469", "4291471",
    "4291500", "4291501", "4299002", "4299003", "4299100", "4299400", "4299404",
    "4299406", "4900402", "4900404", "6391700", "6399402", "6399413", "6399421",
    "6399423", "6399424", "6399425", "6399427", "6399429",
]  # 208 YJ codes (Column_10 – Column_217)

# ---------------------------------------------------------------------------
# ICD-10 Chapter C diagnosis codes
# Ordered list corresponding to Column_220 – Column_287 of the feature matrix.
# Each element is an ICD-10 category code for a malignant neoplasm.
# A feature flag is set to 1 when the corresponding diagnosis is present in
# the patient record.
#
# Note: Columns are shifted by one from the previous model version because
# a dummy drug feature (Column_218 in the old layout) was removed.
# ---------------------------------------------------------------------------
DIAGNOSIS_CODE_ORDER = [
    "C",   "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
    "C10", "C11", "C12", "C13", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    "C22", "C23", "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C37",
    "C38", "C40", "C41", "C42", "C44", "C47", "C48", "C49", "C50", "C51", "C52",
    "C53", "C54", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65",
    "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
    "C77", "C80",
]  # 68 ICD-10 codes (Column_220 – Column_287)

# ---------------------------------------------------------------------------
# Training-set median values for clinical variables (Column_0 – Column_9)
# These constants are used for median imputation when a feature value is
# missing (None) in the incoming request.  Values were computed from the
# training cohort and must remain fixed at inference time.
# ---------------------------------------------------------------------------
MEDIAN_VALUES = {
    "LastNEUT":   2980.0,   # Column_0: Absolute neutrophil count (cells/μL)
    "LastWBC":    4900.0,   # Column_1: White blood cell count (cells/μL)
    "AGE":          66,     # Column_2: Age (years)
    "MS_HEIGHT":   161.0,   # Column_3: Height (cm)
    "MS_WEIGHT":    56.3,   # Column_4: Weight (kg)
    "TEMPR":        36.4,   # Column_5: Body temperature (°C)
    "PULSE":        76.0,   # Column_6: Heart rate (beats/min)
    "RESP":         16.0,   # Column_7: Respiratory rate (breaths/min)
    "BPH":         119.0,   # Column_8: Systolic blood pressure (mmHg)
    "BPL":          69.0,   # Column_9: Diastolic blood pressure (mmHg)
}


def map_to_288_features(data_dict: dict) -> np.ndarray:
    """
    Convert a raw patient data dictionary to a 288-dimensional feature vector.

    This function implements the feature engineering pipeline applied during
    model training.  The resulting vector can be passed directly to the
    LightGBM classifier without further preprocessing.

    Feature construction
    --------------------
    1. **Continuous clinical features** (indices 0 – 9):
       Values are taken from ``data_dict`` as-is.  When a value is absent
       (``None``), it is replaced by the corresponding entry in
       ``MEDIAN_VALUES`` (median imputation using training-set statistics).

    2. **Drug dose features** (indices 10 – 217):
       For each of the 208 recognised YJ codes, the cumulative dose (mg) is
       looked up in ``data_dict["drugs"]``.  Unrecognised or absent codes
       receive a value of 0 (not administered).

    3. **Prophylactic G-CSF dose** (index 218):
       Total prophylactic G-CSF dose (μg) administered before the
       chemotherapy cycle.

    4. **Biological sex** (index 219):
       Binary indicator: 0 = female, 1 = male.

    5. **ICD-10 diagnosis flags** (indices 220 – 287):
       One-hot (binary) flags indicating the presence of each of the 68
       ICD-10 Chapter C categories in the patient record.

    Parameters
    ----------
    data_dict : dict
        Patient data dictionary, typically produced by
        ``PredictionRequest.model_dump()``.

    Returns
    -------
    np.ndarray, shape (288,)
        Feature vector ready for model inference.
    """
    vector = np.zeros(288)

    # ------------------------------------------------------------------
    # Indices 0 – 9: Continuous clinical variables with median imputation
    # ------------------------------------------------------------------
    vector[0] = data_dict.get("LastNEUT") if data_dict.get("LastNEUT") is not None else MEDIAN_VALUES["LastNEUT"]
    vector[1] = data_dict.get("LastWBC")  if data_dict.get("LastWBC")  is not None else MEDIAN_VALUES["LastWBC"]
    vector[2] = data_dict.get("AGE")      if data_dict.get("AGE")      is not None else MEDIAN_VALUES["AGE"]
    vector[3] = data_dict.get("MS_HEIGHT") if data_dict.get("MS_HEIGHT") is not None else MEDIAN_VALUES["MS_HEIGHT"]
    vector[4] = data_dict.get("MS_WEIGHT") if data_dict.get("MS_WEIGHT") is not None else MEDIAN_VALUES["MS_WEIGHT"]
    vector[5] = data_dict.get("TEMPR")    if data_dict.get("TEMPR")    is not None else MEDIAN_VALUES["TEMPR"]
    vector[6] = data_dict.get("PULSE")    if data_dict.get("PULSE")    is not None else MEDIAN_VALUES["PULSE"]
    vector[7] = data_dict.get("RESP")     if data_dict.get("RESP")     is not None else MEDIAN_VALUES["RESP"]
    vector[8] = data_dict.get("BPH")      if data_dict.get("BPH")      is not None else MEDIAN_VALUES["BPH"]
    vector[9] = data_dict.get("BPL")      if data_dict.get("BPL")      is not None else MEDIAN_VALUES["BPL"]

    # ------------------------------------------------------------------
    # Indices 10 – 217: Chemotherapy drug doses (208 YJ codes)
    # Absent drugs default to 0 (not administered).
    # ------------------------------------------------------------------
    drugs = data_dict.get("drugs") or {}
    for i, yj_code in enumerate(YJ_CODE_ORDER):
        vector[10 + i] = drugs.get(yj_code, 0)

    # ------------------------------------------------------------------
    # Index 218: Prophylactic G-CSF total dose (μg)
    # (Shifted by one from legacy layout after dummy feature removal.)
    # ------------------------------------------------------------------
    vector[218] = data_dict.get("TOTAL_GLAS_PREDOSE", 0)

    # ------------------------------------------------------------------
    # Index 219: Biological sex
    # (Shifted by one from legacy layout after dummy feature removal.)
    # ------------------------------------------------------------------
    vector[219] = data_dict.get("PT_SEX", 0)

    # ------------------------------------------------------------------
    # Indices 220 – 287: ICD-10 Chapter C diagnosis flags (68 codes)
    # Each flag is set to 1 if the code is present in the patient record.
    # (Shifted by one from legacy layout after dummy feature removal;
    #  C80 "malignant neoplasm, primary site unknown" is included.)
    # ------------------------------------------------------------------
    diagnoses = data_dict.get("diagnoses") or []
    for i, diag_code in enumerate(DIAGNOSIS_CODE_ORDER):
        if diag_code in diagnoses:
            vector[220 + i] = 1

    return vector
