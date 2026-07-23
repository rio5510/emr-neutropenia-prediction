"""
FN Risk Prediction API — Backend Entry Point
=============================================
FastAPI application that serves a LightGBM-based febrile neutropenia (FN)
risk prediction model.  The model accepts 288 clinical features (vital signs,
laboratory values, chemotherapy drug doses identified by YJ codes, and ICD-10
cancer diagnoses) and returns a probabilistic risk score.

Endpoints
---------
POST /predict  — Run inference on a single patient record.
GET  /health   — Health check; confirms the model is loaded.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from utils import map_to_288_features

app = FastAPI(
    title="FN Risk Prediction API",
    description=(
        "REST API for predicting febrile neutropenia (FN) risk in cancer "
        "patients undergoing chemotherapy, powered by a LightGBM classifier."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins so that the React frontend can communicate with
# this API during development and deployment.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the serialised LightGBM model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "all_lgb_model.joblib")

# ---------------------------------------------------------------------------
# Model loading — performed once at application startup to minimise per-
# request latency.  A startup failure is non-fatal: subsequent requests to
# /predict will return HTTP 500 with an informative message.
# ---------------------------------------------------------------------------
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """
    Schema for a single patient prediction request.

    Clinical features (Column_0 – Column_9)
    ----------------------------------------
    These ten features correspond to the first ten columns of the training
    dataset.  Missing values (``None``) are imputed with training-set medians
    inside ``map_to_288_features``.

    Parameters
    ----------
    LastNEUT : float, optional
        Most recent absolute neutrophil count (cells/μL).
    LastWBC : float, optional
        Most recent white blood cell count (cells/μL).
    AGE : int, optional
        Patient age at the time of chemotherapy initiation (years).
    MS_HEIGHT : float, optional
        Patient height (cm).
    MS_WEIGHT : float, optional
        Patient weight (kg).
    TEMPR : float, optional
        Body temperature at most recent measurement (°C).
    PULSE : float, optional
        Heart rate at most recent measurement (beats/min).
    RESP : float, optional
        Respiratory rate at most recent measurement (breaths/min).
    BPH : float, optional
        Systolic blood pressure (mmHg).
    BPL : float, optional
        Diastolic blood pressure (mmHg).
    PT_SEX : int
        Biological sex of the patient (0 = female, 1 = male).
    TOTAL_GLAS_PREDOSE : float
        Total prophylactic granulocyte colony-stimulating factor (G-CSF)
        dose administered prior to the chemotherapy cycle (μg).
    drugs : dict of {str: float}, optional
        Chemotherapy drugs encoded as YJ code → cumulative dose (mg) pairs.
        Up to 208 distinct YJ codes are recognised by the model.
    diagnoses : list of str, optional
        ICD-10 Chapter C (malignant neoplasm) codes present in the patient
        record, e.g. ``["C34", "C50"]``.
    """

    # Continuous clinical measurements — optional to support partial records
    LastNEUT: Optional[float] = None            # Absolute neutrophil count (cells/μL)
    LastWBC: Optional[float] = None             # White blood cell count (cells/μL)
    AGE: Optional[int] = None                   # Age (years)
    MS_HEIGHT: Optional[float] = None           # Height (cm)
    MS_WEIGHT: Optional[float] = None           # Weight (kg)
    TEMPR: Optional[float] = None               # Body temperature (°C)
    PULSE: Optional[float] = None               # Heart rate (beats/min)
    RESP: Optional[float] = None                # Respiratory rate (breaths/min)
    BPH: Optional[float] = None                 # Systolic blood pressure (mmHg)
    BPL: Optional[float] = None                 # Diastolic blood pressure (mmHg)

    # Required fields
    PT_SEX: int                                  # Sex (0 = female, 1 = male)
    TOTAL_GLAS_PREDOSE: float                    # Prophylactic G-CSF total dose (μg)

    # Drug doses keyed by YJ pharmaceutical classification code
    drugs: Optional[Dict[str, float]] = None

    # ICD-10 malignant neoplasm (Chapter C) diagnosis codes
    diagnoses: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", summary="Predict FN risk for a single patient")
async def predict(request: PredictionRequest):
    """
    Run the LightGBM FN risk classifier on the supplied patient record.

    The request body is mapped to a 288-dimensional feature vector, which is
    then passed to the model.  The response contains the predicted class
    (0 = low risk, 1 = high risk), the positive-class probability (risk
    score), and a plain-language interpretation.

    Returns
    -------
    dict
        ``risk_score``     — Predicted FN probability (0.0 – 1.0).
        ``prediction``     — Binary class label (0 or 1).
        ``interpretation`` — "High Risk" if risk_score > 0.5, else "Low Risk".
        ``level``          — "high" or "low" (machine-readable alias).

    Raises
    ------
    HTTPException(500)
        If the model is not loaded or an unexpected error occurs during
        inference.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        data_dict = request.model_dump()
        feature_vector = map_to_288_features(data_dict)
        data = np.array(feature_vector).reshape(1, -1)

        proba = model.predict_proba(data)[0][1]   # Positive-class probability
        prediction = int(model.predict(data)[0])  # Binary prediction (0 or 1)

        return {
            "risk_score": float(proba),
            "prediction": prediction,
            "interpretation": "High Risk" if proba > 0.5 else "Low Risk",
            "level": "high" if proba > 0.5 else "low",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@app.get("/health", summary="API health check")
async def health():
    """
    Return the operational status of the API and whether the model is loaded.
    """
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------------
# Standalone execution (development only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
