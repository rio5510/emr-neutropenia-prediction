# FN-PRED-AI: A LightGBM-Based Clinical Decision-Support System for Febrile Neutropenia Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev/)

## Overview

**FN-PRED-AI** is an open-source clinical decision-support system for predicting the risk of **febrile neutropenia (FN)** in cancer patients undergoing cytotoxic chemotherapy.  The system combines a trained **LightGBM** gradient-boosting classifier with a **React/FastAPI** web interface designed for bedside clinical use.

Febrile neutropenia is a life-threatening oncological emergency characterised by fever in the setting of a significantly reduced neutrophil count.  Early risk stratification can guide prophylactic interventions (e.g., G-CSF administration, antibiotic prophylaxis) and potentially reduce treatment-related mortality.

The model accepts **288 patient-level features** spanning demographics, vital signs, haematological laboratory values, chemotherapy drug exposures (encoded by 7-digit YJ pharmaceutical classification code), and ICD-10 malignant-neoplasm diagnosis codes, and outputs a continuous risk probability (0–1) along with a binary risk classification (High / Low).

---

## Repository Structure

```
app/
├── backend/
│   ├── main.py                # FastAPI application entry point
│   ├── utils.py               # Feature engineering pipeline
│   ├── feature_mapping.csv    # Human-readable feature index reference
│   ├── feature_mapping.json   # Machine-readable feature mapping
│   └── all_lgb_model.joblib   # Serialised LightGBM model artifact
└── frontend/
    ├── src/
    │   ├── App.jsx            # Main React component (UI and API client)
    │   ├── App.css            # Component-level styles
    │   ├── main.jsx           # React DOM entry point
    │   └── index.css          # Global styles
    ├── public/
    ├── index.html
    ├── package.json
    ├── eslint.config.js
    └── vite.config.js
```

---

## Feature Engineering

The 288-dimensional feature vector is constructed as follows:

| Column index   | Count | Description                                                                                          |
|----------------|-------|------------------------------------------------------------------------------------------------------|
| 0 – 9          | 10    | Continuous clinical variables (ANC, WBC, age, height, weight, temperature, heart rate, respiratory rate, systolic BP, diastolic BP). Missing values are imputed with training-set medians. |
| 10 – 217       | 208   | Cumulative chemotherapy drug doses identified by YJ pharmaceutical classification code (mg).         |
| 218            | 1     | Prophylactic granulocyte colony-stimulating factor (G-CSF) total dose (μg).                         |
| 219            | 1     | Biological sex (0 = female, 1 = male).                                                               |
| 220 – 287      | 68    | ICD-10 Chapter C (malignant neoplasm) diagnosis flags (binary, one per ICD-10 category code).        |
| **Total**      | **288** |                                                                                                    |

Missing values in columns 0–9 are handled by **median imputation** using fixed constants derived from the training cohort (see `backend/utils.py`, `MEDIAN_VALUES`).

---

## System Requirements

### Backend

| Dependency   | Version   |
|--------------|-----------|
| Python       | ≥ 3.10    |
| FastAPI      | ≥ 0.100   |
| Uvicorn      | ≥ 0.23    |
| LightGBM     | ≥ 4.0     |
| scikit-learn | ≥ 1.3     |
| joblib       | ≥ 1.3     |
| NumPy        | ≥ 1.24    |
| Pydantic     | ≥ 2.0     |

### Frontend

| Dependency | Version |
|------------|---------|
| Node.js    | ≥ 18    |
| React      | 19      |
| Vite       | 8       |

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rio5510/emr-neutropenia-prediction.git
cd emr-neutropenia-prediction/app
```

### 2. Start the backend API server

```bash
cd backend
pip install fastapi uvicorn lightgbm scikit-learn joblib numpy pydantic
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive API documentation (Swagger UI) is accessible at `http://localhost:8000/docs`.

### 3. Start the frontend development server

```bash
cd ../frontend
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173` (default Vite port).

---

## API Reference

### `POST /predict`

Run FN risk inference on a single patient record.

**Request body** (`application/json`):

```json
{
  "LastNEUT": 1540,
  "LastWBC": 4500,
  "AGE": 65,
  "MS_HEIGHT": 165.0,
  "MS_WEIGHT": 62.5,
  "TEMPR": 36.6,
  "PULSE": 72,
  "RESP": 16,
  "BPH": 128,
  "BPL": 82,
  "PT_SEX": 1,
  "TOTAL_GLAS_PREDOSE": 0,
  "drugs": { "4291002": 150.0 },
  "diagnoses": ["C34"]
}
```

All fields except `PT_SEX` and `TOTAL_GLAS_PREDOSE` are optional.  Missing continuous variables are imputed with training-set medians.

**Response** (`application/json`):

```json
{
  "risk_score": 0.312,
  "prediction": 0,
  "interpretation": "Low Risk",
  "level": "low"
}
```

| Field           | Type    | Description                                              |
|-----------------|---------|----------------------------------------------------------|
| `risk_score`    | float   | Predicted FN probability (0.0 – 1.0).                   |
| `prediction`    | int     | Binary class label: 0 = Low Risk, 1 = High Risk.        |
| `interpretation`| string  | Plain-language risk classification.                      |
| `level`         | string  | Machine-readable alias: `"high"` or `"low"`.            |

### `GET /health`

Returns the operational status of the API and whether the model artifact is loaded.

---

## Clinical Disclaimer

> **This system is a research prototype intended solely for academic evaluation and is not a certified medical device.**  
> Risk scores produced by this tool constitute supplementary decision-support information and must not replace the independent clinical judgment of a licensed physician. All final diagnostic and therapeutic decisions remain the sole responsibility of the treating clinician.

---

## Citation

If you use FN-PRED-AI in your research, please cite the following:

```bibtex
@misc{fnpredai2025,
  author       = {Okamoto, Ryosuke},
  title        = {{FN-PRED-AI}: A LightGBM-Based Clinical Decision-Support System for Febrile Neutropenia Risk Prediction},
  year         = {2025},
  howpublished = {\url{https://github.com/rio5510/emr-neutropenia-prediction}},
}
```

---

## License

This project is distributed under the [MIT License](../LICENSE).
