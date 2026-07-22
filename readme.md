# **ML-Based Prediction of Chemotherapy-Induced Neutropenic Events**

This repository contains the machine learning pipeline to predict chemotherapy-induced neutropenic events in patients with solid tumors, using real-world Electronic Medical Record (EMR) data.

## **📌 Project Overview**

This study develops predictive models (Random Forest, XGBoost, LightGBM) to forecast high-risk neutropenic events at the time of chemotherapy administration (Day 1\) within a 28-day window.

* **Data Source:** Kyoto University Hospital EMR data (Jan 2016 – Mar 2025).  
* **Target Population:** Patients with solid tumors.  
* **Composite Outcome (Label = 1):** Occurrence of either Grade 4 neutropenia, clinical Febrile Neutropenia (FN), or therapeutic G-CSF administration within 28 days post-chemotherapy.

## **📂 Repository Structure**

.  
├── notebooks/  
│   ├── step1_all_cancer.ipynb  
│   ├── step1_all_drug_taken_data.ipynb  
│   ├── step1_all_injection_data.ipynb  
│   ├── step1_all_sample_rslt_data_prep.ipynb  
│   ├── step1_all_somatometry.ipynb  
│   ├── step1_all_vital_data.ipynb  
│   ├── step2_all_data_processing_refactored.ipynb  
│   ├── step3_all_model_train.ipynb  
│   └── step4_all_model_evaluation.ipynb  
├── data/                            # Local directory (not included in the repository)  
│   ├── step1_dwh_outputs/             
│   ├── step2_processing_outputs/      
│   ├── step3_model_outputs/           
│   └── step4_evaluation_outputs/      
└── README.md
