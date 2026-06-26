# **ML-Based Prediction of Chemotherapy-Induced Neutropenic Events**

This repository contains the machine learning pipeline to predict chemotherapy-induced neutropenic events in patients with solid tumors, using real-world Electronic Medical Record (EMR) data.

## **📌 Project Overview**

This study develops predictive models (Random Forest, XGBoost, LightGBM) to forecast high-risk neutropenic events at the time of chemotherapy administration (Day 1\) within a 28-day window.

* **Data Source:** Kyoto University Hospital EMR data (Jan 2016 – Mar 2025).  
* **Target Population:** Patients with solid tumors.  
* **Composite Outcome (Label \= 1):** Occurrence of either Grade 4 neutropenia, clinical Febrile Neutropenia (FN), or therapeutic G-CSF administration within 28 days post-chemotherapy.

## **📂 Repository Structure**

.  
├── notebooks/  
│   ├── step1\_all\_cancer.ipynb  
│   ├── step1\_all\_drug\_taken\_data.ipynb  
│   ├── step1\_all\_injection\_data.ipynb  
│   ├── step1\_all\_sample\_rslt\_data\_prep.ipynb  
│   ├── step1\_all\_somatometry.ipynb  
│   ├── step1\_all\_vital\_data.ipynb  
│   ├── step2\_all\_data\_processing\_refactored.ipynb  
│   ├── step3\_all\_model\_train\_rf.ipynb  
│   └── step4\_all\_model\_evaluation.ipynb  
├── data/                            \# Local directory (not included in the repository)  
│   ├── step1\_dwh\_outputs/             
│   ├── step2\_processing\_outputs/      
│   ├── step3\_model\_outputs/           
│   └── step4\_evaluation\_outputs/      
└── README.md
