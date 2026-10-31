# 🩺 Patient Health Record Dashboard
**A Data-Science-Based Healthcare Analytics & Predictive Dashboard**  
**Author:** ANSHU SHAKYA  
**Domain:** Data Science + Python + Machine Learning  

---

## 📌 Executive Summary & Introduction
The **Patient Health Record Dashboard** is an end-to-end data-science-driven healthcare application developed to collect, organize, clean, analyze, and visualize patient electronic health records (EHR). It transforms raw clinical records into actionable health insights and implements five machine learning algorithms for risk stratification and clinical decision support.

---

## 🏥 Problem Statement
Traditional healthcare facilities generate massive quantities of patient data across demographic parameters, vital signs, and laboratory investigations. Manual analysis is time-consuming, prone to human oversight, and conceals critical non-linear correlations between physiological markers (e.g., blood pressure, blood glucose, cholesterol, and BMI).

This project addresses these challenges by developing a centralized, interactive dashboard capable of:
1. Cleaning and preprocessing inconsistent or missing clinical records.
2. Uncovering demographic and biometric trends through Exploratory Data Analysis (EDA) & an **Interactive 3D Physiological Feature Space**.
3. Providing **One-Click Clinical Presets** for instant cohort filtering (e.g., Diabetic, Stage 2 Hypertension, Senior Citizens).
4. Providing an interactive **'What-If' Lifestyle & Treatment Health Optimizer** for individual patient dossiers.
5. Offering **Dynamic New Patient Intake & Real-Time Classification** that appends directly to the active EHR registry.
6. Comparing 5 industry-standard machine learning classification models with **Multi-Model ROC Curves** and interactive **Decision Threshold Tuning**.
7. Providing real-time patient disease-risk prediction with an **Animated Speedometer Gauge** and downloadable clinical evaluation reports.

---

## 🎯 Project Objectives
1. Collect and preprocess patient healthcare data.
2. Clean missing, duplicate, and inconsistent data.
3. Perform Exploratory Data Analysis (EDA) on patient records.
4. Identify relationships between different health parameters.
5. Create interactive graphs and visualizations.
6. Provide patient-level and dataset-level health insights.
7. Apply suitable Machine Learning algorithms for prediction/classification.
8. Evaluate the performance of the developed ML models.
9. Provide an easy-to-use dashboard for viewing healthcare information.
10. Demonstrate the application of data science techniques in healthcare analytics.

---

## 📊 Dataset Attributes Specification
| Attribute | Description | Typical Value / Range |
| :--- | :--- | :--- |
| **Patient ID** | Unique patient alphanumeric identifier | `PAT-1001` to `PAT-2500` |
| **Age** | Age in completed years | 18 – 85 years |
| **Gender** | Sex of patient | Male / Female / Other |
| **Blood Pressure** | Systolic / Diastolic pressure | `120/80 mmHg` (85–200 / 55–120) |
| **Heart Rate** | Resting pulse rate | 50 – 125 beats per minute (bpm) |
| **Blood Sugar** | Fasting blood glucose concentration | 65 – 290 mg/dL |
| **BMI** | Body Mass Index | 16.5 – 45.0 kg/m² |
| **Cholesterol** | Total serum cholesterol level | 120 – 340 mg/dL |
| **Medical History** | Documented prior chronic conditions | Hypertension, Type 2 Diabetes, Asthma, None |
| **Diagnosis** | Primary diagnosed medical condition | Healthy/Normal, Hypertension, Type 2 Diabetes, etc. |
| **Medication** | Prescribed pharmacological treatment | Metformin, Lisinopril, Atorvastatin, None |
| **Test Results** | Laboratory diagnostic outcome | Normal, Borderline, Abnormal |
| **Risk Category** | Clinical disease-risk classification | **Low**, **Moderate**, **High** |

---

## ⚙️ Data Science Methodology
The project strictly implements the 7-step data science lifecycle described in the project synopsis:
```
Step 1: Data Collection (Structured multi-attribute healthcare records)
   ↓
Step 2: Data Cleaning (Deduplication, Median Imputation, Outlier Capping)
   ↓
Step 3: Data Preprocessing (Binary/Label Encoding, StandardScaler)
   ↓
Step 4: Exploratory Data Analysis (EDA, Pearson Heatmaps, AHA BP Staging)
   ↓
Step 5: Machine Learning (5 Models: LR, DT, RF, KNN, SVM)
   ↓
Step 6: Visualization (Plotly interactive charts, KPI metrics)
   ↓
Step 7: Dashboard Deployment (Streamlit Web Application)
```

---

## 🤖 Machine Learning Model Benchmark Results
Five algorithms were trained and evaluated on unseen test data using Stratified 80/20 train-test splits:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | **86.67%** | **0.8675** | **0.8667** | **0.8667** | **0.9556** |
| **Support Vector Machine (SVM)** | 82.33% | 0.8257 | 0.8233 | 0.8241 | 0.9352 |
| **K-Nearest Neighbours (KNN)** | 80.67% | 0.8023 | 0.8067 | 0.8026 | 0.9131 |
| **Decision Tree** | 80.00% | 0.7972 | 0.8000 | 0.7973 | 0.9258 |
| **Logistic Regression** | 78.33% | 0.7850 | 0.7833 | 0.7839 | 0.9187 |

> **Top Performer:** Random Forest achieved the highest Accuracy (86.67%) and ROC-AUC (0.9556).

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.13
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Visualizations:** `plotly`, `seaborn`, `matplotlib`
- **Interactive UI Framework:** `streamlit`
- **Model Serialization:** `joblib`

---

## 🚀 Installation & Running Instructions

### 1. Open Terminal / PowerShell in the Project Directory:
```bash
cd C:\Users\anshu\.gemini\antigravity\scratch\patient_health_dashboard
```

### 2. Install Required Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline (Optional - Already Pre-trained):
```bash
python src/data_generator.py
python src/data_cleaning.py
python src/model_trainer.py
```

### 4. Launch the Interactive Dashboard:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Repository Directory Structure
```
patient_health_dashboard/
│
├── data/
│   ├── raw_healthcare_dataset.csv       <- Raw synthetic patient data with missing values/duplicates
│   └── cleaned_healthcare_dataset.csv   <- Preprocessed & cleaned patient dataset
│
├── models/
│   ├── trained_models_bundle.joblib     <- Serialized models, scaler, encoder & metrics
│   └── model_benchmark_results.csv      <- Comparison table of all 5 ML algorithms
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py                <- Generates realistic clinical records
│   ├── data_cleaning.py                 <- Cleaning pipeline (median imputation, outlier handling)
│   ├── model_trainer.py                 <- Trains and evaluates the 5 ML models
│   └── prediction_engine.py             <- Inference engine & clinical recommendations
│
├── app.py                               <- Full Streamlit interactive dashboard application
├── requirements.txt                     <- Project dependencies
└── README.md                            <- Comprehensive project documentation
```

---

## ⚖️ Ethical & Medical Disclaimer
*The machine learning models and visualizations in this project are designed solely as a data-science educational and clinical decision-support demonstration. They do not constitute certified medical diagnosis or replace consultation with licensed healthcare practitioners.*
