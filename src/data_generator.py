"""
data_generator.py
Generates a realistic, clinically-grounded patient healthcare dataset
based on the attributes specified in the Project Synopsis Report:
- Patient ID, Age, Gender, Blood Pressure (Systolic & Diastolic)
- Heart Rate, Blood Sugar, BMI, Cholesterol
- Medical History, Diagnosis, Medication, Test Results, Disease Risk Category
"""

import numpy as np
import pandas as pd
import os

def generate_patient_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Patient ID
    patient_ids = [f"PAT-{1000 + i:04d}" for i in range(1, n_samples + 1)]
    
    # 2. Demographics
    ages = np.random.randint(18, 85, size=n_samples)
    genders = np.random.choice(["Male", "Female", "Other"], size=n_samples, p=[0.49, 0.49, 0.02])
    
    # 3. Biometric Features conditioned on Age and lifestyle factors
    # BMI: typically normal distribution centered at 26.5 with variance
    bmi = np.round(np.random.normal(loc=27.0 + (ages > 50) * 1.5, scale=4.5, size=n_samples), 1)
    bmi = np.clip(bmi, 16.5, 45.0)
    
    # Blood Pressure: Systolic & Diastolic correlated with Age & BMI
    base_systolic = 110 + (ages * 0.35) + ((bmi - 22) * 1.2) + np.random.normal(0, 10, n_samples)
    systolic_bp = np.round(np.clip(base_systolic, 85, 200)).astype(int)
    
    base_diastolic = 70 + (ages * 0.15) + ((bmi - 22) * 0.7) + np.random.normal(0, 7, n_samples)
    diastolic_bp = np.round(np.clip(base_diastolic, 55, 120)).astype(int)
    
    # Heart Rate (bpm)
    heart_rate = np.round(np.random.normal(74 + (ages > 60)*3, 10, n_samples)).astype(int)
    heart_rate = np.clip(heart_rate, 50, 125)
    
    # Blood Sugar / Fasting Blood Glucose (mg/dL)
    # Higher for older age, higher BMI
    sugar_base = 85 + (ages * 0.3) + ((bmi - 24) * 2.2) + np.random.normal(0, 20, n_samples)
    # Add spikes for diabetic population
    diabetic_spikes = (np.random.rand(n_samples) < 0.22) * np.random.uniform(40, 120, n_samples)
    blood_sugar = np.round(np.clip(sugar_base + diabetic_spikes, 65, 290)).astype(int)
    
    # Cholesterol (mg/dL)
    chol_base = 160 + (ages * 0.5) + ((bmi - 24) * 1.8) + np.random.normal(0, 28, n_samples)
    cholesterol = np.round(np.clip(chol_base, 120, 340)).astype(int)
    
    # 4. Clinical Risk Scoring & Diagnosis assignment
    # Calculating a multi-factorial clinical risk score
    risk_score = (
        (systolic_bp >= 140) * 2.5 +
        (systolic_bp >= 130) * 1.0 +
        (diastolic_bp >= 90) * 2.0 +
        (blood_sugar >= 126) * 3.0 +
        (blood_sugar >= 100) * 1.0 +
        (cholesterol >= 240) * 2.5 +
        (cholesterol >= 200) * 1.0 +
        (bmi >= 30.0) * 2.5 +
        (bmi >= 25.0) * 1.0 +
        (ages >= 60) * 2.0 +
        (ages >= 45) * 1.0 +
        np.random.normal(0, 0.8, n_samples)
    )
    
    # Risk Category
    risk_category = []
    diagnosis = []
    medication = []
    medical_history = []
    test_results = []
    
    for i in range(n_samples):
        score = risk_score[i]
        sys = systolic_bp[i]
        glu = blood_sugar[i]
        chol = cholesterol[i]
        bm = bmi[i]
        ag = ages[i]
        
        # Risk classification
        if score >= 6.5:
            cat = "High"
        elif score >= 3.0:
            cat = "Moderate"
        else:
            cat = "Low"
        risk_category.append(cat)
        
        # Primary Diagnosis
        if glu >= 140 and sys >= 140:
            diag = "Cardiometabolic Risk"
            med = np.random.choice(["Metformin + Lisinopril", "Combination Therapy", "Amlodipine + Metformin"])
            hist = np.random.choice(["Hypertension & Diabetes", "Family History of Heart Disease", "Type 2 Diabetes"])
            test = "Abnormal"
        elif glu >= 126:
            diag = "Type 2 Diabetes"
            med = np.random.choice(["Metformin", "Glipizide", "Insulin", "Lifestyle & Metformin"])
            hist = np.random.choice(["Gestational Diabetes", "Prediabetes", "Family History of Diabetes", "None"])
            test = np.random.choice(["Abnormal", "Borderline"], p=[0.7, 0.3])
        elif sys >= 140 or diastolic_bp[i] >= 90:
            diag = "Hypertension"
            med = np.random.choice(["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide"])
            hist = np.random.choice(["Hypertension", "High Stress", "Family History of Stroke", "None"])
            test = np.random.choice(["Abnormal", "Borderline"], p=[0.6, 0.4])
        elif chol >= 240:
            diag = "Hyperlipidemia"
            med = np.random.choice(["Atorvastatin", "Rosuvastatin", "Simvastatin"])
            hist = np.random.choice(["High Cholesterol", "Coronary Artery Disease", "None"])
            test = "Abnormal"
        elif bm >= 32.0:
            diag = "Metabolic Syndrome"
            med = np.random.choice(["Dietary Therapy", "Metformin", "None"])
            hist = np.random.choice(["Obesity", "Fatty Liver", "Sedentary Lifestyle"])
            test = "Borderline"
        elif ag > 50 and np.random.rand() < 0.15:
            diag = "Arrhythmia / Tachycardia" if heart_rate[i] > 90 else "Osteoarthritis"
            med = "Beta Blocker" if heart_rate[i] > 90 else "NSAIDs"
            hist = "Joint Pain" if heart_rate[i] <= 90 else "Cardiac Murmur"
            test = "Borderline"
        else:
            diag = "Healthy / Normal"
            med = "None"
            hist = np.random.choice(["None", "Seasonal Allergies", "Minor Asthma"], p=[0.8, 0.1, 0.1])
            test = np.random.choice(["Normal", "Borderline"], p=[0.85, 0.15])
            
        diagnosis.append(diag)
        medication.append(med)
        medical_history.append(hist)
        test_results.append(test)
        
    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Blood_Pressure": [f"{s}/{d}" for s, d in zip(systolic_bp, diastolic_bp)],
        "Heart_Rate": heart_rate,
        "Blood_Sugar": blood_sugar,
        "BMI": bmi,
        "Cholesterol": cholesterol,
        "Medical_History": medical_history,
        "Diagnosis": diagnosis,
        "Medication": medication,
        "Test_Results": test_results,
        "Risk_Category": risk_category
    })
    
    return df

def introduce_data_quality_imperfections(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Introduces missing values, duplicates, and inconsistent casing/formatting
    to demonstrate Step 2 (Data Cleaning) of the Data Science project methodology.
    """
    df_raw = df.copy()
    np.random.seed(random_state)
    n = len(df_raw)
    
    # 1. Missing values (~2.5% in some biometric fields)
    missing_bmi_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df_raw.loc[missing_bmi_idx, "BMI"] = np.nan
    
    missing_sugar_idx = np.random.choice(n, size=int(n * 0.018), replace=False)
    df_raw.loc[missing_sugar_idx, "Blood_Sugar"] = np.nan
    
    missing_chol_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df_raw.loc[missing_chol_idx, "Cholesterol"] = np.nan

    # 2. Inconsistent string entries (e.g. lowercase gender)
    inconsistent_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
    df_raw.loc[inconsistent_idx, "Gender"] = df_raw.loc[inconsistent_idx, "Gender"].str.lower()
    
    # 3. Add duplicates (e.g. 15 duplicated rows)
    duplicates = df_raw.sample(n=15, random_state=random_state)
    df_with_dups = pd.concat([df_raw, duplicates], ignore_index=True)
    
    return df_with_dups

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating base healthcare data...")
    clean_base_df = generate_patient_data(n_samples=1500)
    raw_df = introduce_data_quality_imperfections(clean_base_df)
    
    raw_path = os.path.join(output_dir, "raw_healthcare_dataset.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to: {raw_path} with shape {raw_df.shape}")
