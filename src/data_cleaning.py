"""
data_cleaning.py
Implements Step 2 (Data Cleaning) and Step 3 (Data Preprocessing)
from the Project Synopsis:
- Handle missing values (median imputation)
- Remove duplicate records
- Correct inconsistent string cases
- Detect and handle outliers
- Feature engineering: BP categories, BMI categories, Pulse Pressure
- Prepare preprocessed features for Machine Learning models
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def clean_healthcare_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw patient healthcare data and returns cleaned dataframe plus summary metrics.
    """
    df = raw_df.copy()
    report = {}
    
    # 1. Duplicate detection and removal
    initial_rows = len(df)
    duplicates_count = int(df.duplicated(subset=["Patient_ID"]).sum())
    df = df.drop_duplicates(subset=["Patient_ID"], keep="first").reset_index(drop=True)
    report["duplicates_removed"] = duplicates_count
    
    # 2. Inconsistent string cleaning
    df["Gender"] = df["Gender"].astype(str).str.capitalize().str.strip()
    df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})
    # Retain valid genders
    valid_genders = ["Male", "Female", "Other"]
    df["Gender"] = df["Gender"].apply(lambda g: g if g in valid_genders else "Other")
    
    # 3. Missing Value Handling
    missing_report = df.isnull().sum().to_dict()
    report["missing_values_before"] = {k: int(v) for k, v in missing_report.items() if v > 0}
    
    # Impute BMI with median by Gender & Age bracket
    if df["BMI"].isnull().any():
        df["BMI"] = df.groupby(["Gender"])["BMI"].transform(lambda x: x.fillna(x.median()))
        df["BMI"] = df["BMI"].fillna(df["BMI"].median())
        
    # Impute Blood Sugar with median
    if df["Blood_Sugar"].isnull().any():
        df["Blood_Sugar"] = df.groupby(["Gender"])["Blood_Sugar"].transform(lambda x: x.fillna(x.median()))
        df["Blood_Sugar"] = df["Blood_Sugar"].fillna(df["Blood_Sugar"].median())
        
    # Impute Cholesterol with median
    if df["Cholesterol"].isnull().any():
        df["Cholesterol"] = df.groupby(["Gender"])["Cholesterol"].transform(lambda x: x.fillna(x.median()))
        df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
        
    # Handle Medical_History and Medication textual representations
    df["Medical_History"] = df["Medical_History"].fillna("No Prior Chronic Condition")
    df["Medication"] = df["Medication"].fillna("No Active Prescriptions")
    df["Test_Results"] = df["Test_Results"].fillna("Normal")
    
    report["missing_values_after"] = int(df.isnull().sum().sum())
    
    # 4. Outlier Detection and Handling (IQR Capping on physiological parameters)
    outlier_summary = {}
    numerical_cols = ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Sugar", "BMI", "Cholesterol"]
    for col in numerical_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)
        upper_bound = Q3 + 1.5 * IQR
        outliers_detected = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_summary[col] = outliers_detected
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    report["outliers_detected_and_clipped"] = outlier_summary
    
    # 5. Feature Engineering / Health Categorization
    # BMI Category
    def get_bmi_cat(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25.0:
            return "Normal"
        elif bmi < 30.0:
            return "Overweight"
        else:
            return "Obese"
    df["BMI_Category"] = df["BMI"].apply(get_bmi_cat)
    
    # BP Category according to AHA (American Heart Association)
    def get_bp_cat(row):
        sys = row["Systolic_BP"]
        dia = row["Diastolic_BP"]
        if sys < 120 and dia < 80:
            return "Normal"
        elif 120 <= sys <= 129 and dia < 80:
            return "Elevated"
        elif (130 <= sys <= 139) or (80 <= dia <= 89):
            return "Hypertension Stage 1"
        elif sys >= 140 or dia >= 90:
            return "Hypertension Stage 2"
        return "Normal"
    df["BP_Category"] = df.apply(get_bp_cat, axis=1)
    
    # Pulse Pressure (Systolic - Diastolic)
    df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]
    
    # Consistent Blood Pressure string
    df["Blood_Pressure"] = df["Systolic_BP"].astype(int).astype(str) + "/" + df["Diastolic_BP"].astype(int).astype(str)
    
    report["total_cleaned_records"] = len(df)
    return df, report

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, "data", "raw_healthcare_dataset.csv")
    clean_path = os.path.join(base_dir, "data", "cleaned_healthcare_dataset.csv")
    
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        clean_df, report = clean_healthcare_data(raw_df)
        clean_df.to_csv(clean_path, index=False)
        print("Data cleaning completed successfully.")
        print("Cleaning Summary:", report)
    else:
        print("Raw dataset not found. Run data_generator.py first.")
