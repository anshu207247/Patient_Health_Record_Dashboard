"""
prediction_engine.py
Inference and Clinical Decision Support engine.
Loads trained models, predicts patient disease-risk category,
computes risk probabilities, compares vitals with normal reference ranges,
and generates tailored clinical guidance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

class PatientRiskPredictor:
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            bundle_path = os.path.join(base_dir, "models", "trained_models_bundle.joblib")
        
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Run model_trainer.py first.")
            
        self.bundle = joblib.load(bundle_path)
        self.models = self.bundle["models"]
        self.scaler = self.bundle["scaler"]
        self.le = self.bundle["label_encoder"]
        self.features_list = self.bundle["features_list"]
        self.metrics_df = self.bundle["metrics_df"]
        self.classes = self.bundle["classes"]
        
    def predict(
        self,
        age: float,
        gender: str,
        systolic_bp: float,
        diastolic_bp: float,
        heart_rate: float,
        blood_sugar: float,
        bmi: float,
        cholesterol: float,
        model_name: str = "Random Forest"
    ) -> Dict[str, Any]:
        """
        Runs clinical risk prediction for an individual patient.
        """
        pulse_pressure = systolic_bp - diastolic_bp
        is_male = 1 if gender.lower() == "male" else 0
        
        # Build feature vector in exact order
        input_data = pd.DataFrame([{
            "Age": age,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp,
            "Heart_Rate": heart_rate,
            "Blood_Sugar": blood_sugar,
            "BMI": bmi,
            "Cholesterol": cholesterol,
            "Pulse_Pressure": pulse_pressure,
            "Is_Male": is_male
        }])
        
        # Scale input
        scaled_input = self.scaler.transform(input_data[self.features_list])
        
        # Selected Model
        clf = self.models.get(model_name, self.models["Random Forest"])
        pred_label_idx = clf.predict(scaled_input)[0]
        predicted_category = self.le.inverse_transform([pred_label_idx])[0]
        
        # Probabilities
        probabilities = {}
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(scaled_input)[0]
            for cls_name, p in zip(self.classes, probs):
                probabilities[cls_name] = round(float(p) * 100, 2)
        else:
            probabilities[predicted_category] = 100.0
            
        # Clinical Risk Analysis & Biomarker flags
        flags = []
        if systolic_bp >= 140 or diastolic_bp >= 90:
            flags.append(f"Hypertension Stage 2 ({int(systolic_bp)}/{int(diastolic_bp)} mmHg)")
        elif systolic_bp >= 130 or diastolic_bp >= 80:
            flags.append(f"Hypertension Stage 1 ({int(systolic_bp)}/{int(diastolic_bp)} mmHg)")
            
        if blood_sugar >= 126:
            flags.append(f"High Blood Glucose ({int(blood_sugar)} mg/dL - Diabetic range)")
        elif blood_sugar >= 100:
            flags.append(f"Elevated Fasting Glucose ({int(blood_sugar)} mg/dL - Prediabetic range)")
            
        if cholesterol >= 240:
            flags.append(f"High Cholesterol ({int(cholesterol)} mg/dL - Hypercholesterolemia)")
        elif cholesterol >= 200:
            flags.append(f"Borderline High Cholesterol ({int(cholesterol)} mg/dL)")
            
        if bmi >= 30:
            flags.append(f"Obesity (BMI {bmi:.1f})")
        elif bmi >= 25:
            flags.append(f"Overweight (BMI {bmi:.1f})")
            
        # Actionable clinical recommendations
        recommendations = []
        if predicted_category == "High":
            recommendations.append("Priority consultation with Primary Care Physician or Cardiologist recommended.")
            recommendations.append("Comprehensive metabolic panel, HbA1c testing, and lipid panel follow-up.")
            recommendations.append("Continuous ambulatory BP monitoring and targeted pharmacological therapy.")
        elif predicted_category == "Moderate":
            recommendations.append("Schedule routine health checkup within 30 to 60 days.")
            recommendations.append("Lifestyle modifications: DASH diet, aerobic exercise 150 min/week, and sodium restriction.")
            recommendations.append("Re-test fasting blood glucose and lipid panel in 3 months.")
        else:
            recommendations.append("Vitals within normal limits. Maintain balanced Mediterranean or DASH diet.")
            recommendations.append("Annual wellness and preventative screening recommended.")
            recommendations.append("Maintain active physical lifestyle and adequate sleep hygiene.")
            
        return {
            "predicted_risk": predicted_category,
            "probabilities": probabilities,
            "model_used": model_name,
            "biomarker_flags": flags,
            "recommendations": recommendations,
            "raw_inputs": {
                "Age": age,
                "Gender": gender,
                "Blood Pressure": f"{int(systolic_bp)}/{int(diastolic_bp)} mmHg",
                "Heart Rate": f"{int(heart_rate)} bpm",
                "Blood Sugar": f"{int(blood_sugar)} mg/dL",
                "BMI": f"{bmi:.1f}",
                "Cholesterol": f"{int(cholesterol)} mg/dL"
            }
        }
