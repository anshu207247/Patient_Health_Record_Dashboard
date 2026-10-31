"""
model_trainer.py
Trains and evaluates the 5 Machine Learning algorithms specified in the Project Synopsis:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbours (KNN)
5. Support Vector Machine (SVM)

Evaluates on:
- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-Score (Weighted)
- ROC-AUC (One-vs-Rest)
- Confusion Matrix
"""

import os
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

# Feature definitions
NUMERICAL_FEATURES = [
    "Age",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar",
    "BMI",
    "Cholesterol",
    "Pulse_Pressure"
]

TARGET_COLUMN = "Risk_Category"

def train_and_evaluate_models(data_path: str, models_save_dir: str) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    
    # 1. Feature Preprocessing
    # Gender one-hot or binary
    df_features = df.copy()
    df_features["Is_Male"] = (df_features["Gender"] == "Male").astype(int)
    features_list = NUMERICAL_FEATURES + ["Is_Male"]
    
    X = df_features[features_list]
    y_raw = df_features[TARGET_COLUMN]
    
    # Label encode target: Low, Moderate, High
    le = LabelEncoder()
    # Ensure ordered classes: Low (0), Moderate (1), High (2)
    le.fit(["Low", "Moderate", "High"])
    y = le.transform(y_raw)
    
    # Train-Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, random_state=42)
    }
    
    metrics_list = []
    confusion_matrices = {}
    trained_model_objects = {}
    roc_curves = {}
    
    high_class_idx = list(le.classes_).index("High")
    y_test_binary_high = (y_test == high_class_idx).astype(int)
    
    for name, clf in models.items():
        # Train
        clf.fit(X_train_scaled, y_train)
        trained_model_objects[name] = clf
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        
        # Probabilities for ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_scaled)
            try:
                auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
                fpr, tpr, _ = roc_curve(y_test_binary_high, y_proba[:, high_class_idx])
                roc_curves[name] = {
                    "fpr": [round(x, 4) for x in fpr.tolist()],
                    "tpr": [round(x, 4) for x in tpr.tolist()],
                    "auc": round(auc(fpr, tpr), 4)
                }
            except Exception:
                auc_score = np.nan
        else:
            auc_score = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc_score, 4) if not np.isnan(auc_score) else "N/A"
        })
        confusion_matrices[name] = cm.tolist()
        
    metrics_df = pd.DataFrame(metrics_list)
    
    # Feature Importance extraction (from Random Forest)
    rf_model = trained_model_objects["Random Forest"]
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": features_list,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Save artifacts
    os.makedirs(models_save_dir, exist_ok=True)
    artifacts = {
        "models": trained_model_objects,
        "scaler": scaler,
        "label_encoder": le,
        "features_list": features_list,
        "metrics_df": metrics_df,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves,
        "classes": le.classes_.tolist(),
        "feature_importances": feat_imp_df,
        "test_data": {
            "X_test": X_test,
            "y_test": y_test.tolist()
        }
    }
    
    artifact_path = os.path.join(models_save_dir, "trained_models_bundle.joblib")
    joblib.dump(artifacts, artifact_path)
    
    # Also save metrics_df to CSV for quick inspection
    metrics_csv_path = os.path.join(models_save_dir, "model_benchmark_results.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    return {
        "metrics_df": metrics_df,
        "feature_importances": feat_imp_df,
        "artifact_path": artifact_path
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    clean_csv = os.path.join(base_dir, "data", "cleaned_healthcare_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    
    if os.path.exists(clean_csv):
        print("Training models...")
        results = train_and_evaluate_models(clean_csv, models_dir)
        print("\n=== Model Benchmark Comparison ===")
        print(results["metrics_df"].to_string(index=False))
    else:
        print("Cleaned dataset not found. Run data_cleaning.py first.")
