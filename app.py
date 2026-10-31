"""
app.py - Advanced Interactive Patient Health Record Dashboard
Data Science + Python Healthcare Analytics & Clinical Decision Support System
Author: ANSHU SHAKYA
Domain: Data Science + Python + Machine Learning
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st

# Set global Plotly light theme template
pio.templates.default = "plotly_white"

# Setup Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_cleaning import clean_healthcare_data
from src.prediction_engine import PatientRiskPredictor

# Page Configuration
st.set_page_config(
    page_title="Patient Health Record Dashboard | Data Science & Python",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Clinical Light Theme CSS
st.markdown("""
<style>
    /* Global Clean Light Theme Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    /* Force Light Background across entire app */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.8) !important;
        backdrop-filter: blur(8px);
    }
    
    /* Clean Crisp Light Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.02) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    /* Typography High Contrast in Light Theme */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    p, span, label, div {
        color: #334155;
    }
    
    /* Vibrant Medical Cyan/Blue Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e40af 0%, #0284c7 50%, #0d9488 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: #ffffff !important;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0f2fe !important;
        margin-top: 6px;
        margin-bottom: 14px;
        font-weight: 400;
    }
    
    .hero-badge-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
    }
    
    .hero-pill {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #ffffff !important;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .hero-pill * {
        color: #ffffff !important;
    }
    
    .hero-pill-status {
        background: rgba(16, 185, 129, 0.3) !important;
        border-color: rgba(52, 211, 153, 0.6) !important;
        color: #f0fdf4 !important;
    }
    
    /* Interactive Crisp White KPI Cards */
    .kpi-container {
        background: #ffffff !important;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.05);
        transition: all 0.25s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.1);
        border-color: #3b82f6 !important;
    }
    
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .kpi-icon {
        font-size: 1.4rem;
        background: #f1f5f9;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0;
        line-height: 1.2;
    }
    
    .kpi-subtext {
        font-size: 0.82rem;
        color: #64748b !important;
        margin-top: 4px;
    }
    
    /* Risk Badges in Light Theme */
    .badge-low {
        background-color: #dcfce7 !important;
        color: #15803d !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid #bbf7d0;
    }
    .badge-moderate {
        background-color: #fef9c3 !important;
        color: #a16207 !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid #fef08a;
    }
    .badge-high {
        background-color: #fee2e2 !important;
        color: #b91c1c !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    
    /* Patient Dossier Card */
    .patient-dossier-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-top: 10px;
    }
    
    /* Light Theme Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        padding: 0 16px;
        background-color: transparent;
        font-weight: 600;
        color: #475569 !important;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    }
    
    /* Clean Light Buttons */
    div.stButton > button {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    
    div.stButton > button:hover {
        background-color: #f8fafc;
        border-color: #2563eb;
        color: #2563eb;
    }
    
    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }
    
    /* Dataframes and Tables */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Data Loading with Cache
@st.cache_data
def load_datasets():
    clean_path = os.path.join(BASE_DIR, "data", "cleaned_healthcare_dataset.csv")
    raw_path = os.path.join(BASE_DIR, "data", "raw_healthcare_dataset.csv")
    
    if not os.path.exists(clean_path):
        from src.data_generator import generate_patient_data, introduce_data_quality_imperfections
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        raw_df = introduce_data_quality_imperfections(generate_patient_data(1500))
        raw_df.to_csv(raw_path, index=False)
        clean_df, _ = clean_healthcare_data(raw_df)
        clean_df.to_csv(clean_path, index=False)
    else:
        clean_df = pd.read_csv(clean_path)
        raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else clean_df
        
    return clean_df, raw_df

@st.cache_resource
def load_predictor():
    try:
        return PatientRiskPredictor()
    except Exception:
        from src.model_trainer import train_and_evaluate_models
        train_and_evaluate_models(
            os.path.join(BASE_DIR, "data", "cleaned_healthcare_dataset.csv"),
            os.path.join(BASE_DIR, "models")
        )
        return PatientRiskPredictor()

df_clean_base, df_raw = load_datasets()
predictor = load_predictor()

# Session State for interactive dynamic records and cohort preset filters
if "patient_records" not in st.session_state:
    st.session_state.patient_records = df_clean_base.copy()

if "preset_filter" not in st.session_state:
    st.session_state.preset_filter = "All"

current_df = st.session_state.patient_records

# Top Hero Header
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">🩺 Patient Health Record Dashboard</h1>
            <p class="hero-subtitle">Intelligent Healthcare Analytics, Exploratory Data Science & Multi-Model Predictive ML</p>
        </div>
        <div style="text-align: right; margin-top: 4px;">
            <div class="hero-pill hero-pill-status">● System Live | Pipeline Synchronized</div>
        </div>
    </div>
    <div class="hero-badge-container">
        <div class="hero-pill">👨‍💻 Lead Researcher: <strong>ANSHU SHAKYA</strong></div>
        <div class="hero-pill">⚡ Domain: <strong>Data Science + Python</strong></div>
        <div class="hero-pill">📊 Active Cohort: <strong>{:,} Patients</strong></div>
        <div class="hero-pill">🏆 Top Algorithm: <strong>Random Forest (86.7% Acc | 0.956 AUC)</strong></div>
    </div>
</div>
""".format(len(current_df)), unsafe_allow_html=True)

# Sidebar Navigation & Dynamic Filters
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=64)
    st.markdown("### 🎛️ Navigation Portal")
    page = st.radio(
        "Select Dashboard View:",
        [
            "🩺 Overview & Health KPIs",
            "📊 Exploratory Data Analysis (EDA)",
            "📋 Patient Records Explorer",
            "🤖 ML Model Benchmark & Comparison",
            "🔮 Live Risk & Diagnosis Predictor",
            "📑 Project Synopsis & Methodology"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Global Cohort Filters")
    
    # Age range slider
    min_age, max_age = int(current_df["Age"].min()), int(current_df["Age"].max())
    age_range = st.slider("Patient Age Filter (years):", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    
    # Gender filter
    all_genders = sorted(current_df["Gender"].unique().tolist())
    selected_genders = st.multiselect("Gender Selection:", options=all_genders, default=all_genders)
    
    # Risk Category filter
    all_risks = ["Low", "Moderate", "High"]
    selected_risks = st.multiselect("Risk Classification:", options=all_risks, default=all_risks)
    
    # Diagnosis filter
    all_diagnoses = sorted(current_df["Diagnosis"].unique().tolist())
    selected_diagnoses = st.multiselect("Clinical Diagnoses:", options=all_diagnoses, default=all_diagnoses)
    
    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.preset_filter = "All"
        st.rerun()

# Apply Filters
filtered_df = current_df[
    (current_df["Age"] >= age_range[0]) &
    (current_df["Age"] <= age_range[1]) &
    (current_df["Gender"].isin(selected_genders if selected_genders else all_genders)) &
    (current_df["Risk_Category"].isin(selected_risks if selected_risks else all_risks)) &
    (current_df["Diagnosis"].isin(selected_diagnoses if selected_diagnoses else all_diagnoses))
]

# ==============================================================================
# VIEW 1: Overview & Health KPIs
# ==============================================================================
if page == "🩺 Overview & Health KPIs":
    st.markdown("### ⚡ Interactive Clinical Cohort Presets")
    st.caption("Click any preset button below to rapidly filter and analyze specific patient sub-cohorts:")
    
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
    with pcol1:
        if st.button("🌟 All Patients", use_container_width=True):
            st.session_state.preset_filter = "All"
            st.rerun()
    with pcol2:
        if st.button("🚨 High Risk Cohort", use_container_width=True):
            st.session_state.preset_filter = "High"
            st.rerun()
    with pcol3:
        if st.button("🩸 Diabetic (≥126 mg/dL)", use_container_width=True):
            st.session_state.preset_filter = "Diabetic"
            st.rerun()
    with pcol4:
        if st.button("🫀 Stage 2 HTN (≥140)", use_container_width=True):
            st.session_state.preset_filter = "HTN"
            st.rerun()
    with pcol5:
        if st.button("👴 Seniors (Age 60+)", use_container_width=True):
            st.session_state.preset_filter = "Seniors"
            st.rerun()
    with pcol6:
        if st.button("🛡️ Healthy Baseline", use_container_width=True):
            st.session_state.preset_filter = "Healthy"
            st.rerun()
            
    # Apply Preset filter if active
    preset = st.session_state.preset_filter
    if preset == "High":
        active_cohort_df = filtered_df[filtered_df["Risk_Category"] == "High"]
        st.info("🔎 **Active Preset:** Displaying High Risk Patients only.")
    elif preset == "Diabetic":
        active_cohort_df = filtered_df[filtered_df["Blood_Sugar"] >= 126]
        st.info("🔎 **Active Preset:** Displaying Diabetic Cohort (Fasting Blood Sugar ≥ 126 mg/dL).")
    elif preset == "HTN":
        active_cohort_df = filtered_df[(filtered_df["Systolic_BP"] >= 140) | (filtered_df["Diastolic_BP"] >= 90)]
        st.info("🔎 **Active Preset:** Displaying Patients with Stage 2 Hypertension (BP ≥ 140/90 mmHg).")
    elif preset == "Seniors":
        active_cohort_df = filtered_df[filtered_df["Age"] >= 60]
        st.info("🔎 **Active Preset:** Displaying Senior Citizens (Age ≥ 60 years).")
    elif preset == "Healthy":
        active_cohort_df = filtered_df[filtered_df["Risk_Category"] == "Low"]
        st.info("🔎 **Active Preset:** Displaying Healthy Baseline Cohort (Low Risk).")
    else:
        active_cohort_df = filtered_df

    st.markdown("---")
    
    # Top Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    
    total_pts = len(active_cohort_df)
    high_risk_count = (active_cohort_df["Risk_Category"] == "High").sum() if total_pts > 0 else 0
    pct_high = (high_risk_count / total_pts * 100) if total_pts > 0 else 0
    avg_sys = active_cohort_df["Systolic_BP"].mean() if total_pts > 0 else 0
    avg_dia = active_cohort_df["Diastolic_BP"].mean() if total_pts > 0 else 0
    avg_sugar = active_cohort_df["Blood_Sugar"].mean() if total_pts > 0 else 0
    avg_chol = active_cohort_df["Cholesterol"].mean() if total_pts > 0 else 0
    
    with m1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Cohort Size</span>
                <span class="kpi-icon">👥</span>
            </div>
            <p class="kpi-value">{total_pts:,}</p>
            <p class="kpi-subtext">Active Patient Records</p>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">High Risk Share</span>
                <span class="kpi-icon">🚨</span>
            </div>
            <p class="kpi-value" style="color: {'#ef4444' if pct_high > 30 else '#eab308'};">{pct_high:.1f}%</p>
            <p class="kpi-subtext">{high_risk_count:,} Critical Patients</p>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Mean Blood Pressure</span>
                <span class="kpi-icon">🫀</span>
            </div>
            <p class="kpi-value">{avg_sys:.0f}/{avg_dia:.0f}</p>
            <p class="kpi-subtext">mmHg (Systolic/Diastolic)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Mean Blood Glucose</span>
                <span class="kpi-icon">🩸</span>
            </div>
            <p class="kpi-value">{avg_sugar:.1f}</p>
            <p class="kpi-subtext">mg/dL (Fasting Sugar)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">
                <span class="kpi-label">Mean Cholesterol</span>
                <span class="kpi-icon">🧪</span>
            </div>
            <p class="kpi-value">{avg_chol:.1f}</p>
            <p class="kpi-subtext">mg/dL (Total Serum)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    c1, c2 = st.columns([1.1, 1.4])
    
    with c1:
        st.markdown("#### 🛡️ Patient Risk Stratification")
        risk_counts = active_cohort_df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Category", "Count"]
        colors = {"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"}
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts["Risk_Category"],
            values=risk_counts["Count"],
            hole=0.55,
            marker=dict(colors=[colors.get(x, "#3b82f6") for x in risk_counts["Risk_Category"]]),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )])
        fig_donut.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c2:
        st.markdown("#### 📋 Clinical Condition Prevalence")
        diag_counts = active_cohort_df["Diagnosis"].value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Patients"]
        
        fig_bar = px.bar(
            diag_counts,
            x="Patients",
            y="Diagnosis",
            orientation="h",
            color="Patients",
            color_continuous_scale="Teal",
            text="Patients"
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(t=20, b=20, l=10, r=10),
            coloraxis_showscale=False
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Statistical Metrics Table
    st.markdown("#### 📐 Biomarker Statistical Distributions")
    num_cols = ["Age", "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Sugar", "BMI", "Cholesterol", "Pulse_Pressure"]
    if len(active_cohort_df) > 0:
        stat_summary = active_cohort_df[num_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        stat_summary.columns = ["Mean", "Std Dev", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max"]
        st.dataframe(stat_summary.round(2), use_container_width=True)
    else:
        st.warning("No patient records match current filter criteria.")

# ==============================================================================
# VIEW 2: Exploratory Data Analysis (EDA)
# ==============================================================================
elif page == "📊 Exploratory Data Analysis (EDA)":
    st.markdown("### 📊 Interactive Exploratory Data Analysis (EDA)")
    st.caption("Deep-dive exploratory analytics across physiological distributions, clinical correlations, and 3D parameter spaces.")
    
    eda_tabs = st.tabs([
        "🌐 3D Biometric Space",
        "🫀 Blood Pressure & Cardiac Staging",
        "🩸 Glucose & Cholesterol Interplay",
        "🧬 Correlation Matrix & Dynamic Regressor",
        "👥 Demographics & Cleaning Audit"
    ])
    
    # 3D Space Tab
    with eda_tabs[0]:
        st.markdown("#### 🌐 Interactive 3D Biometric Feature Space")
        st.write("Drag, rotate, and zoom into this 3D physiological space to observe how **Age**, **Systolic BP**, and **Blood Sugar** separate clinical risk tiers:")
        
        fig_3d = px.scatter_3d(
            filtered_df,
            x="Age",
            y="Systolic_BP",
            z="Blood_Sugar",
            color="Risk_Category",
            size="BMI",
            color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"},
            hover_name="Patient_ID",
            hover_data=["Diagnosis", "Cholesterol", "Blood_Pressure"],
            opacity=0.78,
            title="3D Physiological Projection: Age × Systolic BP × Blood Sugar"
        )
        fig_3d.update_layout(
            height=600,
            scene=dict(
                xaxis_title="Age (Years)",
                yaxis_title="Systolic BP (mmHg)",
                zaxis_title="Blood Sugar (mg/dL)"
            ),
            margin=dict(t=30, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("💡 *Tip: Click and drag with mouse to rotate the 3D model; use scroll wheel to zoom in/out.*")
        
    # BP Tab
    with eda_tabs[1]:
        bp1, bp2 = st.columns(2)
        with bp1:
            st.markdown("#### 🫀 Systolic vs. Diastolic (AHA Staging)")
            fig_bp = px.scatter(
                filtered_df,
                x="Systolic_BP",
                y="Diastolic_BP",
                color="BP_Category",
                color_discrete_map={
                    "Normal": "#10b981",
                    "Elevated": "#3b82f6",
                    "Hypertension Stage 1": "#f59e0b",
                    "Hypertension Stage 2": "#ef4444"
                },
                hover_data=["Patient_ID", "Age", "Diagnosis", "Blood_Pressure"]
            )
            # Guideline boundaries
            fig_bp.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="Diastolic 80")
            fig_bp.add_hline(y=90, line_dash="dash", line_color="crimson", annotation_text="Stage 2 Diastolic 90")
            fig_bp.add_vline(x=120, line_dash="dash", line_color="gray", annotation_text="Systolic 120")
            fig_bp.add_vline(x=140, line_dash="dash", line_color="crimson", annotation_text="Stage 2 Systolic 140")
            fig_bp.update_layout(height=400)
            st.plotly_chart(fig_bp, use_container_width=True)
            
        with bp2:
            st.markdown("#### 📈 Pulse Pressure Distribution by Age Group")
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy["Age_Group"] = pd.cut(
                filtered_df_copy["Age"],
                bins=[17, 35, 50, 65, 90],
                labels=["18-35 yrs", "36-50 yrs", "51-65 yrs", "66+ yrs"]
            )
            fig_pulse = px.box(
                filtered_df_copy,
                x="Age_Group",
                y="Pulse_Pressure",
                color="Risk_Category",
                color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"},
                points="outliers"
            )
            fig_pulse.update_layout(height=400, yaxis_title="Pulse Pressure (Sys - Dia mmHg)")
            st.plotly_chart(fig_pulse, use_container_width=True)
            
    # Sugar & Chol Tab
    with eda_tabs[2]:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("#### 🩸 Fasting Glucose vs. Total Cholesterol")
            fig_sc = px.scatter(
                filtered_df,
                x="Blood_Sugar",
                y="Cholesterol",
                color="Risk_Category",
                color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"},
                size="BMI",
                hover_data=["Patient_ID", "Age", "Diagnosis"]
            )
            fig_sc.add_vline(x=126, line_dash="dot", line_color="red", annotation_text="Diabetic Threshold (126)")
            fig_sc.add_hline(y=200, line_dash="dot", line_color="orange", annotation_text="Borderline Chol (200)")
            fig_sc.update_layout(height=400)
            st.plotly_chart(fig_sc, use_container_width=True)
            
        with sc2:
            st.markdown("#### ⚖️ BMI Classification Breakdown")
            bmi_counts = filtered_df["BMI_Category"].value_counts().reset_index()
            bmi_counts.columns = ["BMI_Category", "Count"]
            fig_bmi = px.bar(
                bmi_counts,
                x="BMI_Category",
                y="Count",
                color="BMI_Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text="Count"
            )
            fig_bmi.update_traces(textposition="outside")
            fig_bmi.update_layout(height=400)
            st.plotly_chart(fig_bmi, use_container_width=True)
            
    # Correlation & Dynamic Regressor Tab
    with eda_tabs[3]:
        st.markdown("#### 🧬 Dynamic Parameter Regression & Scatter Explorer")
        st.caption("Select any two biomarkers to investigate correlation, OLS linear regression, and distributions:")
        
        reg_c1, reg_c2, reg_c3 = st.columns(3)
        num_options = ["Age", "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Sugar", "BMI", "Cholesterol", "Pulse_Pressure"]
        with reg_c1:
            x_var = st.selectbox("Select X-Axis Biomarker:", num_options, index=0)
        with reg_c2:
            y_var = st.selectbox("Select Y-Axis Biomarker:", num_options, index=1)
        with reg_c3:
            color_var = st.selectbox("Color By:", ["Risk_Category", "Gender", "Diagnosis"], index=0)
            
        fig_dynamic = px.scatter(
            filtered_df,
            x=x_var,
            y=y_var,
            color=color_var,
            trendline="ols",
            marginal_x="box",
            marginal_y="violin",
            color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"} if color_var == "Risk_Category" else None,
            hover_data=["Patient_ID", "Age", "Diagnosis"]
        )
        fig_dynamic.update_layout(height=460)
        st.plotly_chart(fig_dynamic, use_container_width=True)
        
        st.markdown("#### 📊 Full Pearson Correlation Heatmap")
        corr_matrix = filtered_df[num_options].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            origin="lower"
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # Demographics & Cleaning Audit Tab
    with eda_tabs[4]:
        dem1, dem2 = st.columns(2)
        with dem1:
            st.markdown("#### 👥 Age & Gender Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x="Age",
                color="Gender",
                barmode="overlay",
                nbins=25,
                opacity=0.75
            )
            fig_hist.update_layout(height=360)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with dem2:
            st.markdown("#### 🧹 Step 2: Data Cleaning & Preprocessing Audit")
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px;">
                <h5 style="color: #0f172a; margin-top: 0;">Verified Data Pipeline Summary</h5>
                <p>• <strong>Raw Input Shape:</strong> 1,515 records × 15 features</p>
                <p>• <strong>Duplicates Removed:</strong> 15 duplicate patient profiles pruned</p>
                <p>• <strong>Missing Data Imputation:</strong> 30 BMI, 26 Blood Sugar, 30 Cholesterol records imputed via demographic medians</p>
                <p>• <strong>Residual Missing Values:</strong> <span style="color: green; font-weight: 700;">0 (100% Clean)</span></p>
                <p>• <strong>Outlier Handling:</strong> IQR (1.5 × IQR) boundaries applied</p>
                <p>• <strong>Engineered Biomarkers:</strong> Pulse Pressure, AHA BP Stage, BMI Tiers</p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 3: Patient Records Explorer & Dossier
# ==============================================================================
elif page == "📋 Patient Records Explorer":
    st.markdown("### 📋 Electronic Health Records (EHR) Explorer & Clinical Dossier")
    st.caption("Search, analyze individual patient cases, run interactive What-If scenario simulations, and register new patients.")
    
    # Expandable New Patient Registration Form
    with st.expander("➕ Register New Patient Record & Real-Time Classification", expanded=False):
        st.markdown("##### Clinical Intake Form")
        with st.form("new_patient_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                n_id = st.text_input("Patient ID", f"PAT-{len(st.session_state.patient_records)+1001}")
                n_age = st.number_input("Age", 18, 100, 48)
                n_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with f_col2:
                n_sys = st.number_input("Systolic BP (mmHg)", 85, 200, 132)
                n_dia = st.number_input("Diastolic BP (mmHg)", 50, 120, 84)
                n_hr = st.number_input("Heart Rate (bpm)", 50, 130, 75)
            with f_col3:
                n_sugar = st.number_input("Blood Sugar (mg/dL)", 60, 300, 115)
                n_bmi = st.number_input("BMI", 15.0, 50.0, 27.4, step=0.1)
                n_chol = st.number_input("Cholesterol (mg/dL)", 100, 350, 210)
                
            n_hist = st.selectbox("Medical History", ["None", "Hypertension", "Type 2 Diabetes", "Asthma", "High Cholesterol"])
            n_diag = st.selectbox("Clinical Diagnosis", ["Healthy / Normal", "Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Cardiometabolic Risk"])
            
            submit_patient = st.form_submit_button("⚡ Classify & Append to Live Database", type="primary")
            
            if submit_patient:
                # Run real-time prediction
                pred_res = predictor.predict(
                    age=n_age,
                    gender=n_gender,
                    systolic_bp=n_sys,
                    diastolic_bp=n_dia,
                    heart_rate=n_hr,
                    blood_sugar=n_sugar,
                    bmi=n_bmi,
                    cholesterol=n_chol,
                    model_name="Random Forest"
                )
                
                new_row = {
                    "Patient_ID": n_id,
                    "Age": n_age,
                    "Gender": n_gender,
                    "Systolic_BP": n_sys,
                    "Diastolic_BP": n_dia,
                    "Blood_Pressure": f"{n_sys}/{n_dia}",
                    "Heart_Rate": n_hr,
                    "Blood_Sugar": n_sugar,
                    "BMI": n_bmi,
                    "Cholesterol": n_chol,
                    "Medical_History": n_hist,
                    "Diagnosis": n_diag,
                    "Medication": "Lifestyle Monitoring" if pred_res["predicted_risk"] == "Low" else "Pharmacotherapy",
                    "Test_Results": "Normal" if pred_res["predicted_risk"] == "Low" else "Abnormal",
                    "Risk_Category": pred_res["predicted_risk"],
                    "BMI_Category": "Normal" if n_bmi < 25 else ("Overweight" if n_bmi < 30 else "Obese"),
                    "BP_Category": "Normal" if n_sys < 120 and n_dia < 80 else "Hypertension Stage 1",
                    "Pulse_Pressure": n_sys - n_dia
                }
                st.session_state.patient_records = pd.concat([st.session_state.patient_records, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"✅ Patient {n_id} successfully classified as **{pred_res['predicted_risk']} Risk** and added to EHR registry!")
                st.rerun()

    # Search & Table
    search_term = st.text_input("🔍 Instant Search Patient Records (by ID, Diagnosis, Medication, or Condition):", "")
    
    display_df = filtered_df.copy()
    if search_term:
        mask = (
            display_df["Patient_ID"].astype(str).str.contains(search_term, case=False) |
            display_df["Diagnosis"].astype(str).str.contains(search_term, case=False) |
            display_df["Medication"].astype(str).str.contains(search_term, case=False) |
            display_df["Medical_History"].astype(str).str.contains(search_term, case=False)
        )
        display_df = display_df[mask]
        
    st.write(f"Displaying **{len(display_df):,}** patient records:")
    
    cols_to_show = [
        "Patient_ID", "Age", "Gender", "Blood_Pressure", "Heart_Rate",
        "Blood_Sugar", "BMI", "Cholesterol", "Diagnosis", "Medication",
        "Test_Results", "Risk_Category"
    ]
    st.dataframe(display_df[cols_to_show], use_container_width=True, height=280)
    
    # CSV Download
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Filtered EHR Records to CSV",
        data=csv_data,
        file_name="patient_health_records.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### 🔬 Individual Patient Dossier & 'What-If' Health Optimizer")
    
    patient_ids = display_df["Patient_ID"].tolist()
    if patient_ids:
        selected_id = st.selectbox("Select Patient to Inspect Clinical Dossier:", patient_ids)
        p_data = display_df[display_df["Patient_ID"] == selected_id].iloc[0]
        
        # Patient Card
        p1, p2, p3, p4 = st.columns([1.2, 1.8, 1.8, 2.2])
        with p1:
            avatar = "https://img.icons8.com/color/96/user-male-circle.png" if p_data["Gender"] == "Male" else "https://img.icons8.com/color/96/user-female-circle.png"
            st.image(avatar, width=70)
            st.markdown(f"### {p_data['Patient_ID']}")
            st.write(f"**Age:** {p_data['Age']} yrs | **Sex:** {p_data['Gender']}")
            risk_class = f"badge-{p_data['Risk_Category'].lower()}"
            st.markdown(f"<span class='{risk_class}'>{p_data['Risk_Category']} Risk Tier</span>", unsafe_allow_html=True)
            
        with p2:
            st.markdown("##### 🩺 Vital Parameters")
            st.write(f"• **Blood Pressure:** `{p_data['Blood_Pressure']} mmHg`")
            st.write(f"• **Pulse Pressure:** `{p_data['Pulse_Pressure']} mmHg`")
            st.write(f"• **Heart Rate:** `{p_data['Heart_Rate']} bpm`")
            
        with p3:
            st.markdown("##### 🧪 Metabolic Profile")
            st.write(f"• **Blood Sugar:** `{p_data['Blood_Sugar']} mg/dL`")
            st.write(f"• **Cholesterol:** `{p_data['Cholesterol']} mg/dL`")
            st.write(f"• **BMI:** `{p_data['BMI']} kg/m²` ({p_data['BMI_Category']})")
            
        with p4:
            st.markdown("##### 📋 Clinical Management")
            st.write(f"• **Diagnosis:** `{p_data['Diagnosis']}`")
            st.write(f"• **Medical History:** `{p_data['Medical_History']}`")
            st.write(f"• **Prescription:** `{p_data['Medication']}`")
            st.write(f"• **Lab Result Flag:** `{p_data['Test_Results']}`")
            
        # Interactive What-If Scenario Simulator for this Patient
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"🔮 Interactive 'What-If' Health Optimizer for {p_data['Patient_ID']}", expanded=True):
            st.markdown("Simulate how lifestyle interventions or pharmacotherapy would alter this patient's predicted risk score in real-time:")
            
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                delta_bp = st.slider("Blood Pressure Reduction (mmHg):", 0, 40, 15)
            with w_col2:
                delta_sugar = st.slider("Blood Sugar Reduction (mg/dL):", 0, 80, 25)
            with w_col3:
                delta_bmi = st.slider("BMI Reduction (kg/m²):", 0.0, 8.0, 2.5, step=0.1)
                
            sim_sys = max(90, p_data["Systolic_BP"] - delta_bp)
            sim_dia = max(60, p_data["Diastolic_BP"] - (delta_bp * 0.5))
            sim_sugar = max(75, p_data["Blood_Sugar"] - delta_sugar)
            sim_bmi = max(18.5, p_data["BMI"] - delta_bmi)
            
            sim_pred = predictor.predict(
                age=p_data["Age"],
                gender=p_data["Gender"],
                systolic_bp=sim_sys,
                diastolic_bp=sim_dia,
                heart_rate=p_data["Heart_Rate"],
                blood_sugar=sim_sugar,
                bmi=sim_bmi,
                cholesterol=p_data["Cholesterol"],
                model_name="Random Forest"
            )
            
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 18px; margin-top: 10px;">
                <h5 style="color: #166534; margin: 0 0 6px 0;">⚡ Real-Time Optimization Outcome:</h5>
                <p style="margin: 0; color: #1e293b;">
                    • <strong>Current Risk:</strong> <span class='badge-{p_data['Risk_Category'].lower()}'>{p_data['Risk_Category']} Risk</span> ➔ 
                    <strong>Optimized Simulated Risk:</strong> <span class='badge-{sim_pred['predicted_risk'].lower()}'>{sim_pred['predicted_risk']} Risk</span>
                    (High Risk Probability: <strong>{sim_pred['probabilities'].get('High', 0):.1f}%</strong>)
                </p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 4: ML Model Benchmark & Comparison
# ==============================================================================
elif page == "🤖 ML Model Benchmark & Comparison":
    st.markdown("### 🤖 Machine Learning Model Benchmark & Comparison")
    st.markdown("""
    In accordance with the **Project Synopsis Report (Page 4 & 5)**, five classification algorithms were trained and compared:
    `Logistic Regression`, `Decision Tree`, `Random Forest`, `K-Nearest Neighbours (KNN)`, and `Support Vector Machine (SVM)`.
    """)
    
    bundle = predictor.bundle
    metrics_df = bundle["metrics_df"]
    
    # Leaderboard Table
    st.markdown("#### 🏆 Algorithmic Benchmark Leaderboard")
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="#bbf7d0"
        ),
        use_container_width=True
    )
    
    # Multi-Model ROC Curves & Metric Comparison Tabs
    ml_tabs = st.tabs([
        "📈 Multi-Model ROC Curves",
        "🎯 Model Comparison Radar & Bars",
        "🎛️ Interactive Threshold Tuner & Confusion Matrix",
        "🧬 Feature Importance Rankings"
    ])
    
    # ROC Curves Tab
    with ml_tabs[0]:
        st.markdown("#### 📈 Multi-Model Receiver Operating Characteristic (ROC) Comparison")
        st.write("Evaluating True Positive Rate vs False Positive Rate for High Risk classification across all 5 algorithms:")
        
        fig_roc = go.Figure()
        colors_map = {
            "Random Forest": "#10b981",
            "SVM": "#3b82f6",
            "KNN": "#8b5cf6",
            "Decision Tree": "#f59e0b",
            "Logistic Regression": "#ef4444"
        }
        
        roc_data = bundle.get("roc_curves", {})
        for model_name, curve_info in roc_data.items():
            fig_roc.add_trace(go.Scatter(
                x=curve_info["fpr"],
                y=curve_info["tpr"],
                mode="lines",
                name=f"{model_name} (AUC = {curve_info['auc']:.4f})",
                line=dict(width=2.5, color=colors_map.get(model_name, None))
            ))
            
        # Random guess line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random Baseline (AUC = 0.50)"
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=480,
            hovermode="x unified",
            legend=dict(x=0.6, y=0.15)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info("🏆 **Conclusion:** Random Forest dominates the ROC curve with an **AUC of 0.9556**, indicating exceptional discriminative capacity for high-risk triage.")
        
    # Radar & Bar Tab
    with ml_tabs[1]:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### 🎯 Multi-Metric Comparison")
            melted = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
            melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
            fig_bars = px.bar(
                melted,
                x="Model",
                y="value",
                color="variable",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_bars.update_layout(yaxis_range=[0.7, 1.0], height=380)
            st.plotly_chart(fig_bars, use_container_width=True)
            
        with r2:
            st.markdown("#### 🕸️ Algorithmic Performance Radar")
            radar_fig = go.Figure()
            categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            for _, row in metrics_df.iterrows():
                vals = [row[c] for c in categories]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories,
                    fill="toself",
                    name=row["Model"]
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.75, 1.0])),
                height=380,
                margin=dict(t=25, b=25, l=25, r=25)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    # Threshold Tuner & Confusion Matrix
    with ml_tabs[2]:
        st.markdown("#### 🎛️ Interactive Decision Threshold Tuner")
        st.caption("Adjust the classification threshold to observe the dynamic trade-off between sensitivity and specificity for High Risk detection:")
        
        th_col1, th_col2 = st.columns([1, 2])
        with th_col1:
            threshold = st.slider("Risk Decision Cutoff Threshold:", 0.10, 0.90, 0.50, 0.05)
            st.write(f"• Current Cutoff: `{threshold:.2f}`")
            st.write("• Lower threshold ➔ Higher Recall (Catch more high risk patients)")
            st.write("• Higher threshold ➔ Higher Precision (Reduce false alarms)")
            
        with th_col2:
            model_cm_name = st.selectbox("Inspect Confusion Matrix for Model:", metrics_df["Model"].tolist(), index=2)
            cm_arr = np.array(bundle["confusion_matrices"][model_cm_name])
            classes = bundle["classes"]
            fig_cm = px.imshow(
                cm_arr,
                x=classes,
                y=classes,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Risk", y="Actual Risk")
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance Tab
    with ml_tabs[3]:
        st.markdown("#### 🧬 Biomarker Feature Importance (Random Forest Gini Impurity)")
        feat_df = bundle["feature_importances"]
        fig_imp = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            text="Importance"
        )
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.info("💡 **Clinical Significance:** Blood Sugar, Systolic BP, and BMI account for over **70% of total predictive importance**, matching real-world clinical cardiovascular risk models.")

# ==============================================================================
# VIEW 5: Live Risk & Diagnosis Predictor
# ==============================================================================
elif page == "🔮 Live Risk & Diagnosis Predictor":
    st.markdown("### 🔮 Real-Time Patient Risk & Clinical Decision Support")
    st.markdown("Enter patient physiological indicators to generate live clinical risk stratification, confidence scores, and evidence-based guidance.")
    
    in_col1, in_col2, in_col3 = st.columns(3)
    with in_col1:
        st.markdown("##### 👥 Demographics & Vitals")
        p_age = st.slider("Patient Age (Years):", 18, 100, 54)
        p_gender = st.selectbox("Biological Sex:", ["Male", "Female", "Other"])
        p_sys = st.slider("Systolic Blood Pressure (mmHg):", 85, 200, 142)
        p_dia = st.slider("Diastolic Blood Pressure (mmHg):", 50, 120, 92)
        
    with in_col2:
        st.markdown("##### 🩸 Metabolic Markers")
        p_hr = st.slider("Heart Rate (bpm):", 45, 130, 80)
        p_sugar = st.slider("Fasting Blood Sugar (mg/dL):", 60, 300, 145)
        p_bmi = st.slider("Body Mass Index (BMI kg/m²):", 15.0, 48.0, 30.5, step=0.1)
        p_chol = st.slider("Total Cholesterol (mg/dL):", 110, 350, 235)
        
    with in_col3:
        st.markdown("##### 🤖 Algorithm Selection")
        selected_model = st.selectbox(
            "Prediction Model:",
            ["Random Forest", "Support Vector Machine (SVM)", "Decision Tree", "Logistic Regression", "KNN"],
            index=0
        )
        clean_model_name = "SVM" if "SVM" in selected_model else selected_model
        
        st.markdown("""
        <div style="background: #f1f5f9; padding: 12px 14px; border-radius: 8px; margin-top: 16px; border-left: 4px solid #3b82f6;">
            <p style="margin: 0; font-size: 0.85rem; color: #334155;">
                🏆 <strong>Recommended:</strong> Random Forest achieves the top benchmark F1-Score (0.8667) and ROC-AUC (0.9556).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        predict_click = st.button("⚡ Run Real-Time Clinical Evaluation", type="primary", use_container_width=True)

    if predict_click:
        st.markdown("---")
        res = predictor.predict(
            age=p_age,
            gender=p_gender,
            systolic_bp=p_sys,
            diastolic_bp=p_dia,
            heart_rate=p_hr,
            blood_sugar=p_sugar,
            bmi=p_bmi,
            cholesterol=p_chol,
            model_name=clean_model_name
        )
        
        predicted_risk = res["predicted_risk"]
        probs = res["probabilities"]
        high_prob = probs.get("High", 0.0)
        
        out1, out2 = st.columns([1.3, 1.7])
        with out1:
            st.markdown("#### ⚡ Clinical Risk Assessment")
            if predicted_risk == "High":
                st.error(f"### ⚠️ PREDICTED TIER: HIGH RISK")
            elif predicted_risk == "Moderate":
                st.warning(f"### ⚡ PREDICTED TIER: MODERATE RISK")
            else:
                st.success(f"### ✅ PREDICTED TIER: LOW RISK")
                
            # Animated Speedometer Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_prob,
                title={"text": "High Risk Probability Score (%)", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if high_prob > 60 else ("#f59e0b" if high_prob > 30 else "#10b981")},
                    "steps": [
                        {"range": [0, 33], "color": "#dcfce7"},
                        {"range": [33, 66], "color": "#fef9c3"},
                        {"range": [66, 100], "color": "#fee2e2"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 66
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(t=30, b=10, l=25, r=25))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.write(f"**Model Used:** `{clean_model_name}`")
            for cat, val in probs.items():
                st.write(f"• **{cat} Risk Probability:** `{val:.1f}%`")
                st.progress(val / 100.0)
                
        with out2:
            st.markdown("#### 🕸️ Biometric Comparison vs. Healthy Baseline")
            categories = ["Systolic BP", "Diastolic BP", "Blood Sugar", "BMI", "Cholesterol", "Heart Rate"]
            patient_vals = [
                min(100, max(0, (p_sys - 80) / 1.2)),
                min(100, max(0, (p_dia - 50) / 0.7)),
                min(100, max(0, (p_sugar - 70) / 2.0)),
                min(100, max(0, (p_bmi - 18.5) / 0.25)),
                min(100, max(0, (p_chol - 120) / 2.2)),
                min(100, max(0, (p_hr - 50) / 0.75))
            ]
            normal_vals = [33, 42, 25, 26, 36, 33]
            
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=patient_vals,
                theta=categories,
                fill="toself",
                name="Patient Physiological Profile",
                line_color="#ef4444" if predicted_risk == "High" else "#3b82f6"
            ))
            radar.add_trace(go.Scatterpolar(
                r=normal_vals,
                theta=categories,
                fill="toself",
                name="Healthy Clinical Midpoint",
                line_color="#10b981",
                opacity=0.4
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
                height=340,
                margin=dict(t=25, b=25, l=25, r=25)
            )
            st.plotly_chart(radar, use_container_width=True)
            
        # Clinical Guidance
        st.markdown("---")
        flag_c, rec_c = st.columns(2)
        with flag_c:
            st.markdown("##### 🚨 Abnormal Biomarker Alerts")
            if res["biomarker_flags"]:
                for fl in res["biomarker_flags"]:
                    st.markdown(f"• ⚠️ **{fl}**")
            else:
                st.markdown("• ✅ All primary biometric values are within normal adult reference limits.")
                
        with rec_c:
            st.markdown("##### 🩺 Evidence-Based Clinical Recommendations")
            for rec in res["recommendations"]:
                st.markdown(f"• {rec}")
                
        # Generate Downloadable Patient Summary
        summary_text = f"""PATIENT CLINICAL EVALUATION REPORT
==================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Demographics: Age {p_age} | Sex {p_gender}
Vital Signs: BP {p_sys}/{p_dia} mmHg | HR {p_hr} bpm
Metabolic Profile: Fasting Glucose {p_sugar} mg/dL | BMI {p_bmi} | Cholesterol {p_chol} mg/dL
Predicted Risk Tier: {predicted_risk.upper()} RISK
Model Applied: {clean_model_name}
Probability Breakdown: {probs}
Biomarker Alerts: {', '.join(res['biomarker_flags']) if res['biomarker_flags'] else 'None'}
Clinical Recommendations: {', '.join(res['recommendations'])}
==================================================
Medical Disclaimer: Data-science prediction tool only. Not a certified substitute for professional medical diagnosis.
"""
        st.download_button(
            "📥 Download Patient Clinical Evaluation Summary (TXT)",
            data=summary_text,
            file_name="patient_clinical_summary.txt",
            mime="text/plain"
        )
        
        st.caption("⚖️ **Medical Disclaimer:** *The model is presented as a data-science prediction tool, not as a substitute for professional medical diagnosis or clinical judgment.*")

# ==============================================================================
# VIEW 6: Project Synopsis & Methodology
# ==============================================================================
elif page == "📑 Project Synopsis & Methodology":
    st.markdown("### 📑 Data Science Project Synopsis Report")
    st.markdown("#### **Title:** Patient Health Record Dashboard")
    st.markdown("**Author / Candidate:** ANSHU SHAKYA | **Domain:** Data Science + Python + Machine Learning")
    
    st.markdown("---")
    syn_c1, syn_c2 = st.columns(2)
    with syn_c1:
        st.markdown("##### 📌 Introduction & Problem Statement")
        st.write("""
        Healthcare organizations generate large volumes of patient data including age, gender, blood pressure,
        blood sugar, heart rate, medical history, diagnoses, medications, and laboratory results. When stored
        in raw tables, it is difficult to identify patterns, clinical correlations, and at-risk cohorts.
        
        The **Patient Health Record Dashboard** converts raw patient data into interactive visualizations,
        statistical summaries, and machine-learning-driven predictive insights.
        """)
        
        st.markdown("##### 🎯 10 Core Objectives")
        st.markdown("""
        1. Collect and preprocess patient healthcare data.
        2. Clean missing, duplicate, and inconsistent data.
        3. Perform Exploratory Data Analysis (EDA) on patient records.
        4. Identify relationships between different health parameters.
        5. Create interactive graphs and visualizations.
        6. Provide patient-level and dataset-level health insights.
        7. Apply suitable Machine Learning algorithms for prediction/classification.
        8. Evaluate the performance of developed ML models.
        9. Provide an easy-to-use dashboard for viewing healthcare information.
        10. Demonstrate the application of data science techniques in healthcare analytics.
        """)
        
    with syn_c2:
        st.markdown("##### ⚙️ 7-Stage Methodology Pipeline")
        st.markdown("""
        ```
        Step 1: Data Collection (Structured clinical cohort)
             ↓
        Step 2: Data Cleaning (Deduplication, Median Imputation, Outlier Handling)
             ↓
        Step 3: Data Preprocessing (Binary Encoding, StandardScaler)
             ↓
        Step 4: Exploratory Data Analysis (EDA, AHA BP Staging, Heatmaps)
             ↓
        Step 5: Machine Learning (5 Models: LR, DT, RF, KNN, SVM)
             ↓
        Step 6: Evaluation & Benchmarking (Accuracy, Precision, Recall, F1, ROC-AUC)
             ↓
        Step 7: Interactive Dashboard & Live Predictor Deployment
        ```
        """)
        
        st.markdown("##### 🛠️ Technology Stack")
        st.markdown("""
        • **Programming Language:** Python 3.13  
        • **Data Manipulation:** Pandas, NumPy  
        • **Machine Learning:** Scikit-learn  
        • **Visualization:** Plotly Express & Graph Objects, Seaborn, Matplotlib  
        • **Interactive Web Application:** Streamlit  
        • **Model Persistence:** Joblib  
        """)

    st.markdown("---")
    st.markdown("##### 📋 Dataset Attributes Specification")
    attr_df = pd.DataFrame([
        {"Attribute": "Patient ID", "Description": "Unique identifier of patient (PAT-XXXX)"},
        {"Attribute": "Age", "Description": "Age of patient in completed years (18 - 85)"},
        {"Attribute": "Gender", "Description": "Biological sex: Male / Female / Other"},
        {"Attribute": "Blood Pressure", "Description": "Systolic and diastolic pressures (mmHg)"},
        {"Attribute": "Heart Rate", "Description": "Resting pulse rate (bpm)"},
        {"Attribute": "Blood Sugar", "Description": "Fasting blood glucose concentration (mg/dL)"},
        {"Attribute": "BMI", "Description": "Body Mass Index (kg/m²)"},
        {"Attribute": "Cholesterol", "Description": "Total serum cholesterol level (mg/dL)"},
        {"Attribute": "Medical History", "Description": "Documented prior conditions (e.g., Hypertension, Diabetes)"},
        {"Attribute": "Diagnosis", "Description": "Primary clinical diagnosis"},
        {"Attribute": "Medication", "Description": "Prescribed medication"},
        {"Attribute": "Test Results", "Description": "Laboratory test outcome: Normal / Borderline / Abnormal"},
        {"Attribute": "Risk Category", "Description": "Clinical risk tier: Low / Moderate / High Risk"}
    ])
    st.table(attr_df)
    st.caption("Developed by ANSHU SHAKYA | Data Science + Python Healthcare Project")
