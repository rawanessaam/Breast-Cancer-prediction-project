# 🎗️ Breast Cancer Prediction App using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load model
# ---------------------------
model = joblib.load("models/best_model (1).pkl")

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="🎗️ Breast Cancer Predictor", layout="wide")

st.title("🎗️ Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign (B)** or **Malignant (M)** based on cell nucleus measurements.")

# ---------------------------
# Sidebar input form
# ---------------------------
st.sidebar.header("🧬 Enter Cell Nucleus Data")

def user_input_features():
    radius_mean = st.sidebar.slider("Radius Mean", 5.0, 30.0, 14.0)
    texture_mean = st.sidebar.slider("Texture Mean", 5.0, 40.0, 19.0)
    perimeter_mean = st.sidebar.slider("Perimeter Mean", 40.0, 200.0, 90.0)
    area_mean = st.sidebar.slider("Area Mean", 100.0, 2500.0, 600.0)
    smoothness_mean = st.sidebar.slider("Smoothness Mean", 0.0, 0.2, 0.1)
    compactness_mean = st.sidebar.slider("Compactness Mean", 0.0, 1.0, 0.2)
    concavity_mean = st.sidebar.slider("Concavity Mean", 0.0, 1.0, 0.3)
    concave_points_mean = st.sidebar.slider("Concave Points Mean", 0.0, 0.3, 0.1)
    symmetry_mean = st.sidebar.slider("Symmetry Mean", 0.0, 0.5, 0.18)
    fractal_dimension_mean = st.sidebar.slider("Fractal Dimension Mean", 0.0, 0.2, 0.06)

    data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction Result:")
        if prediction == 1 or prediction == "M":
            st.error("⚠️ Malignant (Cancerous Tumor Detected)")
        else:
            st.success("✅ Benign (Non-Cancerous Tumor)")
    except Exception as e:
        st.warning("⚠️ Error during prediction. Please check your model or input format.")
        st.text(f"Details: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with ❤️ by Team 6 — Breast Cancer Prediction Project")
