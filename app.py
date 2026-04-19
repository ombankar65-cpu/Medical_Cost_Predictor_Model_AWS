import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏥 Insurance Charges Predictor")
st.markdown("Fill in the details below to predict the estimated insurance charge.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    children = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)

with col2:
    gender = st.selectbox("Gender", options=["male", "female"])
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# ── Encode categoricals ───────────────────────────────────────────────────────
# These mappings must match what was used during model training
gender_map   = {"male": 1, "female": 0}
smoker_map   = {"yes": 1, "no": 0}
region_map   = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

st.divider()

if st.button("🔍 Predict Charges", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "age":      age,
        "Gender":   gender_map[gender],
        "bmi":      bmi,
        "children": children,
        "smoker":   smoker_map[smoker],
        "region":   region_map[region],
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Insurance Charge: ₹ {prediction:,.2f}")

    st.divider()
    st.subheader("📋 Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Gender", "BMI", "Children", "Smoker", "Region"],
        "Value":   [age, gender, bmi, children, smoker, region],
    })
    st.table(summary.set_index("Feature"))

st.markdown("---")
st.caption("Model: Decision Tree Regressor · Features: age, gender, bmi, children, smoker, region")
