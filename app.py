import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Insurance Premium Predictor")
st.write("Enter the details below to get a prediction.")

# Create input fields based on model features: age, Gender, bmi, children, smoker, region
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    # Convert inputs to numeric representations if your model was trained on encoded data
    # (Update these mappings if your model uses different encoding values)
    gender = st.selectbox("Gender", options=["male", "female"])
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["northwest", "northeast", "southwest", "southeast"])

# Encoding mappings (Example - adjust based on how you trained the model)
gender_map = {"female": 0, "male": 1}
smoker_map = {"no": 0, "yes": 1}
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

if st.button("Predict"):
    # Prepare input data in the exact order: age, Gender, bmi, children, smoker, region
    input_data = np.array([[
        age, 
        gender_map[gender], 
        bmi, 
        children, 
        smoker_map[smoker], 
        region_map[region]
    ]])
    
    prediction = model.predict(input_data)
    st.success(f"The predicted value is: ${prediction[0]:,.2f}")
