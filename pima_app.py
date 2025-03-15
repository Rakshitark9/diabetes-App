#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pima_app.py
import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction App")

# Define user inputs
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=10, max_value=100, value=33)

# Create input array
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success("Prediction: " + ("Diabetic" if prediction[0] == 1 else "Not Diabetic"))

