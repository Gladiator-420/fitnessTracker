import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
def load_model():
    with open("fitness_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit UI
st.title("🏋️ Personal Fitness Tracker")
st.write("Fill in your details to estimate the calories you'll burn!")

# Sidebar Inputs
age = st.slider("Age:", 10, 100, 25)
bmi = st.slider("BMI:", 15, 40, 22)
duration = st.slider("Workout Duration (min):", 0, 60, 30)
heart_rate = st.slider("Heart Rate (bpm):", 60, 130, 80)
body_temp = st.slider("Body Temperature (°C):", 36, 42, 37)

gender = st.radio("Gender:", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

# Display User Inputs
data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Duration": [duration],
    "Heart Rate": [heart_rate],
    "Body Temp": [body_temp],
    "Gender_Male": [gender_val]
})
st.write("### Your Input Data:", data)

# Make Prediction
prediction = model.predict(data)[0]
st.write("## Estimated Calories Burned:")
st.write(f"🔥 You are expected to burn around **{round(prediction, 2)} kcal**!")  
