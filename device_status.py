import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("model_device.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Device Status Prediction")

# Input fields for sensor data
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
gas_concentration = st.number_input("Gas Concentration")

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[temperature, humidity, pressure, gas_concentration]])
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)

    # Display result
    status = "Normal" if prediction[0] == 0 else "Abnormal"
    st.write(f"### Device Status: *{status}*")