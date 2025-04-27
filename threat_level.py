import streamlit as st
import pickle
import numpy as np

# Load the scaler
with open('threat_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the model
with open('threat_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set background image
def set_bg():
    bg_image_url = "https://img.freepik.com/free-photo/hacker-working-darkness_53876-94580.jpg?ga=GA1.1.292118901.1745739635&semt=ais_hybrid&w=740"
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg()

# Titles
st.markdown("<h1 style='color:red; font-weight:bold;'>🛡️ Threat Level Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gold; font-weight:bold;'>Enter features below to check the threat level.</p>", unsafe_allow_html=True)

# Feature names
feature_names = ["Temperature", "Noise Level", "Crowd Density", "Incident Report", "CCTV Alert", "Internet Traffic"]

# Class labels
class_labels = {0: "Safe", 1: "Moderate", 2: "Critical"}

# Create input fields
user_input = []
for feature in feature_names:
    val = st.number_input(f"Enter value for {feature}", step=0.01)
    user_input.append(val)

# Predict button
if st.button("Predict"):
    # Convert to numpy array
    user_input = np.array(user_input).reshape(1, -1)

    # Scale the input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Map to label
    predicted_label = class_labels[predicted_class]

    # Display result
    st.success(f"🛡️ Predicted Situation: {predicted_label}")
