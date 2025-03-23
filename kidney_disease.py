import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("kidney_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("kindney_graph.pkl", "rb") as f:
    graph_data = pickle.load(f)


def set_bg():
    bg_image_url = "https://img.freepik.com/free-vector/hexagonal-grid-background_52683-327.jpg?t=st=1742706448~exp=1742710048~hmac=cebf8b6d48d98c0153176467c434dd140e3b1bd9c1bfb3b990a7d131a7b8a9c8&w=1380"
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


st.markdown("<h1 style='color:cyan; font-weight:bold;'> KIDNEY DISEASE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:orange; font-weight:bold;'>Enter parameters to check.</p>", unsafe_allow_html=True)

st.title("Enter 10 Parameters")


feature_names = ['Age', 'Blood Pressure (bp)', 'Specific Gravity (sg)', 'Albumin (al)', 'Sugar (su)', 'Blood Glucose Random (bgr)', 'Blood Urea (bu)', 'Serum Creatinine (sc)', 'Hemoglobin (hemo)', 'Packed Cell Volume (pcv)']

user_inputs = []
for i, feature in enumerate(feature_names):
    value = st.text_input(f"Enter {feature}", key=f"input_{i}")
    user_inputs.append(value)


if st.button("Predict"):
    try:
        
        numeric_inputs = np.array(user_inputs, dtype=float).reshape(1, -1)
        prediction = model.predict(numeric_inputs)[0]
       
        if prediction == 1:
            st.error("🚨 You have a chronic kidney disease!")
        else:
            st.success("✅ NO disease predicted.")

    except ValueError:
        st.warning("⚠ Please enter valid numerical inputs for all fields.")


st.title("Actual vs. Predicted Results")
fig, ax = plt.subplots()
graph_data.plot(kind='bar', ax=ax)
st.pyplot(fig)


