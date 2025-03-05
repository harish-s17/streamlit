import streamlit as st
import pickle

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def set_bg():
    bg_image_url="https://img.freepik.com/free-photo/cybersecurity-concept-collage-design_23-2151877155.jpg?t=st=1741011544~exp=1741015144~hmac=f56c0ee2bdc44abcb609ad710475dd853523b6542b32a2af2d49d346131ecc91&w=1380"
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

st.markdown("<h1 style='color:cyan; font-weight:bold;'>📩 Spam Message Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:orange; font-weight:bold;'>Enter a message below to check if it's spam or not.</p>", unsafe_allow_html=True)


user_input = st.text_input("Enter Message:", "")
 

if st.button("Predict"):
    if user_input:
        
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)

        
        if prediction[0] == 1:
            st.error("🚨 This message is *SPAM*!")
        else:
            st.success("✅ This message is *NOT SPAM*.")
    else:
        st.warning("⚠ Please enter a message to classify.")
