import streamlit as st

st.title("LOGIN")

username = st.text_input("Enter username", key="username")
password = st.text_input("Enter password", type="password", key="password")

if st.button("Login"):
    if username and password:
        st.success("Logged in successfully!")
    else:
        st.error("Please fill in all fields.")