import streamlit as st

def show():

    st.title("LOGIN")

    username = st.text_input("Enter username", key="username")
    password = st.text_input("Enter password", type="password", key="password")

    col1, col2 = st.columns([1, 1])  

    with col1:
        if st.button("Login"):
            if username and password:
                st.success("Logged in successfully!")
            else:
                st.error("Please fill in all fields.")

    with col2:
        if st.button("Signup"):
            st.query_params["page"] = "signup"  
            st.rerun()  
