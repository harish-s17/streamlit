import streamlit as st
import loginpage
import signup


query_params = st.query_params
page = query_params.get("page", "login")


if page == "login":
    loginpage.show()
elif page == "signup":
    signup.show()

