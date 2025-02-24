import streamlit as st

def show():
    st.title("CREATE ACCOUNT")

    username = st.text_input("Enter username", key="usernameinput")
    password = st.text_input("Enter password", type="password", key="passwordinput")
    confirmpassword = st.text_input("Confirm password", type="password", key="confirmpasswordinput")

    if st.button("Create Account"):
        if username and password and confirmpassword:
            if password == confirmpassword:
                st.success("Account created successfully!")
            else:
                st.error("Passwords do not match. Please try again.")
        else:
            st.error("Please fill in all fields.")

    
    if st.button("Back to Login"):
        st.query_params["page"] = "login"
        st.rerun() 
