import streamlit as st

st.write('Hello Horror')
st.title('HI')
st.caption('Welcome') 
st.header('Thankyou')
st.subheader('Good bye')


name = st.text_input("Enter your name")
if st.button("Submit"):
    st.write(f"Hello,{name}")

option = st.selectbox("Choose a fruit:", ["Apple", "Banana", "Orange"])
st.write("You selected:", option)

age = st.slider("Select your age:", 0, 100, 25)
st.write("Your age is:", age)

