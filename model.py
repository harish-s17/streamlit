import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Function to apply background image
def set_bg():
    bg_image_url = "https://media.istockphoto.com/id/1317587887/photo/trading-charts-on-a-display.jpg?s=612x612&w=0&k=20&c=Yy2ULUp7jRXFRaOeClA45OUQr5Juv34cf7GCTRhLKO4="

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

# Apply background image
set_bg()

# Title Section
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style="
            font-weight: bold; 
            color: red; 
            text-shadow: 2px 2px black; 
            border: 3px solid orange;
            padding: 15px;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.7);
        ">
            Stock Price Prediction App
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # Adds extra space after title

# Sidebar: Choose stock dataset
option = st.sidebar.selectbox("Choose a stock:", ["Tesla", "Reliance"])

# Load dataset based on selection
dataset_path = {
    "Tesla": r"D:\ML\warmupdata\TSLA.csv",
    "Reliance": r"D:\ML\warmupdata\RELIANCE.csv"
}

data = pd.read_csv(dataset_path[option])

# Handle missing values BEFORE splitting the dataset
data.fillna(data.mean(numeric_only=True), inplace=True)

# Define features & target variable
X = data[['High', 'Low', 'Open', 'Volume']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

# Create DataFrame for Actual vs. Predicted values
data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})

# Navigation Buttons
st.write("")  # Space before buttons
st.markdown("<h3 style='text-align: center; color:lightgreen; font-weight:bold;'>Navigation</h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Home"):
    st.session_state.page = "Home"
if col2.button("Data Preprocessing"):
    st.session_state.page = "Data Preprocessing"
if col3.button("Model Building"):
    st.session_state.page = "Model Building"
if col4.button("Visualization"):
    st.session_state.page = "Visualization"
if col5.button("Prediction"):
    st.session_state.page = "Prediction"

# Default page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Page Layouts
if st.session_state.page == "Home":
    st.markdown(f"<h2 style='color:orange; font-weight:bold;'>Welcome to {option} Stock Prediction</h2>", unsafe_allow_html=True)
    st.write(f"""
    Welcome to {option} Stock Prediction! Stay ahead in the market with predictions based on historical data.
    
    **We Offer:**
    - 📈 Real-time Data Analysis
    - 🛠 Missing Value Handling
    - 🎯 Accurate Predictions
    - 📊 Visual Insights
    """)

elif st.session_state.page == "Data Preprocessing":
    st.markdown("<h4 style='color:orange; font-weight:bold;'>Dataset Overview:</h4>", unsafe_allow_html=True)
    st.write(data.head())

    st.markdown("<h4 style='color:orange; font-weight:bold;'>Missing Data (Before Preprocessing):</h4>", unsafe_allow_html=True)
    st.write(data.isnull().sum())

    st.markdown("<h4 style='color:orange; font-weight:bold;'>Missing Data (After Preprocessing):</h4>", unsafe_allow_html=True)
    st.write(data.isnull().sum())

elif st.session_state.page == "Visualization":
    st.markdown("<h4 style='color:orange; font-weight:bold;'>Line Plot of Stock Prices</h4>", unsafe_allow_html=True)
    columns = st.multiselect("Select columns to visualize", data.columns.tolist(), default=["High", "Low", "Open", "Close"])
    if columns:
        st.line_chart(data[columns])

    st.markdown("<h4 style='color:orange; font-weight:bold;'>Bar Graph (Actual vs Predicted)</h4>", unsafe_allow_html=True)
    st.bar_chart(data1.head(20))

elif st.session_state.page == "Model Building":
    st.markdown("<h4 style='color:orange; font-weight:bold;'>Model Coefficients</h4>", unsafe_allow_html=True)
    st.write(regressor.coef_)

    mae = metrics.mean_absolute_error(y_test, predicted)
    mse = metrics.mean_squared_error(y_test, predicted)
    rmse = math.sqrt(mse)

    st.markdown(f"<h4 style='color:orange; font-weight:bold;'>Mean Absolute Error: {mae:.2f}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:orange; font-weight:bold;'>Mean Squared Error: {mse:.2f}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:orange; font-weight:bold;'>Root Mean Squared Error: {rmse:.2f}</h4>", unsafe_allow_html=True)

elif st.session_state.page == "Prediction":
    st.markdown("<h4 style='color:orange; font-weight:bold;'>Predict Future Stock Prices</h4>", unsafe_allow_html=True)
    
    high = st.number_input("High Price:")
    low = st.number_input("Low Price:")
    open_price = st.number_input("Open Price:")
    volume = st.number_input("Volume:")

    if st.button("Predict"):
        new_data = np.array([[high, low, open_price, volume]])

        # Ensure the input is properly shaped
        if np.isnan(new_data).any():
            st.warning("Please enter valid values for all fields.")
        else:
            predicted_price = regressor.predict(new_data)
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Predicted Close Price: {predicted_price[0]:.2f}</h4>", unsafe_allow_html=True)
