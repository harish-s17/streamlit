import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from io import StringIO


# Title for the app
st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color:rgb(132, 255, 0); 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        border: 3px solid #FFD700;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        Stock Price Prediction App
    </h1>
    """,
    unsafe_allow_html=True,
)
# Upload CSV file

# Load the dataset
option = st.sidebar.selectbox(
    "Choose a page:",
    ["Tesla", "Reliance", "AAPL"]
)


with st.container():
    st.markdown(
    '''
    <style>
    .stApp{
    background-image: url("https://media.istockphoto.com/id/1465618017/photo/businessmen-investor-think-before-buying-stock-market-investment-using-smartphone-to-analyze.jpg?s=2048x2048&w=is&k=20&c=ocYlO-ILbQNIpV70O32Ja3P4kMLi9_Yj-78Xrf-Y6L8=");
    background-size: cover;
    filter: brightness(90%); /* Reduces brightness to 50% */
    }
    
    </style>
    ''', unsafe_allow_html=True)


def preprocess_data(df):
    if df is not None:
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ["int64", "float64"]:
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
    return df


 ########################################################################################################################################
        

if option == "Tesla":
    
    st.markdown(
    """
    <h1 style="
        font-weight: bold; 
        color: #FFD700; 
        text-shadow: 2px 2px 0px black, -2px -2px 0px black, 
                     -2px 2px 0px black, 2px -2px 0px black; 
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        display: inline-block;">
        TESLA Stock Price Prediction
    </h1>
    """,
    unsafe_allow_html=True,
)  
     
     
    data = pd.read_csv(r"D:\ML\warmupdata\TSLA.csv")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button('Home'):
            st.session_state.page = 'Home'
    with col2:
        if st.button('Data Preprocessing'):
            st.session_state.page = 'Data Preprocessing'
    with col3:
        if st.button('Model Building'):
            st.session_state.page = 'Model Building'
    with col4:
        if st.button('Visualization'):
            st.session_state.page = 'Visualization'
    with col5:
        if st.button('Prediction'):
            st.session_state.page = 'Prediction'


    col1, col2, col3 ,col4= st.columns(4)
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    regressor = LinearRegression()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})
    
    
    
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if st.session_state.page == "Home":
        st.markdown(f"<h2 style='color:#ff0078d7; font-weight:bold;'>Welcome to the Stock Price Prediction App</h2>", unsafe_allow_html=True)

        st.write("""
        Welcome to Tesla Stock Prediction Stay ahead in the market with predictions for Tesla (TSLA). 
            We use advanced algorithms to forecast future price movements based on historical data.      
        WE OFFER : 
        - Real-time Data: Track Tesla stock prices live.
        - Missing value handling
        - Accurate Predictions: Get forecasts based on historical trends.
        - Visual Insights: Interactive charts for better decision-making
    """)
      

    elif st.session_state.page == "Data Preprocessing":
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Overview:</h4>", unsafe_allow_html=True)
            st.write(data.head())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (Before Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            
            data = preprocess_data(data)

            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Missing Data (After Preprocessing):</h4>", unsafe_allow_html=True)
            st.write(data.isnull().sum())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Information:</h4>", unsafe_allow_html=True)
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Dataset Description:</h4>", unsafe_allow_html=True)
            st.write(data.describe())
    
    
    # Train-test split
    elif st.session_state.page == "Visualization":
        graph = data1.head(20)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>LINE PLOT</h4>", unsafe_allow_html=True)

        columns = st.multiselect("Select columns to visualize", data.columns.tolist(),
                                 default=["High", "Low", "Open", "Close"])
        if columns:
            st.line_chart(data[columns])
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>BAR GRAPH (Actual Vs Predicted)</h4>", unsafe_allow_html=True)

        st.bar_chart(graph)
    
    elif st.session_state.page == "Model Building":
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Coefficients </h4>", unsafe_allow_html=True)
        st.write(regressor.coef_)

        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Actual Vs Predicted </h4>", unsafe_allow_html=True)
        st.write(data1.head(20))
        mae = metrics.mean_absolute_error(y_test, predicted)
        mse = metrics.mean_squared_error(y_test, predicted)
        rmse = math.sqrt(mse)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Model Intercept: {regressor.intercept_}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Absolute Error: {mae}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Mean Squared Error: {mse}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Root Mean Squared Error: {rmse}</h4>", unsafe_allow_html=True)
    
    
    elif st.session_state.page == "Prediction":
    # New data prediction
        st.write("Predict New Data:")
        high = st.number_input("High Price:")
        low = st.number_input("Low Price:")
        open_price = st.number_input("Open Price:")
        volume = st.number_input("Volume:")

        if st.button("Predict"):
            new_data = np.array([[high, low, open_price, volume]])
            predicted_price = regressor.predict(new_data)
            st.markdown(f"<h4 style='color:#ff0078d7; font-weight:bold;'>Predicted Close Price  : {predicted_price[0]}</h4>", unsafe_allow_html=True)


            
###########################################################################################################################################
