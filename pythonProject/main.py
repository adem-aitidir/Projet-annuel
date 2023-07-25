import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup
import xgboost as xgb
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Stock Forecast App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define theme colors
primary_color = "#007BFF"
secondary_color = "#6C757D"
background_color = "#F8F9FA"
text_color = "#333333"

# Apply theme styles
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {primary_color};
        }}
        .Widget.stSelectbox>div>div:nth-child(2) {{
            color: {text_color};
            background-color: {secondary_color};
        }}
        .cursor-pointer {{
            cursor: pointer;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOGL', 'AAPL', 'MSFT', 'META', 'AMZN', 'ETH', 'XRP', 'SOL')
selected_stock = st.selectbox('Sélectionne le dataset souhaité', stocks)

company_names = {
    'GOOGL': 'Google LLC',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'META': 'Meta Platforms, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'ETH': 'Ethereum',
    'XRP': 'Ripple XRP',
    'SLN': 'Solana'

}

st.markdown(f'<p style="font-size:25px;color:skyblue;">{company_names[selected_stock]}</p>', unsafe_allow_html=True)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Téléchargement des données ...')
data = load_data(selected_stock)
data_load_state.text('Données téléchargées... done!')

st.subheader('Dataset')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='red')))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Load the appropriate model based on the selected stock
if selected_stock == 'MSFT':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/MSFT_mdl.h5'
elif selected_stock == 'XRP':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/XRP_mdl.h5'
elif selected_stock == 'ETH':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/ETH_mdl.h5'
elif selected_stock == 'SOL':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/SOL_mdl.h5'
elif selected_stock == 'GOOGL':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/GOOGL_mdl.h5'
elif selected_stock == 'AAPL':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/AAPL_mdl.h5'
elif selected_stock == 'META':
    model_path = '/Users/benadem/Desktop/Projet annuel/pythonProject/venv/META_mdl.h5'
elif selected_stock == 'AMZN':
    model_path = '/Users/Desktop/Projet annuel/pythonProject/venv/AMZN_mdl.h5'
else:
    # Default to MSFT model if the selected stock is not recognized
    model_path = '/Users/benadem/PycharmProjects/pythonProject/venv/META_mdl.h5'

model_path = '/Users/benadem/PycharmProjects/pythonProject/venv/META_mdl.h5'

model = load_model(model_path)

# Prepare the data for prediction
window_size = 7  # Adjust this to match the window size used in your model
X = data['Close'].values[-window_size:][None, :, None]  # Get the last window_size number of closing prices

# Predict
predicted_close_price = model.predict(X)[0][0]

# Convert the prediction to a date for tomorrow
tomorrow_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

# Calculate the true close price for tomorrow
true_close_price_tomorrow = data['Close'].iloc[-window_size]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error([true_close_price_tomorrow], [predicted_close_price]))

st.subheader('Prédiction du stock')
st.write(f" la prédiction de demain ({tomorrow_date}) est de :  {predicted_close_price:.2f}")

st.subheader('(RMSE)')
st.write(f"RMSE : {rmse:.2f}")

# Plot the window data and prediction
def plot_window_and_prediction():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'].tail(window_size), y=data['Close'].tail(window_size), name="Window Data", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[tomorrow_date], y=[predicted_close_price], mode='markers', name="Prediction", marker=dict(color='red', size=10)))
    fig.layout.update(title_text='Affichage', xaxis_rangeslider_visible=False, xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

plot_window_and_prediction()