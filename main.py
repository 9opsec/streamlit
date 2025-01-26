import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# Function to fetch stock data using Yahoo Finance API
@st.cache
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to process data for LSTM input
def process_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x_data, y_data = [], []
    for i in range(look_back, len(scaled_data)):
        x_data.append(scaled_data[i-look_back:i, 0])
        y_data.append(scaled_data[i, 0])
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))  # reshape for LSTM
    return x_data, y_data, scaler

# Build the LSTM model
def build_lstm_model(input_shape, layers=1, units=50, batch_size=32, epochs=10):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=(layers > 1), input_shape=input_shape))
    for _ in range(layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Set up Streamlit interface
st.title('Stock Price Prediction')
st.sidebar.header("Adjustable Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
look_back = st.sidebar.slider("Look-back Period (Days)", 1, 100, 60)
epochs = st.sidebar.slider("Epochs", 1, 100, 10)
layers = st.sidebar.slider("LSTM Layers", 1, 5, 1)
units = st.sidebar.slider("LSTM Units per Layer", 10, 100, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

st.sidebar.text("Model configuration parameters")

# Fetch historical data
data = get_stock_data(ticker, start_date, end_date)

# Process the data for LSTM
x_data, y_data, scaler = process_data(data, look_back)

# Split data into training and testing sets
train_size = int(len(x_data) * 0.8)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Build and train the LSTM model
model = build_lstm_model(x_train.shape[1:], layers, units, batch_size, epochs)

# Predict future stock prices
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Create a DataFrame for historical and predicted prices
predicted_df = data.iloc[train_size+look_back:]
predicted_df['Predicted Close'] = predicted_prices

# Visualize data
st.subheader(f"Stock Price Prediction for {ticker}")
fig = go.Figure()

# Add Historical Data
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'))

# Add Predicted Data
fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['Predicted Close'], mode='lines', name='Predicted Price'))

fig.update_layout(title=f"Stock Price Prediction for {ticker}",
                  xaxis_title="Date",
                  yaxis_title="Stock Price",
                  template="plotly_dark")
st.plotly_chart(fig)

# Download option for CSV
st.subheader("Download Data")
download_data = pd.DataFrame({
    'Date': predicted_df.index,
    'Actual Close': data['Close'].iloc[train_size+look_back:].values,
    'Predicted Close': predicted_df['Predicted Close']
})
download_data_csv = download_data.to_csv(index=False)
st.download_button(label="Download Predictions as CSV", data=download_data_csv, file_name=f"{ticker}_predictions.csv", mime="text/csv")

# How it Works Section
st.markdown("""
    ### How it works:
    - Select the stock ticker and date range.
    - Configure the machine learning model parameters such as epochs, layers, and batch size.
    - The model will predict the stock prices for the upcoming week based on historical data.
    - You can download the predicted data as a CSV.
""")
