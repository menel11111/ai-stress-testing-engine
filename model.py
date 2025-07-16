import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class RevenueForecaster:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, parse_dates=['date'])
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self):
        revenue_data = self.df[['revenue']].values
        scaled = self.scaler.fit_transform(revenue_data)
        
        X, y = [], []
        for i in range(12, len(scaled)):
            X.append(scaled[i-12:i])
            y.append(scaled[i])
        
        return np.array(X), np.array(y), scaled

    def train(self, epochs=50, batch_size=8):
        X, y, _ = self.prepare_data()
        self.model = Sequential([
            LSTM(64, input_shape=(X.shape[1], 1)),
            Dense(1)
        ])
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def forecast_next(self):
        _, _, scaled = self.prepare_data()
        last_seq = scaled[-12:].reshape(1, 12, 1)
        prediction_scaled = self.model.predict(last_seq)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        return int(prediction[0][0])
