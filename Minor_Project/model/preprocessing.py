import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("processed_drought_data.csv")

# Filter for Anantapur and aggregate by date (average rainfall across satellites)
df_anantapur = df[df['Location'] == 'Anantapur'].groupby('Date').agg({
    'NDVI': 'mean',
    'SMI': 'mean',
    'Avg_rainfall': 'mean'
}).reset_index()

# Convert Date to datetime and sort
df_anantapur['Date'] = pd.to_datetime(df_anantapur['Date'], format='%m/%d/%y')
df_anantapur = df_anantapur.sort_values('Date')

# Extract features
data = df_anantapur[['NDVI', 'SMI', 'Avg_rainfall']].values

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (6 months input, 1 month output)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict NDVI
    return np.array(X), np.array(y)

seq_length = 6
X, y = create_sequences(data_scaled, seq_length)

# Split into train (2019-2022) and test (2023-2024)
split_date = pd.to_datetime('01/01/23', format='%m/%d/%y')
split_idx = df_anantapur[df_anantapur['Date'] < split_date].shape[0] - seq_length
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for CNN and Hybrid (add channel dimension)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")