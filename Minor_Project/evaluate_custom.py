import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv("processed_drought_data.csv")
except FileNotFoundError:
    print("Error: processed_drought_data.csv not found.")
    exit()

# 2. Preprocessing (Must match training exactly)
print("Preprocessing data...")
# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df = df.sort_values('Date')
data = df[['NDVI', 'SMI', 'Avg_rainfall']].values

# Store min/max for inverse scaling later
# We need to scale exactly as the model was trained. 
# Usually, we fit scaler on the whole dataset or just training set. 
# The training script fit on the WHOLE dataset before splitting.
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define Sequence Length (Must match model training)
seq_length = 8 

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Predict NDVI (index 0) at step i + seq_length
        y.append(data[i + seq_length, 0]) 
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, seq_length)
dates = df['Date'].iloc[seq_length:].values

# Split Data (Matches training logic)
split_date = pd.to_datetime('01/01/23', format='%m/%d/%y')
train_mask = dates < split_date

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
dates_test = dates[~train_mask]

# Reshape for CNN/LSTM Input (samples, timesteps, features)
# The training script used:
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
# Which is (samples, 8, 3). This is default from create_sequences, 
# but let's be explicit if dimensionality was added.
# In cnn+lstm.py: X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
# This effectively does nothing if it's already 3D, but good for safety.

print(f"Test Set Shape: {X_test.shape}")

# 3. Load Model
model_path = "model/finetuned_cnn_lstm_hybrid_entire_dataset.keras"
print(f"Loading model from {model_path}...")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 4. Predict
print("Predicting...")
y_pred_scaled = model.predict(X_test)

# 5. Inverse Transform to get Actual Values
# We need to inverse transform both y_test (shape N,) and y_pred (shape N, 1)
# The scaler was fit on 3 columns [NDVI, SMI, Rainfall]. 
# We only predicted NDVI (col 0).
# To inverse transform, we need a dummy array of shape (N, 3)

def inverse_transform_ndvi(y_scaled, scaler):
    # Create a dummy array with 3 columns, fill col 0 with predictions
    dummy = np.zeros((len(y_scaled), 3))
    dummy[:, 0] = y_scaled.flatten()
    # Inverse transform
    dummy_original = scaler.inverse_transform(dummy)
    # Return only the first column
    return dummy_original[:, 0]

y_test_original = inverse_transform_ndvi(y_test, scaler)
y_pred_original = inverse_transform_ndvi(y_pred_scaled, scaler)

# 6. Calculate Accuracy Metrics
rmse_scaled = np.sqrt(np.mean((y_test - y_pred_scaled.flatten()) ** 2))
mae_scaled = np.mean(np.abs(y_test - y_pred_scaled.flatten()))

rmse_original = np.sqrt(np.mean((y_test_original - y_pred_original) ** 2))
mae_original = np.mean(np.abs(y_test_original - y_pred_original))
mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-6))) * 100

with open("accuracy_report.txt", "w") as f:
    f.write("="*30 + "\n")
    f.write("MODEL ACCURACY REPORT\n")
    f.write("="*30 + "\n")
    f.write(f"Metrics on SCALED Data (0-1 range):\n")
    f.write(f"  RMSE: {rmse_scaled:.4f}\n")
    f.write(f"  MAE:  {mae_scaled:.4f}\n")
    f.write("-" * 30 + "\n")
    f.write(f"Metrics on ORIGINAL Data (NDVI range):\n")
    f.write(f"  RMSE: {rmse_original:.4f}\n")
    f.write(f"  MAE:  {mae_original:.4f}\n")
    f.write(f"  MAPE: {mape:.2f}%\n")
    f.write("-" * 30 + "\n")
    f.write(f"Interpretation:\n")
    f.write(f"- On average, the prediction deviates by {mae_original:.4f} NDVI units.\n")
    f.write(f"- Since NDVI ranges from -1 to 1 (usually 0 to 1 for vegetation),\n")
    f.write(f"  an error of {mae_original:.4f} is considered {'Low' if mae_original < 0.1 else 'Moderate' if mae_original < 0.2 else 'High'}.\n")
    f.write("="*30 + "\n")

print("Report saved to accuracy_report.txt")
