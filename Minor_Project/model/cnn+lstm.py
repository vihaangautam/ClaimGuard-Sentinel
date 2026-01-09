import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Load and Preprocess
df = pd.read_csv("processed_drought_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df = df.sort_values('Date')
data = df[['NDVI', 'SMI', 'Avg_rainfall']].values

print(f"Raw NDVI range: min={data[:, 0].min():.4f}, max={data[:, 0].max():.4f}")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 8  # Increased to 8 to capture more context
X, y = create_sequences(data_scaled, seq_length)
dates = df['Date'].iloc[seq_length:].values

split_date = pd.to_datetime('01/01/23', format='%m/%d/%y')
train_mask = dates < split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
dates_test = dates[~train_mask]

def augment_data(X, y, noise_factor=0.05):
    X_aug = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return np.concatenate([X, X_aug]), np.concatenate([y, y])

X_train_aug, y_train_aug = augment_data(X_train, y_train)
print(f"Augmented Train shape: {X_train_aug.shape}, Test shape: {X_test.shape}")

# Reshape for CNN (samples, timesteps, features)
X_train_aug = X_train_aug.reshape((X_train_aug.shape[0], X_train_aug.shape[1], X_train_aug.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build Fine-Tuned CNN-LSTM Hybrid Model
model = Sequential([
    Input(shape=(seq_length, 3)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),  # Increased filters and kernel
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),  # Additional CNN layer
    MaxPooling1D(pool_size=2),
    LSTM(150, return_sequences=True),  # First LSTM layer
    LSTM(100, return_sequences=False),  # Second LSTM layer
    BatchNormalization(),
    Dropout(0.4),  # Increased dropout
    Dense(64, activation='relu'),  # Increased dense layer size
    Dropout(0.3),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0005)  # Lower learning rate for stability
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Reduced patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train
history = model.fit(X_train_aug, y_train_aug, 
                    epochs=200, 
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
mae = np.mean(np.abs(y_test - y_pred.flatten()))
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Save Model
save_path = "model/finetuned_cnn_lstm_hybrid_entire_dataset.keras"
try:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved successfully at: {save_path}")
except Exception as e:
    print(f"Error saving model: {e}")

# Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='True NDVI', color='black')
plt.plot(dates_test, y_pred.flatten(), label='Predicted NDVI', linestyle='--', color='blue')
plt.title('Fine-Tuned CNN-LSTM Hybrid Predictions vs True NDVI (All Data 2023-2024)')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()