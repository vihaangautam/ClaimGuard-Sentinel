import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Load and Preprocess
df = pd.read_csv("/Users/kranthikuamarchowdary/Desktop/Research/Minor/Image Processing/processed_drought_data.csv")
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

seq_length = 6
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

# Build Model
model = Sequential([
    Input(shape=(6, 3)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train
history = model.fit(X_train_aug, y_train_aug, 
                    epochs=200, 
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
mae = np.mean(np.abs(y_test - y_pred.flatten()))
print(f"Test RMSE (normalized): {rmse:.4f}")
print(f"Test MAE (normalized): {mae:.4f}")

# Inverse transform
y_test_real = scaler.inverse_transform(np.column_stack([y_test, np.zeros_like(y_test), np.zeros_like(y_test)]))[:, 0]
y_pred_real = scaler.inverse_transform(np.column_stack([y_pred.flatten(), np.zeros_like(y_pred.flatten()), np.zeros_like(y_pred.flatten())]))[:, 0]

rmse_real = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
mae_real = np.mean(np.abs(y_test_real - y_pred_real))
print(f"Test RMSE (real NDVI): {rmse_real:.4f}")
print(f"Test MAE (real NDVI): {mae_real:.4f}")

# Save Model
save_path = "model/cnn_entire_dataset.h5"
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
plt.ylabel('MAE (normalized)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test_real, label='True NDVI', color='black')
plt.plot(dates_test, y_pred_real, label='Predicted NDVI', linestyle='--', color='blue')
plt.title('CNN Predictions vs True NDVI (All Data 2023-2024)')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()