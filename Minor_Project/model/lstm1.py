import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
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

def create_sequences(data, seq_length, with_locs=False):
    X, y, locs = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # NDVI as target
        if with_locs and 'location' in df.columns:
            locs.append(df['location'].iloc[i + seq_length])
    if with_locs and 'location' in df.columns:
        return np.array(X), np.array(y), np.array(locs)
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

# Build and Train Simple LSTM
model = Sequential([
    Input(shape=(seq_length, 3)),
    LSTM(100, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(X_train_aug, y_train_aug, 
                    epochs=200, 
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Predict and Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred.flatten()))
mae = mean_absolute_error(y_test, y_pred.flatten())
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Save Model
save_path = "model/best_lstm_entire_dataset.keras"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"Model saved at: {save_path}")

# 1. Data for Predicted vs. True NDVI Scatter Plot
true_ndvi = y_test  # True NDVI values
pred_ndvi = y_pred.flatten()  # Predicted NDVI values
np.save('true_ndvi.npy', true_ndvi)
np.save('pred_ndvi.npy', pred_ndvi)

# 2. Data for Spatial RMSE Distribution Map
# Note: Assumes 'location' column exists; if not, aggregate or skip this part
if 'location' in df.columns:
    X, y, locs = create_sequences(data_scaled, seq_length, with_locs=True)
    loc_test = locs[~train_mask]
    districts = ['Ramanathapuram', 'Sivaganga', 'Dharmapuri', 'Koppal', 'Vijayapura', 
                 'Raichur', 'Chitradurga', 'Anantapur', 'Kadapa', 'Kurnool', 
                 'Mahbubnagar', 'Palakkad', 'Idukki']
    rmse_per_district = {}
    for district in districts:
        mask = loc_test == district
        if np.sum(mask) > 0:
            rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            rmse_per_district[district] = rmse
    district_rmse = pd.DataFrame(list(rmse_per_district.items()), columns=['District', 'RMSE'])
    district_rmse.to_csv('district_rmse.csv', index=False)
else:
    print("No 'location' column found; skipping district-wise RMSE.")

# 3. Data for Feature Importance Ablation
feature_configs = {
    'NDVI Only': [0],
    'SMI Only': [1],
    'Rainfall Only': [2],
    'All Features': [0, 1, 2]
}
ablation_results = {}
for config_name, indices in feature_configs.items():
    X_ablated = X[:, :, indices]
    X_train_abl, X_test_abl = X_ablated[train_mask], X_ablated[~train_mask]
    model_abl = Sequential([
        Input(shape=(seq_length, len(indices))),
        LSTM(100, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model_abl.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])
    model_abl.fit(X_train_abl, y_train, epochs=200, batch_size=64, 
                  validation_data=(X_test_abl, y_test), 
                  callbacks=[early_stopping, reduce_lr], verbose=0)
    y_pred_abl = model_abl.predict(X_test_abl)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_abl.flatten()))
    mae = mean_absolute_error(y_test, y_pred_abl.flatten())
    ablation_results[config_name] = {'RMSE': rmse, 'MAE': mae}
ablation_df = pd.DataFrame(ablation_results).T
ablation_df.to_csv('ablation_results.csv', index=False)

# Original Plots (kept for reference)
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
plt.title('Best LSTM Predictions vs True NDVI (2023-2024)')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()