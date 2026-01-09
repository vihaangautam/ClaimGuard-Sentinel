import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_csv("processed_drought_data.csv")  # Update path if needed

# --- Preprocessing ---
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# One-hot encode Satellite and Location
encoder = OneHotEncoder(sparse_output=False)
cat_features = encoder.fit_transform(df[["Satellite", "Location"]])
cat_feature_names = encoder.get_feature_names_out(["Satellite", "Location"])
cat_df = pd.DataFrame(cat_features, columns=cat_feature_names)

# Combine with numerical features
features = pd.concat([
    cat_df.reset_index(drop=True),
    df[["Month", "Year", "Avg_rainfall"]].reset_index(drop=True)
], axis=1)

# Scale features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Targets
targets = df[["NDVI", "SMI"]].values

# --- Create Time Window Sequences ---
def create_sequences(X, y, time_window):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_window + 1):
        X_seq.append(X[i:i + time_window])
        y_seq.append(y[i + time_window - 1])  # Predict last step in window
    return np.array(X_seq), np.array(y_seq)

time_window = 3
X_seq, y_seq = create_sequences(features_scaled, targets, time_window)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- Build CNN Model ---
input_shape = (X_train.shape[1], X_train.shape[2])
input_layer = Input(shape=input_shape)

x = Conv1D(64, kernel_size=2, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)

ndvi_output = Dense(1, name='ndvi_output')(x)
smi_output = Dense(1, name='smi_output')(x)

model = Model(inputs=input_layer, outputs=[ndvi_output, smi_output])
model.compile(
    optimizer='adam',
    loss={'ndvi_output': 'mse', 'smi_output': 'mse'},
    metrics={'ndvi_output': ['mae'], 'smi_output': ['mae']}
)



# Make sure y_train is in the correct shape
y_train = y_train.squeeze()  # Shape becomes (samples, 2)

# Define early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, [y_train[:, 0], y_train[:, 1]],  # Each output separately
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --- Evaluate ---
eval_results = model.evaluate(X_test, [y_test[:, 0], y_test[:, 1]])
print(f"Test Loss & MAE: {eval_results}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Predict on test data ---
y_pred = model.predict(X_test)
ndvi_pred, smi_pred = y_pred[0].flatten(), y_pred[1].flatten()
ndvi_true, smi_true = y_test[:, 0], y_test[:, 1]

# --- NDVI Evaluation ---
ndvi_mae = mean_absolute_error(ndvi_true, ndvi_pred)
ndvi_rmse = np.sqrt(mean_squared_error(ndvi_true, ndvi_pred))
ndvi_r2 = r2_score(ndvi_true, ndvi_pred)

# --- SMI Evaluation ---
smi_mae = mean_absolute_error(smi_true, smi_pred)
smi_rmse = np.sqrt(mean_squared_error(smi_true, smi_pred))
smi_r2 = r2_score(smi_true, smi_pred)

# --- Print Results ---
print("\n--- Evaluation Metrics ---")
print(f"NDVI  - MAE: {ndvi_mae:.4f}, RMSE: {ndvi_rmse:.4f}, R²: {ndvi_r2:.4f}")
print(f"SMI   - MAE: {smi_mae:.4f}, RMSE: {smi_rmse:.4f}, R²: {smi_r2:.4f}")


# --- Plot History ---
plt.plot(history.history['ndvi_output_loss'], label='NDVI Loss')
plt.plot(history.history['smi_output_loss'], label='SMI Loss')
plt.plot(history.history['val_ndvi_output_loss'], label='Val NDVI Loss')
plt.plot(history.history['val_smi_output_loss'], label='Val SMI Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid()
plt.show()
# 