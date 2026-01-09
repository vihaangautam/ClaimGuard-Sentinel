from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your Keras model
model_path = "/Users/kranthikuamarchowdary/Desktop/Research/Minor/Image Processing/model/best_lstm_entire_dataset.keras"
model = load_model(model_path)
scaler = MinMaxScaler()

# Function to create sequences (same as your LSTM code)
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Simulate data for the date range (replace with real data in production)
def simulate_data(start_date, end_date, location):
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly data
    np.random.seed(42)  # For reproducibility
    ndvi = np.random.uniform(0.0, 0.8, len(date_range))  # NDVI range from your data
    smi = np.random.uniform(0.0, 1.0, len(date_range))   # SMI range (assumed 0-1)
    rainfall = np.random.uniform(0, 150, len(date_range)) # Rainfall in mm (0-150)

    # Adjust for drier districts
    dry_districts = ['Ramanathapuram', 'Koppal', 'Anantapur', 'Kurnool']
    if location in dry_districts:
        ndvi *= 0.8  # Reduce NDVI for drier areas
        rainfall *= 0.7  # Reduce rainfall for drier areas

    data = np.column_stack((ndvi, smi, rainfall))
    return data

@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return an empty response to suppress the favicon error

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        start_date = data['start_date']
        end_date = data['end_date']
        location = data['location']

        # Simulate data (replace with real data in production)
        data = simulate_data(start_date, end_date, location)

        # Preprocess
        data_scaled = scaler.fit_transform(data)
        seq_length = 6
        if len(data_scaled) < seq_length:
            return jsonify({"error": "Date range must span at least 6 months for prediction."}), 400
        X = create_sequences(data_scaled[-seq_length:], seq_length)

        # Predict NDVI
        pred_ndvi_scaled = model.predict(X)
        pred_ndvi = scaler.inverse_transform(
            np.concatenate((pred_ndvi_scaled, np.zeros((pred_ndvi_scaled.shape[0], 2))), axis=1)
        )[:, 0]

        # Determine drought
        is_drought = pred_ndvi[-1] < 0.3
        prediction = "Drought Likely" if is_drought else "No Drought Expected"
        confidence = (1 - 0.1429) * 100 if is_drought else 0.1429 * 100

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "ndvi": float(pred_ndvi[-1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)