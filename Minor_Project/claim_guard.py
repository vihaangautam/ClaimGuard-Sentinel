import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

class ClaimGuardEngine:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.scaler = MinMaxScaler()
        self.drought_threshold = 0.3  # NDVI threshold for drought (example)
        
    def load_resources(self):
        print("Initializing ClaimGuard Engine...")
        # Load Model
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Load Data
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%y')
            self.data = self.data.sort_values('Date')
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
        print("Engine Ready.")
        return True

    def verify_claim(self, location, claim_date_str):
        """
        Simulates the verification process:
        1. Checks if data exists for the location and date.
        2. Retrieves the actual NDVI from satellite data.
        3. Uses the model to analyze the trend (optional context).
        4. Returns a verification decision.
        """
        claim_date = pd.to_datetime(claim_date_str)
        
        # 1. Filter by Location (Assuming dataset has multiple, currently using 'Anantapur' as default in code)
        # Note: Your current dataset might need more specific Geohash logic later.
        geo_data = self.data[self.data['Location'] == location]
        
        if geo_data.empty:
            return {
                "status": "ERROR",
                "message": f"No satellite data found for region: {location}"
            }

        # 2. Find closest data point
        # In a real app, we'd fetch specific satellite imagery.
        # Here, we look up our historical database.
        record = geo_data[geo_data['Date'] == claim_date]
        
        if record.empty:
            # Find nearest date
            nearest_idx = (geo_data['Date'] - claim_date).abs().idxmin()
            record = geo_data.loc[[nearest_idx]]
            data_date = record['Date'].values[0]
            note = f"Exact date not found. Using nearest available data: {str(data_date)[:10]}"
        else:
            note = "Exact satellite match found."

        ndvi_val = record['NDVI'].values[0]
        smi_val = record['SMI'].values[0]
        rainfall_val = record['Avg_rainfall'].values[0]

        # 3. Decision Logic
        # "Is the NDVI below the drought threshold?"
        is_drought = ndvi_val < self.drought_threshold
        
        # Smart Logic: Verification Score
        # 0.0 (Fraud) -> 1.0 (Valid)
        # We can use SMI and Rainfall to boost confidence.
        confidence = 0.0
        if ndvi_val < 0.3: confidence += 0.5
        if smi_val < 0.3: confidence += 0.3
        if rainfall_val < 2.0: confidence += 0.2 # Example threshold

        decision = "APPROVED" if confidence > 0.6 else "FLAGGED_FOR_REVIEW"
        if ndvi_val > 0.5: decision = "REJECTED (Healthy Vegetation)"

        response = {
            "claim_id": f"CLM-{np.random.randint(10000,99999)}",
            "region": location,
            "claim_date": claim_date_str,
            "satellite_analysis": {
                "NDVI": float(f"{ndvi_val:.4f}"),
                "Soil_Moisture": float(f"{smi_val:.4f}"),
                "Rainfall_Index": float(f"{rainfall_val:.4f}")
            },
            "system_decision": decision,
            "verification_latency": "0.42s", # Simulation
            "confidence_score": f"{confidence * 100:.1f}%",
            "note": note
        }
        
        return response

if __name__ == "__main__":
    # Simulate the inputs
    engine = ClaimGuardEngine(
        model_path="model/finetuned_cnn_lstm_hybrid_entire_dataset.keras",
        data_path="processed_drought_data.csv"
    )
    
    if engine.load_resources():
        print("\n--- INCOMING CLAIM #1 ---")
        print("Farmer claims drought in Anantapur on 2023-05-15")
        result = engine.verify_claim("Anantapur", "2023-05-15")
        print(json.dumps(result, indent=4))

        print("\n--- INCOMING CLAIM #2 ---")
        print("Farmer claims drought in Anantapur on 2021-09-10")
        result = engine.verify_claim("Anantapur", "2021-09-10")
        print(json.dumps(result, indent=4))
