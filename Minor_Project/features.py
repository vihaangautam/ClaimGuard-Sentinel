import os
import cv2
import numpy as np
import pandas as pd
import re

# Define directories for NDVI and SMI images
ndvi_dir = "ndvi_data"  # Change to your NDVI images folder
smi_dir = "smi_data"    # Change to your SMI images folder

# Initialize a list to store extracted features
data = []

def extract_features(image_path):
    """Extract statistical features from an image (mean pixel value)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    if image is None:
        print(f"⚠️ Could not read {image_path}")
        return None
    return np.mean(image)  # Return mean pixel value as a feature

def parse_filename(filename):
    """Extract Satellite, Date, and Location from filenames like 'MODIS2021-01-01Mahbubnagar_ndvi.png'"""
    match = re.match(r"([A-Za-z0-9-]+)(\d{4}-\d{2}-\d{2})([A-Za-z-]+)_ndvi\.png", filename)
    if match:
        satellite, date, location = match.groups()
        return satellite, date, location
    return None, None, None

# Process NDVI images
for img_file in os.listdir(ndvi_dir):
    if img_file.endswith(".png"):
        ndvi_path = os.path.join(ndvi_dir, img_file)

        # Extract metadata from filename
        satellite, date, location = parse_filename(img_file)
        if not satellite or not date or not location:
            print(f"⚠️ Filename format incorrect: {img_file}")
            continue

        # Extract NDVI feature
        ndvi_value = extract_features(ndvi_path)

        # Find corresponding SMI image
        smi_path = os.path.join(smi_dir, img_file.replace("_ndvi.png", "_smi.png"))
        smi_value = extract_features(smi_path) if os.path.exists(smi_path) else None

        # Append to dataset
        data.append([satellite, date, location, ndvi_value, smi_value])

# Convert data to DataFrame
df = pd.DataFrame(data, columns=["Satellite", "Date", "Location", "NDVI", "SMI"])

# Save as CSV
df.to_csv("drought_features.csv", index=False)

print("✅ Feature extraction completed! Data saved to drought_features.csv")
