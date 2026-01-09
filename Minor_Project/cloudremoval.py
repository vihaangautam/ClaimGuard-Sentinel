import cv2
import numpy as np
import os

# Define directories
input_dir = "drought_dataset"
output_dir = "cloud_removed_data"
os.makedirs(output_dir, exist_ok=True)

# Function to remove clouds
def remove_clouds(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define cloud mask (tweak the values if necessary)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply inpainting
    img_cloud_free = cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    # Save the cleaned image
    cv2.imwrite(output_path, img_cloud_free)
    print(f"Processed: {output_path}")

# Process all images
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)
            remove_clouds(input_path, output_path)

print("Cloud removal complete!")
