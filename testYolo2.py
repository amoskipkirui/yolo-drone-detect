from ultralytics import YOLO
from datetime import datetime
import os

# File names
MODEL_FILE = "yolov8x.pt"
IMAGE_FILE = "image.jpg"

# Load YOLO model
model = YOLO(MODEL_FILE)

# Run inference
results = model(IMAGE_FILE)

# Create timestamp for output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"result_{timestamp}.jpg"

# Save results
for result in results:
    result.save(filename=output_file)

print(f"Inference complete! Result saved as {output_file}")
