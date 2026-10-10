import sys
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
fasterrcnn_resnet50_fpn,
FasterRCNN_ResNet50_FPN_Weights,
)

print("\n🤖 INITIALIZING PYTORCH AI LAYER...")

# 1. LOAD THE BRAIN: Pull the pre-trained neural network
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval() # Put the model in prediction mode

print("🧠 Neural Network loaded successfully!")

# 2. DATA INPUT: Load image via PIL for PyTorch processing
image_path = "sample_pallet.jpg"
img_pil = Image.open(image_path).convert("RGB")

# Transform image pixels into an optimized math Tensor
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img_pil)
img_tensor = img_tensor.unsqueeze(0) # Add required batch dimension

print(f"📦 Image converted to PyTorch Tensor format: {img_tensor.shape}")

# 3. AI INFERENCE: Run the model
print("🕵️‍♂️ Running object detection analysis...")
with torch.no_grad():
    predictions = model(img_tensor)

# 4. PARSE RESTRAINTS: Extract locations and confidence scores
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
labels = predictions[0]["labels"]

# Reload image in OpenCV to draw bounding box results
output_img = cv2.imread(image_path)
detections_count = 0

print("\n--- 📊 AI PREDICTION RESULTS ---")
for i in range(len(scores)):
# Only look at objects the AI is more than 50% sure about
    if scores[i] > 0.50:
        detections_count += 1
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = boxes[i].numpy().astype(int)

        print(
            f"Detected Object #{detections_count}: Confidence {scores[i]:.2f} -> Coordinates: [{x1}, {y1}, {x2}, {y2}]"
        )

        # Draw a bright green rectangle around the item
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(
f"Detected Object #{detections_count}: Confidence {scores[i]:.2f} -> Coordinates: [{x1}, {y1}, {x2}, {y2}]"
)

# Draw a bright green rectangle around the item
cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the final marked-up image
cv2.imwrite("ai_detected_output.jpg", output_img)
print("=========================================")
print(
f"🎉 SUCCESS! Found {detections_count} items. Result saved as 'ai_detected_output.jpg'"
)
print("=========================================\n")

import sqlite3

# 1. Mocking the Visual Information layer data (Option B)
# This simulates what pyzbar or OCR would extract from your box labels
mock_barcode = "TRACK_US_940011120206"
mock_sku = "SKU-PROD-BOX-04"

# 2. Connect to your database
db_connection = sqlite3.connect("warehouse.db")
db_cursor = db_connection.cursor()

# 3. Prepare the upgraded SQL query
sql_query = """
INSERT INTO pallet_scans (image_name, object_count, detected_sku, tracking_barcode)
VALUES (?, ?, ?, ?)
"""

# 4. Execute using your live PyTorch count combined with your mocked labels
db_cursor.execute(
sql_query, ("sample_pallet.jpg", detections_count, mock_sku, mock_barcode)
)

db_connection.commit()
db_connection.close()

print("\n=========================================")
print("💾 DATA LAYER RECORD DEPLOYED SUCCESSFUL!")
print(f" -> Logged SKU: {mock_sku}")
print(f" -> Logged Barcode: {mock_barcode}")
print("=========================================\n")