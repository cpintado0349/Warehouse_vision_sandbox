import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
fasterrcnn_resnet50_fpn,
FasterRCNN_ResNet50_FPN_Weights
)

# 1. SETUP FOLDER DIRECTORIES
RAW_DIR = os.path.join("data", "raw_incoming")
PROCESSED_DIR = os.path.join("data", "processed_marked")
CROP_DIR = os.path.join("data", "cropped_items")

# 2. INITIALIZE DEEP LEARNING MODEL
print("\n🤖 LOADING AI INFRASTRUCTURE...")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
print("🧠 PyTorch Object Detector active!")

# 3. DISCOVER ALL IMAGES IN RAW FOLDER
# Looks for any file ending in .jpg, .jpeg, or .png
image_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"📦 Found {len(image_files)} image assets waiting to be processed in raw queue.")
print("==================================================")

# 4. START THE BATCH AUTOMATION LOOP
for img_name in image_files:
    print(f"\n🕵️‍♂️ Analyzing image: {img_name}")

# Generate complete relative file path locations
raw_path = os.path.join(RAW_DIR, img_name)
processed_path = os.path.join(PROCESSED_DIR, f"marked_{img_name}")

# Load via PIL for PyTorch model tensor conversion
img_pil = Image.open(raw_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img_pil).unsqueeze(0)

# Run AI inference to track object bounding maps
with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]

# Load via OpenCV to execute visualization layers
output_cv_img = cv2.imread(raw_path)
valid_counts = 0

for i in range(len(scores)):
    if scores[i] > 0.50: # Only look at highly confident matches
        valid_counts += 1
        x1, y1, x2, y2 = boxes[i].numpy().astype(int)

        cv2.rectangle(output_cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- ✂️ EXTRACT CROPPED OBJECT ARRAYS ---
# Python matrix slicing strips out just the individual carton item
cropped_asset = output_cv_img[y1:y2, x1:x2]
crop_name = f"crop_{valid_counts}_{img_name}"
crop_path = os.path.join(CROP_DIR, crop_name)

# Save individual crop snippet inside your dataset folder tree
cv2.imwrite(crop_path, cropped_asset)

# Draw green indicator box tracking coordinates on primary visual
cv2.rectangle(output_cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save master analyzed output directly inside processed path location
cv2.imwrite(processed_path, output_cv_img)
print(f" -> Complete! Found {valid_counts} assets. Logged crops to directory structure.")

print("\n==================================================")
print("🎉 BATCH RUN COMPLETED SUCCESSFULLY! Dataset processing complete.")
print("==================================================\n")