import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class WarehousePalletDataset(Dataset):
    """
    Custom PyTorch Dataset for loading warehouse pallet images
    and pairing them with their structural bounding box text labels.
    """
    def __init__(self, img_dir="data/raw_incoming", label_dir="data/labels", transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Gather all valid pallet images
        valid_extensions = (".jpg", ".jpeg", ".png")
        self.image_filenames = [
        f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1. Load the raw image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

        # 2. Resize image to standard network input dimensions
        image_resized = cv2.resize(image, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # 3. Locate and load the matching bounding box label file
        name_w_ext, _ = os.path.splitext(img_name)
        label_name = f"{name_w_ext}.txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        # Parse the float values: class_id, x_center, y_center, width, height
                        parts = [float(x) for x in line.split()]
                        boxes.append(parts)

        # Convert box list to a PyTorch Tensor
        if len(boxes) > 0:
            labels_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            # Handle edge case where an image has no annotations
            labels_tensor = torch.empty((0, 5), dtype=torch.float32)

        return image_tensor, labels_tensor, img_name

if __name__ == "__main__":
    # Test execution block to verify the multi-input loading pipeline
    dataset = WarehousePalletDataset()
    print(f"Dataset successfully initialized. Found {len(dataset)} items.")

    if len(dataset) > 0:
        sample_img, sample_labels, name = dataset[0]
        print(f"\n--- Loaded Specimen: {name} ---")
        print(f"Image Tensor Shape: {sample_img.shape}")
        print(f"Labels Tensor Shape: {sample_labels.shape} (Objects Found: {sample_labels.size(0)})")
        print("Sample Label Vector (Class, X_Center, Y_Center, Width, Height):")
        print(sample_labels[0])