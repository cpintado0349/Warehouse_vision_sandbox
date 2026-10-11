import os
from ultralytics import YOLO

def train_warehouse_model():
    """
    Initializes a pre-trained YOLOv8 nano model and sets up
    the training loop parameters using our custom dataset yaml map.
    """
    yaml_config_path = "warehouse_dataset.yaml"

    if not os.path.exists(yaml_config_path):
        print(f"[Error] Configuration file '{yaml_config_path}' not found.")
        return

    print("Loading pre-trained YOLOv8 Nano model weights...")
    # Load a small, fast model architecture ideal for prototyping pipelines
    model = YOLO("yolov8n.pt")

    print("\nStarting model training verification loop...")
    # Train the model using our custom configurations
    # We set epochs=3 for a quick smoke test loop to verify everything functions
    results = model.train(
    data=yaml_config_path,
    epochs=3,
    imgsz=512,
    batch=2,
    workers=0 # Set to 0 to prevent multi-threading overhead errors on Windows/macOS
    )

    print("\nTraining verification complete.")
    print("Model checkpoints saved under the 'runs/detect/train/' folder.")

if __name__ == "__main__":
    train_warehouse_model()