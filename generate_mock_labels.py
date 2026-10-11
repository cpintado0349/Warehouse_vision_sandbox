import os

def create_mock_yolo_labels(img_dir="data/raw_incoming", output_dir="data/labels"):
    """
    Generates simulated ground-truth bounding box text annotations
    for the uniform cardboard boxes on the pallet stacks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Target all valid warehouse imagery files
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No assets found in '{img_dir}'. Run your batch script or add images first.")
        return

    # Mock bounding box layout representing rows of boxes on a pallet stack
    # Coordinates are normalized (0.0 to 1.0) relative to image width and height
    # Format: [class_id, x_center, y_center, width, height]
    mock_boxes = [
    [0, 0.45, 0.35, 0.15, 0.18], # Box 1 (Top Layer Left)
    [0, 0.55, 0.35, 0.15, 0.18], # Box 2 (Top Layer Right)
    [0, 0.38, 0.55, 0.16, 0.20], # Box 3 (Middle Layer Left)
    [0, 0.50, 0.55, 0.16, 0.20], # Box 4 (Middle Layer Center)
    [0, 0.62, 0.55, 0.16, 0.20], # Box 5 (Middle Layer Right)
    [0, 0.50, 0.75, 0.45, 0.15] # Wooden Pallet Base
    ]

    print(f"Generating mock annotations for {len(image_files)} imagery components...")

    for filename in image_files:
        # Swap image extension out for a text configuration extension (.txt)
        name_w_ext, _ = os.path.splitext(filename)
        label_filename = os.path.join(output_dir, f"{name_w_ext}.txt")

        with open(label_filename, "w") as f:
            for box in mock_boxes:
                # Format floating points safely for standard ML input specs
                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

    print(f"Mock labels cleanly exported to: '{output_dir}'")

if __name__ == "__main__":
    create_mock_yolo_labels()