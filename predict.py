import os
import cv2
from ultralytics import YOLO
import sqlite3
import random
from datetime import datetime

def log_inference_to_db(image_name, detected_objects, db_path="warehouse.db"):
    """
    Connects to the SQLite database and logs the parent pallet session
    along with individual barcode/SKU data for all detected items.
    """
    if not os.path.exists(db_path):
        # Fallback to initialize database if it doesn't exist yet
        print(f"[Warning] Database '{db_path}' not found. Initializing now...")
        from inventory_db import init_inventory_database
        init_inventory_database(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Log the parent pallet image scan session
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO pallets (source_image, scan_timestamp, total_items_detected)
    VALUES (?, ?, ?)
    """, (image_name, len(detected_objects), timestamp))

    pallet_id = cursor.lastrowid

    # 2. Log each individual detected item box under this pallet ID
    for obj in detected_objects:
        class_name = obj['class']
        conf = obj['conf']

        # Generate clean corporate mock serialization values
        rand_id = random.randint(10000, 99999)
        mock_sku = f"SKU-{class_name[:3].upper()}-{rand_id}"
        mock_barcode = f"BC-{random.randint(10000000, 99999999)}"

        cursor.execute("""
        INSERT INTO inventory_items (pallet_id, sku_string, barcode_id, class_label, confidence_rating)
        VALUES (?, ?, ?, ?, ?)
        """, (pallet_id, mock_sku, mock_barcode, class_name, conf))

    conn.commit()
    conn.close()
    print(f" Ledger Update: Logged Pallet #{pallet_id} tracking {len(detected_objects)} objects to '{db_path}'.")

def run_pallet_inference(image_path, model_weights="runs/detect/train3/weights/best.pt", output_dir="data/processed/ai_predictions"):
    """
    Loads YOLOv8 weights, runs prediction on a test pallet image,
    draws bounding boxes, and logs results directly to the database ledger.
    """
    # 1. Dynamically locate the latest training run weights if default path isn't found
    if not os.path.exists(model_weights):
        detect_dir = "runs/detect"
        if os.path.exists(detect_dir):
            runs = [f for f in os.listdir(detect_dir) if f.startswith("train")]
            if runs:
                runs.sort(key=lambda x: int(x.replace("train", "")) if x.replace("train", "").isdigit() else 0)
                latest_run = runs[-1]
                model_weights = os.path.join(detect_dir, latest_run, "weights", "best.pt")
                print(f"[Info] Specified weights not found. Auto-detected latest run: {model_weights}")

    if not os.path.exists(model_weights):
        print(f"[Error] Trained weights file not found at: '{model_weights}'")
        return

    # 2. Initialize YOLO model
    model = YOLO(model_weights)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path)

    print(f"Running inference on image asset: {base_name}...")
    results = model.predict(source=image_path, conf=0.25, save=False)

    img = cv2.imread(image_path)
    detected_objects_list = []

    # 3. Parse prediction results tensor matrix
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_id = int(box.cls.item())
            conf_score = box.conf.item()
            class_name = "cardboard_box" if cls_id == 0 else "wooden_pallet"

            # Append detections to a list structured for our database ingestion function
            detected_objects_list.append({
            'class': class_name,
            'conf': float(conf_score)
            })

            # Draw visual borders on image canvas
            border_color = (255, 255, 0) if cls_id == 0 else (255, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
            label_text = f"{class_name} ({conf_score:.2f})"
            cv2.putText(img, label_text, (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)

    # 4. Save visual overlay image asset
    output_path = os.path.join(output_dir, f"predicted_{base_name}")
    cv2.imwrite(output_path, img)
    print(f"Inference job complete. Isolated {len(detected_objects_list)} object targets.")
    print(f"Saved visual prediction confirmation to: '{output_path}'")

    # 5. Pipeline Handoff: Log structural findings straight to SQLite backend ledger
    log_inference_to_db(base_name, detected_objects_list)

if __name__ == "__main__":
    test_target = "data/images/pallet_01.jpg"

    if os.path.exists(test_target):
        run_pallet_inference(test_target)
    else:
        print(f"[Error] Target image file path '{test_target}' could not be verified.")