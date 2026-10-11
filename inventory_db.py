import sqlite3
import os
import random
from datetime import datetime

def init_inventory_database(db_path="warehouse.db"):
    """
    Creates an SQLite database file and defines tables for tracking
    detected warehouse pallets and individual inventory items.
    """
    print(f"Initializing warehouse asset tracking ledger at: '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create table for Pallet tracking logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pallets (
    pallet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_image TEXT NOT NULL,
    scan_timestamp TEXT NOT NULL,
    total_items_detected INTEGER NOT NULL
    )
    """)

    # 2. Create table for individual items tracked on pallets
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pallet_id INTEGER,
    sku_string TEXT UNIQUE NOT NULL,
    barcode_id TEXT NOT NULL,
    class_label TEXT NOT NULL,
    confidence_rating REAL NOT NULL,
    FOREIGN KEY (pallet_id) REFERENCES pallets (pallet_id)
    )
    """)

    conn.commit()
    conn.close()
    print("Database architectural tables established successfully.")

def log_detected_pallet_to_db(image_name, detected_objects, db_path="warehouse.db"):
    """
    Logs an inference session record and injects mock barcodes
    and unique enterprise SKUs into the ledger database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert parent pallet tracking row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO pallets (source_image, total_items_detected, scan_timestamp)
    VALUES (?, ?, ?)
    """, (image_name, len(detected_objects), timestamp))

    pallet_id = cursor.lastrowid

    # Insert associated item breakdowns with mock logistical metadata strings
    for obj in detected_objects:
        class_name = obj['class']
        conf = obj['conf']

        # Generate clean mock enterprise logistical values
        rand_id = random.randint(10000, 99999)
        mock_sku = f"SKU-{class_name[:3].upper()}-{rand_id}"
        mock_barcode = f"BC-{random.randint(10000000, 99999999)}"

        cursor.execute("""
        INSERT INTO inventory_items (pallet_id, sku_string, barcode_id, class_label, confidence_rating)
        VALUES (?, ?, ?, ?, ?)
        """, (pallet_id, mock_sku, mock_barcode, class_name, conf))

    conn.commit()
    conn.close()
    print(f"Successfully logged Pallet #{pallet_id} and {len(detected_objects)} tracking objects to database.")

if __name__ == "__main__":
    # Test initialization loop execution
    init_inventory_database()