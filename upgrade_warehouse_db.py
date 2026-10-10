import sqlite3

connection = sqlite3.connect("warehouse.db")
cursor = connection.cursor()

# Drop the old basic table if it exists so we can upgrade the structure
cursor.execute("DROP TABLE IF EXISTS pallet_scans")

# Create an advanced table that mimics an industrial inventory log
cursor.execute("""
CREATE TABLE pallet_scans (
scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
image_name TEXT,
object_count INTEGER,
detected_sku TEXT,
tracking_barcode TEXT
)
""")

connection.commit()
connection.close()
print("🚀 SQL Table successfully upgraded with Barcode and SKU logging fields!")