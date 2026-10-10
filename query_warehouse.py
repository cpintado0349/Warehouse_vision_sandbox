import sqlite3

connection = sqlite3.connect("warehouse.db")
cursor = connection.cursor()

cursor.execute(
"SELECT scan_id, timestamp, object_count, detected_sku, tracking_barcode FROM pallet_scans"
)
records = cursor.fetchall()

print("\n--- 📊 INVENTORY CONTROL SQL LOG ---")
for row in records:
    print(f"ID: {row[0]} | Time: {row[1]} | Objects: {row[2]} | SKU: {row[3]} | Barcode: {row[4]}")
print("------------------------------------\n")

connection.close()