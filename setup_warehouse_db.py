import sqlite3

# 1. Connect to a database file (it creates 'warehouse.db' if it doesn't exist)
connection = sqlite3.connect("warehouse.db")
cursor = connection.cursor()

# 2. Create a structured SQL table for your pallet scans
cursor.execute("""
CREATE TABLE IF NOT EXISTS pallet_scans (
scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
image_name TEXT,
object_count INTEGER,
confidence_score REAL
)
""")

# 3. Save changes and close the connection
connection.commit()
connection.close()

print("🗄️ SQL Database and 'pallet_scans' table created successfully!")