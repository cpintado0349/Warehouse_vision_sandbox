import sqlite3
import os

def query_warehouse_ledger(db_path="warehouse.db"):
    """
    Connects to the warehouse SQLite database and outputs a formatted
    summary of tracked pallets and associated individual inventory items.
    """
    if not os.path.exists(db_path):
        print(f"[Error] Database ledger file '{db_path}' could not be located.")
        print("Please run your 'predict.py' or 'inventory_db.py' scripts first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 70)
    print(f" WAREHOUSE ENTERPRISE DATABASE LEDGER SUMMARY: {db_path}")
    print("=" * 70)

    # 1. Fetch and print parent pallet scan sessions
    cursor.execute("SELECT * FROM pallets")
    pallet_rows = cursor.fetchall()

    if not pallet_rows:
        print("\n[Notice] No pallet sessions have been logged to the database yet.")
    else:
        print(f"\n[PALLET SESSIONS TABLE - Total Rows: {len(pallet_rows)}]")
        print(f"{'ID':<5} | {'Source Image Asset':<22} | {'Scan Timestamp':<20} | {'Detections Count':<10}")
        print("-" * 70)
        for row in pallet_rows:
            print(f"{row[0]:<5} | {row[1]:<22} | {row[2]:<20} | {row[3]:<10}")

    # 2. Fetch and print associated logged individual items
    cursor.execute("SELECT * FROM inventory_items")
    item_rows = cursor.fetchall()

    print("\n" + "=" * 70)
    if not item_rows:
        print("\n[Notice] No individual items or barcode tokens have been logged yet.")
    else:
        print(f"\n[INVENTORY ITEMS SUMMARY - Total Rows: {len(item_rows)}]")
        print(f"{'Item ID':<8} | {'Pallet ID':<9} | {'Enterprise SKU':<18} | {'Barcode String':<14} | {'Class Label':<14}")
        print("-" * 70)
        # Limit printout to the first 15 entries to keep the terminal output readable
        for row in item_rows[:15]:
            print(f"{row[0]:<8} | {row[1]:<9} | {row[2]:<18} | {row[3]:<14} | {row[4]:<14}")

        if len(item_rows) > 15:
            print(f"... and {len(item_rows) - 15} more inventory ledger items.")

    conn.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    query_warehouse_ledger()