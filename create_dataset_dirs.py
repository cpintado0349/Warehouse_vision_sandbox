import os

# Define the dataset structure paths
base_dir = "data"
folders = [
                os.path.join(base_dir, "raw_incoming"), 
                os.path.join(base_dir, "processed_marked"), 
                os.path.join(base_dir, "cropped_items")]

print("Initializing warehouse image storage infrastructure...")

# Python loop to create each folder safely if it doesn't already exist
for folder_path in folders:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f" -> Created directory: {folder_path}")
    else:
        print(f" -> Directory already exists: {folder_path}")

print("Folder setup complete")