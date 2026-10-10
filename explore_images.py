import cv2

# 1. Read the image file using your computer vision layer
image_path = "sample_pallet.jpg"
img = cv2.imread(image_path)

# 2. Check if the file loaded correctly
if img is None:
    print(f"❌ Could not open '{image_path}'. Make sure it's in the same folder!")
else:
    print("\n=========================================")
print("📸 IMAGE LAYER CONVERTED TO DATA MATRIX 📸")
print("=========================================")

# Print out the layout structure of the raw vision data
# Output array maps to: (Height Pixels, Width Pixels, Color Channels)
print("Matrix Data Type:", type(img))
print("Dimensions (H, W, Channels):", img.shape)
print("Total Number of Pixels:", img.size)

# Peek inside the color values of a single pixel point
# Note: OpenCV reads colors in Blue, Green, Red (BGR) order!
print("BGR values at pixel grid coordinate (10,10):", img[10, 10])

#Crop a section of a image
cropped_section = img[50:250, 50:250]
cv2.imwrite("pallet_crop_test.jpg", cropped_section)
print("Cropped section saved as 'pallet_crop_test.jpg'")

print("=========================================\n")