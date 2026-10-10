import cv2
from pyzbar.pyzbar import decode

# 1. Load your image file via OpenCV
image_path = "sample_pallet.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"❌ Could not open '{image_path}'.")
else:
    print("\n=========================================")
print("🏷️ RUNNING VISUAL BARCODE EXTRACTION 🏷️")
print("=========================================")

# 2. Decode any barcodes found anywhere in the image pixels
detected_barcodes = decode(img)

if not detected_barcodes:
    print("🔍 No visible barcodes detected in this image sample.")
    print(" -> Tip: Try using an image that has a clear UPC or tracking label!")
else:
    for barcode in detected_barcodes:
        # Extract raw string data from the barcode bytes
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

print(f"✅ Found Barcode!")
print(f" -> Data Encoded: {barcode_data}")
print(f" -> Label Type: {barcode_type}")

# Extract location coordinates of the barcode to pass to your pipeline
(x, y, w, h) = barcode.rect
print(f" -> Pixel Location: X={x}, Y={y}, Width={w}, Height={h}")

# Use OpenCV to draw a bounding box specifically around the barcode label
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

# Save your barcode-mapped result
cv2.imwrite("barcode_detected_output.jpg", img)
print("\n🎉 Analysis complete! Marked label saved as 'barcode_detected_output.jpg'.")
print("=========================================\n")