import os
import cv2
import numpy as np

def segment_single_image(image_path, processed_dir):
    """
    Advanced segmentation using Distance Transform and Watershed
    to separate tightly packed uniform boxes on a pallet.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f" [Error] Skipping. Could not read image: {image_path}")
        return 0

    base_name = os.path.basename(image_path)
    name_w_ext, ext = os.path.splitext(base_name)

    # 1. Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increased blur size to smooth out minor cardboard textures
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Otsu's Thresholding to separate the pallet stack from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Clean up noise with Morphological Opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Determine sure background area by dilating
    sure_bg = cv2.dilate(opening, kernel, iterations=5)

    # 5. Distance Transform to find the centers (peaks) of each individual box
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Normalize and threshold the distance map to isolate peak centers
    # Lowering the 0.4 multiplier creates larger peaks; raising it isolates sharper centers
    _, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. Find unknown regions (the boundaries where boxes touch)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Label the foreground object markers
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # Mark the unknown region boundaries with 0
    markers[unknown == 255] = 0

    # 8. Apply the Watershed Algorithm
    markers = cv2.watershed(img, markers)

    # 9. Extract and Save Segments based on Watershed markers
    output_img = img.copy()
    segment_count = 0

    mask_sub_dir = os.path.join(processed_dir, "masks")
    overlay_sub_dir = os.path.join(processed_dir, "overlays")
    os.makedirs(mask_sub_dir, exist_ok=True)
    os.makedirs(overlay_sub_dir, exist_ok=True)

    # Unique marker values represent distinct identified objects (excluding background/borders)
    unique_markers = np.unique(markers)

    for marker_id in unique_markers:
        # Skip background (1) and watershed boundaries (-1)
        if marker_id <= 1:
            continue

    # Create binary mask for this specific component marker
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers == marker_id] = 255

    # Find the contour of this individual segment mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

    segment_count += 1

    # Draw individual colored boundary contours on the overlay output
    cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)

    # Save the individual item pixel mask file
    mask_filename = os.path.join(mask_sub_dir, f"{name_w_ext}_mask_{segment_count}{ext}")
    cv2.imwrite(mask_filename, mask)

    # Draw structural watershed lines red on our overlay image
    output_img[markers == -1] = [0, 0, 255]

    vis_filename = os.path.join(overlay_sub_dir, f"{name_w_ext}_overlay{ext}")
    cv2.imwrite(vis_filename, output_img)

    return segment_count

def run_batch_pipeline(raw_dir="data/raw_incoming", processed_dir="data/processed"):
    """
    Iterates through the raw data directory, executing the image analytics
    pipeline on all detected warehouse imagery assets.
    """
    print(f"Starting batch visual intelligence processing...")
    print(f"Scanning directory: '{raw_dir}'\n")

    if not os.path.exists(raw_dir):
        print(f"[Error] Source directory '{raw_dir}' does not exist.")
        print("Please verify your repository dataset tree structure configuration.")
        return

    # Supported asset types
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No image files found in '{raw_dir}'. Add test images to begin processing.")
        return

    total_segmented = 0
    for filename in image_files:
        full_path = os.path.join(raw_dir, filename)
        print(f"Processing asset: {filename}")

        found_objects = segment_single_image(full_path, processed_dir)
        print(f" -> Extracted {found_objects} discrete item segment masks.")
        total_segmented += found_objects

    print(f"\nBatch job complete.")
    print(f"Total processed files: {len(image_files)}")
    print(f"Total masks created: {total_segmented}")

if __name__ == "__main__": 
    run_batch_pipeline()
