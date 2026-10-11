import cv2
import numpy as np
import os

def segment_pallet_items(image_path, output_dir="data/processed/segmented"):
    """Segments items on a pallet from the input image and saves the segmented images to the output directory."""
    #In here we load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at path: {image_path}")
        return

    #Now we create the directory for the output if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    #In here we Pre-process the image grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    base_name = os.path.basename(image_path)

    #In here we perform edge detection
    edges = cv2.Canny(blurred, 30, 100)

    #In we we close the gaps between the boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)   
    dilated = cv2.dilate(closed, kernel, iterations=1)

    #We find the counters which is the boundaries of the items 
    countrours, _ =  cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Now we visualize and isolate the segments
    output_img = img.copy
    segment_count = 0

    for i, contour in enumerate(countrours):
    #we filter out tiny noise contours 
        if cv2.contourArea(contour) < 500:
            continue
        segment_count += 1


    #We draw the outlines on the original images green color and the thickness is 2 
    cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)

    # In here we create an individual binary mask for this single object 
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    #The we save de individual item maske
    mask_filename = os.path.join(output_dir, f"mask_item_{segment_count}_{base_name}")
    cv2.imwrite(mask_filename, mask)

    #We save the final overlaid visualization
    vis_filename = os.path.join(output_dir, f"overlays_{base_name}")
    cv2.imwrite(vis_filename, output_img)

    print(f"processed {base_name}: Found {segment_count} potential item segments")
    print(f"Saved masks and overlays to {output_dir}")

    #test 
    if __name__ == "__main__":
        #test path targeting your raw dataset directory structure 
        test_image_path = "data/raw/pallet_01.jpg"
        segment_pallet_items(test_image_path)

    else:
        print(f"Please place a test image at '{test_image_path}' to run the baseline script.")