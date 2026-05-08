import os
import glob
import cv2
import numpy as np

INPUT_DIR = '/home/wayne/Documents/Progress/SCOUT'
OUTPUT_DIR = 'heatmap'

def get_heatmap(ori_img, mask_img):
    mask_img = cv2.normalize(mask_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_img = cv2.resize(mask_img, (ori_img.shape[1], ori_img.shape[0]))
    heatmap = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
    ori_img = ori_img.astype(np.uint8)
    overlay = cv2.addWeighted(ori_img, 0.7, heatmap, 0.3, 0)
    overlay = cv2.resize(
            overlay, 
            (overlay.shape[1] * 10, overlay.shape[0] * 10),
            interpolation=cv2.INTER_CUBIC
        )
    return overlay

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for base_path in base_image_paths:
        base_name = os.path.splitext(os.path.basename(base_path))[0]

        # Read RGB images
        ori_img = cv2.imread(base_path, cv2.IMREAD_COLOR)
        if ori_img is None:
            print(f"Failed to load {base_path}...")
            continue

        # Read all mask .png (ex: '112_00127_*.png')
        mask_pattern = os.path.join(INPUT_DIR, f"{base_name}_*.png")
        mask_paths = glob.glob(mask_pattern)

        if not mask_paths:
            print(f"Can't find any mask based on {base_name}...")
            continue

        # Process heatmap
        for mask_path in mask_paths:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"Failed to load {mask_path}...")
                continue

            output = get_heatmap(ori_img, mask_img)
            mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
            output_filename = f"{mask_basename}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, output)
            print(f"Saved successfully: {output_path}")

    print("All images processed！")