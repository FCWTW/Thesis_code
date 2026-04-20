import os
import cv2
import numpy as np

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
    mask_img = cv2.imread('00094.png', cv2.IMREAD_GRAYSCALE)
    ori_img = cv2.imread('00094.jpg', cv2.IMREAD_COLOR)
    if ori_img is None:
        print("Failed to read image")
    if mask_img is None:
        print("Failed to read mask ")
    output = get_heatmap(ori_img, mask_img)
    cv2.imwrite('result.png', output)