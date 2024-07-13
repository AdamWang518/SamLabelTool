import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set the directories
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

def get_color(index, total):
    colormaps = ['tab20', 'tab20b', 'tab20c']
    cmap = plt.get_cmap(colormaps[index % len(colormaps)])
    color = cmap((index // len(colormaps)) / (total // len(colormaps) + 1))
    return [int(c * 255) for c in color[:3]] + [128]

def adjust_color(color_rgb, increase_saturation=2.0, increase_value=1.5):
    color_hsv = cv2.cvtColor(np.uint8([[color_rgb[:3]]]), cv2.COLOR_RGB2HSV)
    color_hsv = color_hsv.astype(np.float32)
    color_hsv[..., 1] = np.clip(color_hsv[..., 1] * increase_saturation, 0, 255)
    color_hsv[..., 2] = np.clip(color_hsv[..., 2] * increase_value, 0, 255)
    color_rgb_adjusted = cv2.cvtColor(color_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return [int(c) for c in color_rgb_adjusted[0][0]] + [color_rgb[3]]

def load_image(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

def load_and_display_masks(image_file, mask_file):
    image_rgb, _ = load_image(image_file)
    if image_rgb is None:
        return
    
    with np.load(mask_file) as data:
        masks = [data[key] for key in data]

    overlay = np.zeros_like(image_rgb, dtype=np.uint8)
    for i, mask in enumerate(masks):
        color = get_color(i, len(masks))
        color = adjust_color(color)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        overlay = np.maximum(overlay, colored_mask)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0))
    plt.title('Image with Masks')
    plt.axis('off')
    plt.show()

# Load and display masks for each image
for image_file in os.listdir(IMAGE_DIR):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_file = os.path.join(SAVE_DIR, f"{os.path.splitext(image_file)[0]}_mask.npz")
        if os.path.exists(mask_file):
            load_and_display_masks(os.path.join(IMAGE_DIR, image_file), mask_file)
        else:
            print(f"Mask file not found for image: {image_file}")
