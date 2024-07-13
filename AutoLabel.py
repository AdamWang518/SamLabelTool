import os
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Set up paths and device
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Set image directory and save directory
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load images from directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_color(index, total):
    unique_colors = plt.get_cmap('hsv')
    color = unique_colors(index / total)
    return [int(c * 255) for c in color[:3]] + [128]  # Convert to 8-bit color and set alpha to 128 for semi-transparency

def load_image(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

def generate_and_save_masks(image_rgb, image_path):
    sam_result = mask_generator.generate(image_rgb)
    initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)

    mask_info = []
    for i, result in enumerate(sam_result):
        mask = result['segmentation']
        color = get_color(i, len(sam_result))
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]
        colored_mask = np.dstack((colored_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_mask)
        mask_info.append(mask)

    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

    save_name = os.path.splitext(os.path.basename(image_path))[0]
    np.savez(os.path.join(SAVE_DIR, f"{save_name}_mask.npz"), *mask_info)
    cv2.imwrite(os.path.join(SAVE_DIR, f"{save_name}_segmented.png"), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))

# Iterate through images and generate masks
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(IMAGE_DIR, image_file)
    image_rgb, image_bgr = load_image(image_path)
    if image_rgb is not None:
        print(f"Processing {idx + 1}/{len(image_files)}: {image_file}")
        generate_and_save_masks(image_rgb, image_path)

print("All images processed.")
