import os
import numpy as np
import cv2

# Set the directories
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory
SEGMENTED_DIR = "D:\\SAMTEST\\segmented"  # Directory to save segmented images

# Ensure segmented directory exists
os.makedirs(SEGMENTED_DIR, exist_ok=True)

def load_image(image_path):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None
    if image_bgr.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
    return image_rgb

def apply_masks(image_rgb, masks):
    segmented_images = []
    for i, mask in enumerate(masks):
        segmented_image = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            segmented_image[..., j] = image_rgb[..., j] * mask
        segmented_image[..., 3] = (mask * 255).astype(np.uint8)  # 设置 alpha 通道
        segmented_images.append(segmented_image)
    return segmented_images

def load_and_apply_masks(image_file, mask_file):
    image_rgb = load_image(image_file)
    if image_rgb is None:
        return
    
    with np.load(mask_file) as data:
        masks = [data[key] for key in data]

    segmented_images = apply_masks(image_rgb, masks)

    for idx, segmented_image in enumerate(segmented_images):
        save_path = os.path.join(SEGMENTED_DIR, f"{os.path.splitext(os.path.basename(image_file))[0]}_segmented_{idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2BGRA))
        print(f"Saved segmented image to: {save_path}")

# Load and apply masks for each image
for image_file in os.listdir(IMAGE_DIR):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask_file = os.path.join(SAVE_DIR, f"{os.path.splitext(image_file)[0]}_mask.npz")
        if os.path.exists(mask_file):
            load_and_apply_masks(os.path.join(IMAGE_DIR, image_file), mask_file)
        else:
            print(f"Mask file not found for image: {image_file}")
