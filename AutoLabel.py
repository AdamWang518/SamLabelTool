import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
# Set up paths and device
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
# mask_generator = SamAutomaticMaskGenerator(sam)
# mask_generator = SamAutomaticMaskGenerator(
#     sam, 
#     points_per_side=32,           # 生成点的数量，提高遮罩质量
#     pred_iou_thresh=0.90,         # 设置预测的 IoU 阈值，值越高筛选的遮罩越严格
#     stability_score_thresh=0.90,  # 设置稳定性得分阈值，值越高筛选的遮罩越严格
#     min_mask_region_area=100      # 设置最小遮罩区域，过滤掉小区域
# )
mask_generator = SamAutomaticMaskGenerator(
    sam,
    min_mask_region_area=100      # 设置最小遮罩区域，过滤掉小区域
)
# Set image directory and save directory
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load images from directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

def generate_initial_masks(image_rgb):
    sam_result = mask_generator.generate(image_rgb)
    initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
    mask_info = []

    for i, result in enumerate(sam_result):
        mask = result['segmentation']
        color = get_color(i, len(sam_result))
        color = adjust_color(color)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]
        colored_mask = np.dstack((colored_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_mask)
        mask_info.append(mask)

    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)
    return initial_mask, overlay, mask_info

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    image_rgb, image_bgr = load_image(image_path)
    
    if image_rgb is not None and image_bgr is not None:
        mask_path = os.path.join(SAVE_DIR, f"{os.path.splitext(image_file)[0]}_mask.npz")
        initial_mask, overlay, mask_info = generate_initial_masks(image_rgb)
        
        np.savez(mask_path, *mask_info)
        overlay_path = os.path.join(SAVE_DIR, f"{os.path.splitext(image_file)[0]}_segmented.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
