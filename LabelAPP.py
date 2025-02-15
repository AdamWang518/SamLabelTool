import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Set up paths and device
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
# mask_generator = SamAutomaticMaskGenerator(
#     sam, 
#     points_per_side=32,           # 生成点的数量，提高遮罩质量
#     pred_iou_thresh=0.90,         # 设置预测的 IoU 阈值，值越高筛选的遮罩越严格
#     stability_score_thresh=0.90,  # 设置稳定性得分阈值，值越高筛选的遮罩越严格
#     min_mask_region_area=100      # 设置最小遮罩区域，过滤掉小区域
# )
mask_predictor = SamPredictor(sam)


# Set image directory and save directory
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load images from directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
current_image_idx = 0

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
    print(f"Loading image from: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

def display_image(image_rgb, image_bgr, overlay, initialize=False):
    ax[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Source Image')
    if initialize:
        ax[1].imshow(overlay)
    else:
        ax[1].images[0].set_data(overlay)
    ax[1].set_title('Segmented Image')
    plt.tight_layout()
    fig.canvas.draw()

def generate_initial_masks(image_rgb):
    global initial_mask, overlay, mask_info
    print("Generating initial masks")
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
    print("Initial masks generated")

def load_existing_mask(mask_path, image_rgb):
    global initial_mask, overlay, mask_info
    print(f"Loading mask from: {mask_path}")
    with np.load(mask_path) as data:
        mask_info = [data[key] for key in data]
    initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
    
    for i, mask in enumerate(mask_info):
        color = get_color(i, len(mask_info))
        color = adjust_color(color)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]
        colored_mask = np.dstack((colored_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_mask)
    
    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)
    print("Overlay created")

def update_overlay():
    global overlay, initial_mask, image_rgb
    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)
    if ax[1].images:
        ax[1].images[0].set_data(overlay)
    else:
        ax[1].imshow(overlay)
    while ax[1].lines:
        ax[1].lines[0].remove()
    fig.canvas.draw()

def clear_all_data():
    global initial_mask, overlay, mask_info, input_points, input_labels, blue_points, blue_labels
    initial_mask = np.zeros_like(initial_mask)
    overlay = np.zeros_like(overlay)
    mask_info = []
    input_points = []
    input_labels = []
    blue_points = []
    blue_labels = []
    ax[1].clear()

def load_next_image(event):
    global current_image_idx, image_rgb, image_bgr, mask_info, initial_mask, overlay
    clear_all_data()  # 清空所有相关变量
    current_image_idx = (current_image_idx + 1) % len(image_files)
    image_rgb, image_bgr = load_image(os.path.join(IMAGE_DIR, image_files[current_image_idx]))
    if image_rgb is not None and image_bgr is not None:
        mask_path = os.path.join(SAVE_DIR, f"{os.path.splitext(image_files[current_image_idx])[0]}_mask.npz")
        if os.path.exists(mask_path):
            load_existing_mask(mask_path, image_rgb)
        else:
            generate_initial_masks(image_rgb)
        display_image(image_rgb, image_bgr, overlay, initialize=True)

def load_previous_image(event):
    global current_image_idx, image_rgb, image_bgr, mask_info, initial_mask, overlay
    clear_all_data()  # 清空所有相关变量
    current_image_idx = (current_image_idx - 1) % len(image_files)
    image_rgb, image_bgr = load_image(os.path.join(IMAGE_DIR, image_files[current_image_idx]))
    if image_rgb is not None and image_bgr is not None:
        mask_path = os.path.join(SAVE_DIR, f"{os.path.splitext(image_files[current_image_idx])[0]}_mask.npz")
        if os.path.exists(mask_path):
            load_existing_mask(mask_path, image_rgb)
        else:
            generate_initial_masks(image_rgb)
        display_image(image_rgb, image_bgr, overlay, initialize=True)

def refine_mask(event):
    global input_points, input_labels, initial_mask, overlay, mask_info, image_rgb
    if input_points:
        print("Refining mask")
        mask_predictor.set_image(image_rgb)
        prediction = mask_predictor.predict(point_coords=np.array(input_points), point_labels=np.array(input_labels))

        if len(prediction) == 3:
            masks, scores, logits = prediction
        else:
            masks, scores = prediction

        final_mask = masks[0]

        # Generate a new color for the refined mask
        manual_mask_color = get_color(len(mask_info), len(mask_info) + 1)
        manual_mask_color = adjust_color(manual_mask_color)
        colored_manual_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_manual_mask[final_mask > 0.5, j] = manual_mask_color[j]
        alpha_channel = (final_mask > 0.5).astype(np.uint8) * manual_mask_color[3]
        colored_manual_mask = np.dstack((colored_manual_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_manual_mask)

        mask_info.append(final_mask)

        update_overlay()

        input_points = []
        input_labels = []
        print("Mask refined")

def reset(event):
    global input_points, input_labels, blue_points, blue_labels, initial_mask, overlay, mask_info
    print("Resetting masks")
    input_points = []
    input_labels = []
    blue_points = []
    blue_labels = []
    generate_initial_masks(image_rgb)
    display_image(image_rgb, image_bgr, overlay)
    print("Masks reset")

def cancel_mask(event):
    global input_points, input_labels, blue_points, blue_labels, initial_mask, overlay, mask_info
    if blue_points:
        bx, by = blue_points[0]
        for i, mask in enumerate(mask_info):
            if mask[by, bx]:
                print(f"Cancelling mask at {bx}, {by}")
                mask_info.pop(i)
                initial_mask[mask > 0.5] = 0
                break

        update_overlay()
        blue_points = []
        blue_labels = []
        print("Mask cancelled")

def save(event):
    global overlay, initial_mask, mask_info
    save_name = os.path.splitext(image_files[current_image_idx])[0]
    np.savez(os.path.join(SAVE_DIR, f"{save_name}_mask.npz"), *mask_info)
    cv2.imwrite(os.path.join(SAVE_DIR, f"{save_name}_segmented.png"), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    print(f"Saved mask to: {os.path.join(SAVE_DIR, f'{save_name}_mask.npz')}")

image_rgb, image_bgr = load_image(os.path.join(IMAGE_DIR, image_files[current_image_idx]))
if image_rgb is not None and image_bgr is not None:
    mask_path = os.path.join(SAVE_DIR, f"{os.path.splitext(image_files[current_image_idx])[0]}_mask.npz")
    if os.path.exists(mask_path):
        load_existing_mask(mask_path, image_rgb)
    else:
        generate_initial_masks(image_rgb)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
if image_rgb is not None and image_bgr is not None:
    display_image(image_rgb, image_bgr, overlay, initialize=True)

input_points = []
input_labels = []
blue_points = []
blue_labels = []

def onclick(event):
    if event.inaxes == ax[1]:
        ix, iy = int(event.xdata), int(event.ydata)
        if event.button == 1:
            input_points.append([ix, iy])
            input_labels.append(1)
            ax[1].plot(ix, iy, 'ro')
        elif event.button == 3:
            blue_points.append([ix, iy])
            blue_labels.append(1)
            ax[1].plot(ix, iy, 'bo')
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

ax_button_refine = plt.axes([0.51, 0.01, 0.1, 0.075])
button_refine = Button(ax_button_refine, 'Refine Mask')

ax_button_reset = plt.axes([0.61, 0.01, 0.1, 0.075])
button_reset = Button(ax_button_reset, 'Reset')

ax_button_cancel = plt.axes([0.71, 0.01, 0.1, 0.075])
button_cancel = Button(ax_button_cancel, 'Cancel Mask')

ax_button_save = plt.axes([0.81, 0.01, 0.1, 0.075])
button_save = Button(ax_button_save, 'Save')

ax_button_next = plt.axes([0.91, 0.01, 0.1, 0.075])
button_next = Button(ax_button_next, 'Next Image')

ax_button_previous = plt.axes([0.41, 0.01, 0.1, 0.075])
button_previous = Button(ax_button_previous, 'Previous Image')

button_refine.on_clicked(refine_mask)
button_reset.on_clicked(reset)
button_cancel.on_clicked(cancel_mask)
button_save.on_clicked(save)
button_next.on_clicked(load_next_image)
button_previous.on_clicked(load_previous_image)

plt.show()
