import os
import numpy as np
import matplotlib.pyplot as plt

# Set the directories
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

def get_color(index, total):
    colormaps = ['tab20', 'tab20b', 'tab20c']
    cmap = plt.get_cmap(colormaps[index % len(colormaps)])
    color = cmap((index // len(colormaps)) / (total // len(colormaps) + 1))
    return [int(c * 255) for c in color[:3]]

def load_and_display_masks(mask_file):
    with np.load(mask_file) as data:
        masks = [data[key] for key in data]

    mask_shape = masks[0].shape
    for i, mask in enumerate(masks):
        overlay = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
        color = get_color(i, len(masks))
        for j in range(3):
            overlay[mask > 0.5, j] = color[j]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f'Mask {i+1} of {os.path.basename(mask_file)}')
        plt.axis('off')
        plt.show()

# Load and display masks for each .npz file
for mask_file in os.listdir(SAVE_DIR):
    if mask_file.lower().endswith('_mask.npz'):
        load_and_display_masks(os.path.join(SAVE_DIR, mask_file))
        input("Press any key to continue to the next mask...")
