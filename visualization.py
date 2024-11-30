import os
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
base_dir = "images"
data = []
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    if os.path.isdir(seam_path):
        json_file = os.path.join(seam_path, 'archive_info.json')
        with open(json_file) as f:
            info = json.load(f)
        if info['Current'] == 500 and info['Voltage'] == 500 and info['Speed'] == 500:
            continue  
        data.append({
            'Seam': seam_folder,
            'Current (A)': info['Current'],
            'Voltage (V)': info['Voltage'],
            'Speed (mm/s)': info['Speed'],
            'Performance (W)': info.get('performance', 0),  
            'Energy Intensity (W/mm/s)': info.get('energy_intensity', 0)  
        })
df = pd.DataFrame(data)
sorted_by_performance = df.sort_values(by='Performance (W)').drop_duplicates('Performance (W)')
sorted_by_intensity = df.sort_values(by='Energy Intensity (W/mm/s)').drop_duplicates('Energy Intensity (W/mm/s)')
indices_performance = np.linspace(0, len(sorted_by_performance) - 1, min(6, len(sorted_by_performance)), dtype=int)
indices_intensity = np.linspace(0, len(sorted_by_intensity) - 1, min(6, len(sorted_by_intensity)), dtype=int)
samples_performance = sorted_by_performance.iloc[indices_performance]
samples_intensity = sorted_by_intensity.iloc[indices_intensity]
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle("Welding Seams with Increasing Performance (Top) vs Increasing Intensity (Bottom)")
def load_image(path, size=(150, 150)):
    image = cv2.imread(path)
    if image is not None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)  
for idx, (_, row) in enumerate(samples_performance.iterrows()):
    image_path = os.path.join(base_dir, row['Seam'], 'panorama.jpg')
    image = load_image(image_path)
    axes[0, idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, idx].axis('off')
    axes[0, idx].set_title(f"Perf: {row['Performance (W)']:.1f}\nSeam: {row['Seam']}")
for idx, (_, row) in enumerate(samples_intensity.iterrows()):
    image_path = os.path.join(base_dir, row['Seam'], 'panorama.jpg')
    image = load_image(image_path)
    axes[1, idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, idx].axis('off')
    axes[1, idx].set_title(f"Intens: {row['Energy Intensity (W/mm/s)']:.1f}\nSeam: {row['Seam']}")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
