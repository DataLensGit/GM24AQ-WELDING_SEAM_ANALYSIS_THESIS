import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
clusters_base_dir = 'clustered_images_intensity_only'
original_images_dir = 'images'
intensity_ranges = defaultdict(list)
for cluster_folder in os.listdir(clusters_base_dir):
    cluster_path = os.path.join(clusters_base_dir, cluster_folder)
    if not os.path.isdir(cluster_path):
        continue
    for image_file in os.listdir(cluster_path):
        image_base_name = image_file.split('_augmented_')[0]
        original_folder = os.path.join(original_images_dir, image_base_name)
        json_path = os.path.join(original_folder, 'archive_info.json')
        if not os.path.exists(json_path):
            print(f"No JSON file found for {image_file} in original folder '{original_folder}', skipping...")
            continue
        with open(json_path, 'r') as f:
            info = json.load(f)
            intensity = info.get('energy_intensity', None)
            if intensity is not None:
                intensity_ranges[cluster_folder].append(intensity)
            else:
                print(f"No intensity found in JSON for {image_file}, skipping...")
sorted_intensity_ranges = dict(sorted(intensity_ranges.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else float('inf')))
n_clusters = len(sorted_intensity_ranges)
n_cols = 3
n_rows = (n_clusters + n_cols - 1) // n_cols  
plt.figure(figsize=(n_cols * 5, n_rows * 4))
for idx, (cluster, intensities) in enumerate(sorted_intensity_ranges.items()):
    plt.subplot(n_rows, n_cols, idx + 1)
    sns.boxplot(y=intensities, color="skyblue")
    plt.ylim(0, 250)  
    plt.title(f"{cluster}\n Intensity Range")
    plt.ylabel("Intensity")
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
