import os
import json
import numpy as np
import shutil
from sklearn.cluster import KMeans
from collections import defaultdict
base_dir = 'images'
output_dir = 'clustered_images_intensity_only'  
n_clusters = 6  
intensity_data = []  
os.makedirs(output_dir, exist_ok=True)
for i in range(n_clusters):
    os.makedirs(os.path.join(output_dir, f"Cluster_{i}"), exist_ok=True)
print("Collecting intensity values...")
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        print(f"No JSON file found in {seam_path}, skipping...")
        continue
    with open(json_file) as f:
        info = json.load(f)
        intensity = info.get('energy_intensity', 0)
    for image_file in os.listdir(seam_path):
        if "augmented" in image_file and image_file.endswith('.jpg'):
            image_path = os.path.join(seam_path, image_file)
            intensity_data.append((intensity, image_path))
print(f"Total images collected: {len(intensity_data)}")
intensity_values = np.array([item[0] for item in intensity_data]).reshape(-1, 1)
print("Performing KMeans clustering based on intensity values...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(intensity_values)
print("KMeans clustering completed.")
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    clusters[label].append(intensity_data[idx][1])  
for cluster_id, image_paths in clusters.items():
    cluster_dir = os.path.join(output_dir, f"Cluster_{cluster_id}")
    for image_path in image_paths:
        shutil.copy(image_path, cluster_dir)
        print(f"Copied {image_path} to {cluster_dir}")
print("Images have been organized into clusters based on intensity.")
