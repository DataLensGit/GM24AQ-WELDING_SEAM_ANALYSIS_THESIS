import os
import json
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
base_dir = 'images'
output_dir = 'ssim_clusters'
n_clusters = 15
n_threads = 8
thumbnail_size = (100, 100)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
panorama_data = []
print("Starting panorama image collection process...")
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    panorama_path = os.path.join(seam_path, 'panorama.jpg')
    if not os.path.exists(json_file) or not os.path.exists(panorama_path):
        print(f"Skipping folder '{seam_folder}' (missing JSON or panorama)")
        continue
    with open(json_file) as f:
        info = json.load(f)
        intensity = info.get('energy_intensity', 0)
    panorama_data.append((panorama_path, intensity, seam_folder))
print(f"Total panoramas collected for processing: {len(panorama_data)}")
def load_image_resized(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, thumbnail_size)
    return img
def calculate_ssim_pair(idx1, idx2):
    img1 = load_image_resized(panorama_data[idx1][0])
    img2 = load_image_resized(panorama_data[idx2][0])
    return ssim(img1, img2, multichannel=True)
ssim_matrix = np.zeros((len(panorama_data), len(panorama_data)))
print("Calculating SSIM scores between panorama image pairs...")
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    futures = {}
    total_tasks = (len(panorama_data) * (len(panorama_data) - 1)) // 2
    task_count = 0
    for i in range(len(panorama_data)):
        for j in range(i + 1, len(panorama_data)):
            futures[executor.submit(calculate_ssim_pair, i, j)] = (i, j)
            task_count += 1
            if task_count % 50 == 0 or task_count == total_tasks:
                print(f"Submitted {task_count}/{total_tasks} SSIM tasks")
    completed_tasks = 0
    for future in as_completed(futures):
        i, j = futures[future]
        ssim_matrix[i, j] = ssim_matrix[j, i] = future.result()
        completed_tasks += 1
        if completed_tasks % 50 == 0 or completed_tasks == total_tasks:
            print(f"Completed {completed_tasks}/{total_tasks} SSIM tasks")
print("SSIM matrix calculation finished.")
print(f"Clustering panorama images into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(ssim_matrix)
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    panorama_path, intensity, seam_folder = panorama_data[idx]
    cluster_dir = os.path.join(output_dir, f"Cluster_{label}")
    os.makedirs(cluster_dir, exist_ok=True)
    shutil.copy(panorama_path, cluster_dir)
    print(f"Copied panorama image '{panorama_path}' to Cluster {label}")
    for image_file in os.listdir(os.path.join(base_dir, seam_folder)):
        if "augmented" in image_file and image_file.endswith('.jpg'):
            augmented_path = os.path.join(base_dir, seam_folder, image_file)
            shutil.copy(augmented_path, cluster_dir)
            print(f"Copied augmented image '{augmented_path}' to Cluster {label}")
print("All images have been copied to their respective clusters.")
intensity_ranges = defaultdict(list)
for idx, label in enumerate(labels):
    intensity_ranges[label].append(panorama_data[idx][1])
plt.figure(figsize=(15, 10))
for cluster_id, intensity_list in intensity_ranges.items():
    plt.subplot(3, 5, cluster_id + 1)
    sns.histplot(intensity_list, kde=True, bins=10, color="skyblue")
    plt.title(f"Cluster {cluster_id}\nIntensity Range")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
plt.tight_layout()
plt.suptitle("Intensity Ranges Across Clusters", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
