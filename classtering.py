import os
import json
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops, hog
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
base_dir = 'images'
output_dir = 'clustered_images'
output_images_dir = 'output_images'
n_clusters = 12
knn_model_path = 'knn_model.pkl'
n_threads = 8
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)
for i in range(n_clusters):
    os.makedirs(os.path.join(output_dir, f"Cluster_{i}"), exist_ok=True)
def extract_features(image_path):
    print(f"Extracting features from: {image_path}")
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(grayscale)
    avg_color = np.mean(image, axis=(0, 1))
    edges = cv2.Canny(grayscale, 100, 200)
    edge_count = np.sum(edges > 0)
    glcm = graycomatrix(grayscale, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    _, hog_image = hog(grayscale, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_mean = np.mean(hog_image)
    grayscale_path = os.path.join(output_images_dir, f"{os.path.basename(image_path)}_grayscale.jpg")
    cv2.imwrite(grayscale_path, grayscale)
    edges_path = os.path.join(output_images_dir, f"{os.path.basename(image_path)}_edges.jpg")
    cv2.imwrite(edges_path, edges)
    hog_path = os.path.join(output_images_dir, f"{os.path.basename(image_path)}_hog.jpg")
    cv2.imwrite(hog_path, (hog_image * 255).astype("uint8"))
    return [contrast, *avg_color, edge_count, texture_contrast, texture_homogeneity, hog_mean]
features, cluster_labels, image_paths = [], [], []
tasks = []
print("Collecting tasks for image processing...")
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        continue
    with open(json_file) as f:
        info = json.load(f)
        intensity = info.get('energy_intensity', 0)
        print(f"Loaded intensity {intensity} for {seam_folder}")
    for image_file in os.listdir(seam_path):
        if "augmented" in image_file and image_file.endswith('.jpg'):
            tasks.append((seam_folder, image_file, intensity))
def process_image(seam_folder, image_file, intensity):
    image_path = os.path.join(base_dir, seam_folder, image_file)
    feature_vector = extract_features(image_path)
    return feature_vector, intensity, image_path
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    futures = [executor.submit(process_image, *task) for task in tasks]
    for future in as_completed(futures):
        feature_vector, intensity, image_path = future.result()
        features.append(feature_vector)
        cluster_labels.append(intensity)
        image_paths.append(image_path)
        print(f"Processed image: {image_path}")
print(f"Total images processed: {len(image_paths)}")
features = np.array(features)
cluster_labels = np.array(cluster_labels).reshape(-1, 1)
print("Performing KMeans clustering...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(cluster_labels)
print("KMeans clustering completed.")
print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features, kmeans.labels_)
print("KNN training completed.")
joblib.dump(knn, knn_model_path)
print(f"KNN model saved to {knn_model_path}")
def classify_and_copy(image_path, model, base_output_dir):
    feature_vector = np.array(extract_features(image_path)).reshape(1, -1)
    cluster = model.predict(feature_vector)[0]
    destination_folder = os.path.join(base_output_dir, f"Cluster_{cluster}")
    shutil.copy(image_path, destination_folder)
    print(f"Copied {image_path} to {destination_folder}")
print("Classifying and copying images into clusters...")
model = joblib.load(knn_model_path)
copy_tasks = []
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    panorama_path = os.path.join(seam_path, "panorama.jpg")
    if os.path.exists(panorama_path):
        copy_tasks.append((panorama_path, model, output_dir))
    for image_file in os.listdir(seam_path):
        if "augmented" in image_file and image_file.endswith('.jpg'):
            augmented_path = os.path.join(seam_path, image_file)
            copy_tasks.append((augmented_path, model, output_dir))
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    copy_futures = [executor.submit(classify_and_copy, *task) for task in copy_tasks]
    for future in as_completed(copy_futures):
        future.result()
print("Image classification and sorting complete.")
