import os
import json
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skimage.feature import graycomatrix, graycoprops, hog
from joblib import dump, load
base_dir = 'clustered_images_intensity_only'  
original_images_dir = 'images'  
models_dir = 'intensity_models'  
img_size = (2000, 900)  
model_type = 'random_forest'  
os.makedirs(models_dir, exist_ok=True)
def extract_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(grayscale)
    avg_color = np.mean(image, axis=(0, 1))
    edges = np.sum(cv2.Canny(grayscale, 100, 200) > 0)
    glcm = graycomatrix(grayscale, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    resized_for_hog = cv2.resize(grayscale, (500, 225))  
    hog_features = hog(resized_for_hog, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False)
    hog_mean = np.mean(hog_features)
    return [contrast, *avg_color, edges, texture_contrast, texture_homogeneity, hog_mean]
def train_regression_models(base_dir, original_images_dir, models_dir, model_type='random_forest'):
    for cluster_folder in os.listdir(base_dir):
        cluster_label = int(cluster_folder.replace('Cluster_', ''))
        cluster_path = os.path.join(base_dir, cluster_folder)
        features = []
        intensities = []
        for image_file in os.listdir(cluster_path):
            image_path = os.path.join(cluster_path, image_file)
            img = cv2.imread(image_path)
            if img is None or img.shape[:2] != img_size:
                print(f"Skipping {image_file} due to size mismatch or read error.")
                continue
            features.append(extract_features(img))
            image_base_name = image_file.split('_augmented_')[0]  
            original_folder = os.path.join(original_images_dir, image_base_name)
            json_path = os.path.join(original_folder, 'archive_info.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    intensity = json.load(f).get('energy_intensity')
                    if intensity is not None:
                        intensities.append(intensity)
                    else:
                        print(f"No intensity found in JSON for {image_file}, skipping...")
            else:
                print(f"No JSON file found in original folder for {image_file}, skipping...")
        features = np.array(features)
        intensities = np.array(intensities)
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type specified. Choose 'linear' or 'random_forest'.")
        model.fit(features, intensities)
        model_path = os.path.join(models_dir, f"intensity_model_cluster_{cluster_label}.joblib")
        dump(model, model_path)
        print(f"Model saved for Cluster {cluster_label} at {model_path}")
train_regression_models(base_dir, original_images_dir, models_dir, model_type=model_type)
