import os
import json
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, hog
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
images_dir = 'images'  
model_save_path = 'oneshot_intensity_model.joblib'  
img_size = (2000, 900)  
model_type = 'random_forest'  
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
def train_oneshot_regression_model(images_dir, model_save_path, model_type='random_forest'):
    features = []
    intensities = []
    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        json_path = os.path.join(folder_path, 'archive_info.json')
        if not os.path.exists(json_path):
            print(f"Missing JSON for {folder_name}. Skipping...")
            continue
        with open(json_path, 'r') as f:
            intensity = json.load(f).get('energy_intensity')
            if intensity is None:
                print(f"No intensity found in JSON for {folder_name}, skipping...")
                continue
        for image_file in os.listdir(folder_path):
            if 'augmented' not in image_file or not image_file.endswith(('.jpg', '.png')):
                continue
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            if img is None or img.shape[:2] != img_size:
                print(f"Skipping {image_file} in {folder_name} due to size mismatch or read error.")
                continue
            features.append(extract_features(img))
            intensities.append(intensity)
    features = np.array(features)
    intensities = np.array(intensities)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_val, y_train, y_val = train_test_split(features, intensities, test_size=0.2, random_state=42)
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type specified. Choose 'linear' or 'random_forest'.")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error on validation set: {mse:.2f}")
    dump(model, model_save_path)
    print(f"One-shot model saved at {model_save_path}")
train_oneshot_regression_model(images_dir, model_save_path, model_type=model_type)
