import os
import json
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from joblib import load
import matplotlib.pyplot as plt
images_dir = 'images'  
one_shot_model_path = 'oneshot_intensity_model.joblib'  
img_size = (2000, 900)  
rect_width, rect_height = img_size  
one_shot_model = load(one_shot_model_path)
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
def evaluate_oneshot_model(images_dir, one_shot_model):
    errors = []
    print("Starting one-shot intensity prediction evaluation...")
    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        panorama_path = os.path.join(folder_path, 'panorama.jpg')
        json_path = os.path.join(folder_path, 'archive_info.json')
        if not os.path.exists(panorama_path) or not os.path.exists(json_path):
            print(f"Missing panorama or JSON in {folder_path}. Skipping...")
            continue
        panorama = cv2.imread(panorama_path)
        if panorama is None:
            print(f"Error loading {panorama_path}. Skipping...")
            continue
        center_x = panorama.shape[1] // 2
        center_y = panorama.shape[0] // 2
        start_x = center_x - (rect_width // 2)
        start_y = center_y - (rect_height // 2)
        segment = panorama[start_y:start_y + rect_height, start_x:start_x + rect_width]
        if segment.shape[0] != rect_height or segment.shape[1] != rect_width:
            print(f"Segment dimensions do not match the expected size for {folder_name}. Skipping...")
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            actual_intensity = json_data.get('energy_intensity')
            if actual_intensity is None:
                print(f"No intensity found in JSON for {folder_name}, skipping...")
                continue
        features = extract_features(segment)
        predicted_intensity = one_shot_model.predict([features])[0]
        error = abs(predicted_intensity - actual_intensity)
        errors.append(error)
        print(
            f"Folder: {folder_name} | Actual Intensity: {actual_intensity}, Predicted Intensity: {predicted_intensity}, Error: {error:.2f}")
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.figure(figsize=(12, 6))
    plt.bar(["One-Shot Model"], [mean_error], yerr=[std_error], capsize=5, color='blue')
    plt.title("Mean Absolute Error for One-Shot Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Absolute Error")
    plt.show()
evaluate_oneshot_model(images_dir, one_shot_model)
