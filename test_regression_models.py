import os
import json
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from tensorflow.keras.models import load_model
from joblib import load
from collections import defaultdict
import matplotlib.pyplot as plt
images_dir = 'images'  
clustered_images_dir = 'clustered_images_intensity_only'  
cnn_classification_model_path = 'cnn_model_highres.h5'
models_dir = 'intensity_models'  
cnn_intensity_models_dir = 'cnn_intensity_models'  
img_size = (168, 375)
img_size_cnn = (375, 168)
rect_width, rect_height = 4500, 2000
cnn_classification_model = load_model(cnn_classification_model_path)
def classify_and_predict_intensity(segment, cnn_classification_model, models_dir, cnn_intensity_models_dir):
    segment_resized = cv2.resize(segment, (img_size[0], img_size[1])) / 255.0  
    segment_resized = np.expand_dims(segment_resized, axis=0)
    predicted_cluster = np.argmax(cnn_classification_model.predict(segment_resized), axis=-1)[0]
    print(f"Predicted Cluster: {predicted_cluster}")
    feature_model_path = os.path.join(models_dir, f"intensity_model_cluster_{predicted_cluster}.joblib")
    if os.path.exists(feature_model_path):
        regression_model = load(feature_model_path)
        features = extract_features(segment)
        feature_based_intensity = regression_model.predict([features])[0]
    else:
        print(f"Feature-based regression model not found for cluster {predicted_cluster}. Skipping...")
        feature_based_intensity = None
    segment_resized_cnn = cv2.resize(segment, (img_size_cnn[0], img_size_cnn[1])) / 255.0  
    segment_resized_cnn = np.expand_dims(segment_resized_cnn, axis=0)
    cnn_model_path = os.path.join(cnn_intensity_models_dir, f"cnn_intensity_model_cluster_{predicted_cluster}.h5")
    if os.path.exists(cnn_model_path):
        cnn_intensity_model = load_model(cnn_model_path)
        cnn_based_intensity = cnn_intensity_model.predict(segment_resized_cnn)[0][0]
    else:
        print(f"CNN-based regression model not found for cluster {predicted_cluster}. Skipping...")
        cnn_based_intensity = None
    return feature_based_intensity, cnn_based_intensity, predicted_cluster
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
def get_actual_cluster(seam_folder):
    for cluster_folder in os.listdir(clustered_images_dir):
        cluster_path = os.path.join(clustered_images_dir, cluster_folder)
        if os.path.isdir(cluster_path):
            for image_file in os.listdir(cluster_path):
                if image_file.startswith(seam_folder):
                    return int(cluster_folder.replace('Cluster_', ''))
    return None
def evaluate_intensity_prediction(images_dir, cnn_classification_model, models_dir, cnn_intensity_models_dir):
    correct_cluster_feature_errors = defaultdict(list)
    incorrect_cluster_feature_errors = defaultdict(list)
    correct_cluster_cnn_errors = defaultdict(list)
    incorrect_cluster_cnn_errors = defaultdict(list)
    print("Starting intensity prediction evaluation...")
    for seam_folder in os.listdir(images_dir):
        seam_path = os.path.join(images_dir, seam_folder)
        panorama_path = os.path.join(seam_path, 'panorama.jpg')
        json_path = os.path.join(seam_path, 'archive_info.json')
        if not os.path.exists(panorama_path) or not os.path.exists(json_path):
            print(f"Missing panorama or JSON in {seam_path}. Skipping...")
            continue
        panorama = cv2.imread(panorama_path)
        start_x = (panorama.shape[1] - rect_width) // 2
        start_y = (panorama.shape[0] - rect_height) // 2
        segment_width = rect_width // 3
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            actual_intensity = json_data.get('energy_intensity')
            if actual_intensity is None:
                print(f"No intensity found in JSON for {seam_folder}, skipping...")
                continue
        actual_cluster = get_actual_cluster(seam_folder)
        if actual_cluster is None:
            print(f"Could not determine correct cluster for {seam_folder}. Skipping...")
            continue
        for i in range(1, 3):
            segment_start_x = start_x + (i - 1) * segment_width
            segment = panorama[start_y:start_y + rect_height, segment_start_x:segment_start_x + segment_width]
            feature_based_intensity, cnn_based_intensity, predicted_cluster = classify_and_predict_intensity(
                segment, cnn_classification_model, models_dir, cnn_intensity_models_dir
            )
            if feature_based_intensity is not None:
                error = abs(feature_based_intensity - actual_intensity)
                if predicted_cluster == actual_cluster:
                    correct_cluster_feature_errors[predicted_cluster].append(error)
                else:
                    incorrect_cluster_feature_errors[predicted_cluster].append(error)
            if cnn_based_intensity is not None:
                cnn_error = abs(cnn_based_intensity - actual_intensity)
                if predicted_cluster == actual_cluster:
                    correct_cluster_cnn_errors[predicted_cluster].append(cnn_error)
                else:
                    incorrect_cluster_cnn_errors[predicted_cluster].append(cnn_error)
    plt.figure(figsize=(12, 6))
    plt.bar(correct_cluster_feature_errors.keys(),
            [np.mean(errors) for errors in correct_cluster_feature_errors.values()],
            yerr=[np.std(errors) for errors in correct_cluster_feature_errors.values()], capsize=5, color='green')
    plt.title("Mean Absolute Error per Cluster (Feature-Based, Correct Predictions)")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Absolute Error")
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.bar(incorrect_cluster_feature_errors.keys(),
            [np.mean(errors) for errors in incorrect_cluster_feature_errors.values()],
            yerr=[np.std(errors) for errors in incorrect_cluster_feature_errors.values()], capsize=5, color='orange')
    plt.title("Mean Absolute Error per Cluster (Feature-Based, Incorrect Predictions)")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Absolute Error")
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.bar(correct_cluster_cnn_errors.keys(),
            [np.mean(errors) for errors in correct_cluster_cnn_errors.values()],
            yerr=[np.std(errors) for errors in correct_cluster_cnn_errors.values()], capsize=5, color='blue')
    plt.title("Mean Absolute Error per Cluster (CNN-Based, Correct Predictions)")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Absolute Error")
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.bar(incorrect_cluster_cnn_errors.keys(),
            [np.mean(errors) for errors in incorrect_cluster_cnn_errors.values()],
            yerr=[np.std(errors) for errors in incorrect_cluster_cnn_errors.values()], capsize=5, color='red')
    plt.title("Mean Absolute Error per Cluster (CNN-Based, Incorrect Predictions)")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Absolute Error")
    plt.show()
evaluate_intensity_prediction(images_dir, cnn_classification_model, models_dir, cnn_intensity_models_dir)
