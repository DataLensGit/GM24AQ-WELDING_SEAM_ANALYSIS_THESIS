import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops, hog
import matplotlib.pyplot as plt
images_dir = 'images'  
clustered_dir = 'clustered_images_intensity_only'  
knn_model_path = 'knn_model.pkl'
cnn_model_path = 'cnn_model_highres.h5'
img_size = (168, 375)
rect_width, rect_height = 4500, 2000
knn_model = joblib.load(knn_model_path)
cnn_model = load_model(cnn_model_path)
def extract_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale, (1000, 450))
    contrast = np.std(resized_image)
    avg_color = np.mean(image, axis=(0, 1))
    edges = np.sum(cv2.Canny(resized_image, 100, 200) > 0)
    glcm = graycomatrix(resized_image, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    hog_features = hog(resized_image, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False)
    hog_mean = np.mean(hog_features)
    return [contrast, *avg_color, edges, texture_contrast, texture_homogeneity, hog_mean]
def get_correct_cluster(seam_folder):
    for cluster_folder in os.listdir(clustered_dir):
        cluster_path = os.path.join(clustered_dir, cluster_folder)
        if os.path.isdir(cluster_path):
            for image_file in os.listdir(cluster_path):
                if image_file.startswith(seam_folder):
                    return int(cluster_folder.replace('Cluster_', ''))
    return None
def evaluate_classification(images_dir):
    both_correct = 0
    only_knn_correct = 0
    only_cnn_correct = 0
    both_incorrect = 0
    print("Starting evaluation...")
    for seam_folder in os.listdir(images_dir):
        seam_path = os.path.join(images_dir, seam_folder)
        panorama_path = os.path.join(seam_path, 'panorama.jpg')
        if not os.path.exists(panorama_path):
            print(f"No panorama found in {seam_path}")
            continue
        correct_cluster = get_correct_cluster(seam_folder)
        if correct_cluster is None:
            print(f"No correct cluster found for {seam_folder}")
            continue
        panorama = cv2.imread(panorama_path)
        start_x = (panorama.shape[1] - rect_width) // 2
        start_y = (panorama.shape[0] - rect_height) // 2
        segment_width = rect_width // 3
        for i in range(1, 4):
            segment_start_x = start_x + (i - 1) * segment_width
            crop = panorama[start_y:start_y + rect_height, segment_start_x:segment_start_x + segment_width]
            if crop.size == 0:
                print(f"Empty crop for segment {i} in {seam_folder}")
                continue
            feature_vector = np.array(extract_features(crop)).reshape(1, -1)
            knn_pred = knn_model.predict(feature_vector)[0]
            img_resized = cv2.resize(crop, img_size) / 255.0
            img_resized = np.expand_dims(img_resized, axis=0)
            cnn_pred = np.argmax(cnn_model.predict(img_resized), axis=-1)[0]
            knn_correct = knn_pred == correct_cluster
            cnn_correct = cnn_pred == correct_cluster
            print(f"Folder: {seam_folder} | Segment: {i}\nCorrect Cluster: {correct_cluster}\nKNN Pred: {knn_pred}, CNN Pred: {cnn_pred}")
            if knn_correct and cnn_correct:
                both_correct += 1
            elif knn_correct:
                only_knn_correct += 1
            elif cnn_correct:
                only_cnn_correct += 1
            else:
                both_incorrect += 1
    print(f"Results: Both Correct = {both_correct}, Only KNN Correct = {only_knn_correct}, "
          f"Only CNN Correct = {only_cnn_correct}, Both Incorrect = {both_incorrect}")
    categories = ['Both Correct', 'Only KNN Correct', 'Only CNN Correct', 'Both Incorrect']
    counts = [both_correct, only_knn_correct, only_cnn_correct, both_incorrect]
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=['green', 'blue', 'orange', 'red'])
    plt.title("Classification Accuracy by Model (3 Segments per Image)")
    plt.xlabel("Model Prediction Category")
    plt.ylabel("Number of Segments")
    plt.show()
evaluate_classification(images_dir)
