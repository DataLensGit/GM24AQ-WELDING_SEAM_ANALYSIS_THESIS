
import os
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
base_dir = 'clustered_images_intensity_only'  
knn_model_path = 'knn_model.pkl'
img_size = (225, 500)  
def extract_features(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale, img_size)
    contrast = np.std(resized_image)
    avg_color = np.mean(image, axis=(0, 1))
    edges = np.sum(cv2.Canny(resized_image, 100, 200) > 0)
    glcm = graycomatrix(resized_image, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    hog_features = hog(resized_image, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False)
    hog_mean = np.mean(hog_features)
    return [contrast, *avg_color, edges, texture_contrast, texture_homogeneity, hog_mean]
features = []
labels = []
for cluster_folder in os.listdir(base_dir):
    cluster_label = int(cluster_folder.replace('Cluster_', ''))
    cluster_path = os.path.join(base_dir, cluster_folder)
    for image_file in os.listdir(cluster_path):
        image_path = os.path.join(cluster_path, image_file)
        features.append(extract_features(image_path))
        labels.append(cluster_label)
features = np.array(features)
labels = np.array(labels)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
print("Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, labels_train)
joblib.dump(knn, knn_model_path)
print("KNN model saved as:", knn_model_path)
