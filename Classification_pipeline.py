import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
base_dir = 'clustered_images_intensity_only'  
n_clusters = 6
knn_model_path = 'knn_model.pkl'
cnn_model_path = 'lightweight_cnn_model.h5'
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
images = []
for cluster_folder in os.listdir(base_dir):
    cluster_label = int(cluster_folder.replace('Cluster_', ''))
    cluster_path = os.path.join(base_dir, cluster_folder)
    for image_file in os.listdir(cluster_path):
        image_path = os.path.join(cluster_path, image_file)
        features.append(extract_features(image_path))
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, img_size)
        images.append(img_resized)
        labels.append(cluster_label)
features = np.array(features)
images = np.array(images) / 255.0  
labels = np.array(labels)
features_train, features_test, images_train, images_test, labels_train, labels_test = train_test_split(
    features, images, labels, test_size=0.2, random_state=42
)
print("Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, labels_train)
joblib.dump(knn, knn_model_path)
print("KNN model saved as:", knn_model_path)
print("Training CNN model...")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(225, 500, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_clusters, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(images_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)
cnn_model.save(cnn_model_path)
print("CNN model saved as:", cnn_model_path)
