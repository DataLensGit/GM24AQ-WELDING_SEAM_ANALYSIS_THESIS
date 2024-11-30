import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
base_dir = 'clustered_images_intensity_only'  
cnn_model_path = 'cnn_model_highres.h5'
img_size = (168, 375)  
n_clusters = 6
images = []
labels = []
cluster_image_counts = {folder: len(os.listdir(os.path.join(base_dir, folder))) for folder in os.listdir(base_dir)}
max_images = max(cluster_image_counts.values())
for cluster_folder in os.listdir(base_dir):
    cluster_label = int(cluster_folder.replace('Cluster_', ''))
    cluster_path = os.path.join(base_dir, cluster_folder)
    cluster_images = []
    for image_file in os.listdir(cluster_path):
        image_path = os.path.join(cluster_path, image_file)
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, img_size)
        cluster_images.append(img_resized)
    while len(cluster_images) < max_images:
        cluster_images.append(random.choice(cluster_images))
    images.extend(cluster_images)
    labels.extend([cluster_label] * max_images)
images = np.array(images) / 255.0
labels = np.array(labels)
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
print("Training CNN model...")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[1], img_size[0], 3)),
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
cnn_model.fit(images_train, labels_train, epochs=11, batch_size=20, validation_split=0.2)
cnn_model.save(cnn_model_path)
print("CNN model saved as:", cnn_model_path)
