import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
base_dir = 'clustered_images_intensity_only'  
original_images_dir = 'images'  
models_dir = 'cnn_intensity_models'  
img_size = (168, 375, 3)  
os.makedirs(models_dir, exist_ok=True)
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
def train_cnn_models(base_dir, original_images_dir, models_dir):
    for cluster_folder in os.listdir(base_dir):
        cluster_label = int(cluster_folder.replace('Cluster_', ''))
        cluster_path = os.path.join(base_dir, cluster_folder)
        images = []
        intensities = []
        for image_file in os.listdir(cluster_path):
            image_path = os.path.join(cluster_path, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {image_file} due to read error.")
                continue
            img = cv2.resize(img, (img_size[1], img_size[0])) / 255.0
            images.append(img)
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
        images = np.array(images)
        intensities = np.array(intensities)
        if len(images) == 0:
            print(f"No valid images for cluster {cluster_label}. Skipping...")
            continue
        model = create_cnn_model(input_shape=img_size)
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(images, intensities, epochs=20, batch_size=8, callbacks=[early_stopping])
        model_path = os.path.join(models_dir, f"cnn_intensity_model_cluster_{cluster_label}.h5")
        save_model(model, model_path)
        print(f"Model saved for Cluster {cluster_label} at {model_path}")
train_cnn_models(base_dir, original_images_dir, models_dir)
