import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
images_dir = 'images'  
model_save_path = 'oneshot_cnn_intensity_model.h5'  
img_size = (168, 375, 3)  
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)  
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
    return model
def train_oneshot_cnn_model(images_dir, model_save_path):
    images = []
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
            if img is None:
                print(f"Skipping {image_file} due to read error.")
                continue
            img = cv2.resize(img, (img_size[1], img_size[0])) / 255.0
            images.append(img)
            intensities.append(intensity)
    images = np.array(images)
    intensities = np.array(intensities)
    X_train, X_val, y_train, y_val = train_test_split(images, intensities, test_size=0.2, random_state=42)
    model = create_cnn_model(input_shape=img_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, callbacks=[early_stopping])
    model.save(model_save_path)
    print(f"One-shot CNN model saved at {model_save_path}")
train_oneshot_cnn_model(images_dir, model_save_path)
