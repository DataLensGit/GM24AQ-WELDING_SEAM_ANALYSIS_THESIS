import os
import json
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import cv2
import glob
import matplotlib.pyplot as plt
input_shape = (300, 500, 3)  
def mae_intensity(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))
image_paths = []
labels = []
base_dir = 'images'
print("Loading image paths and labels...")
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        continue
    with open(json_file) as f:
        info = json.load(f)
        label = info.get('energy_intensity', 0)  
    for image_file in glob.glob(os.path.join(seam_path, '*augmented*.jpg')):
        image_paths.append(image_file)
        labels.append(label)
print(f"Found {len(image_paths)} images with labels.")
first_image_path = image_paths[0]
first_image = cv2.imread(first_image_path)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("First Training Image - Visualization")
plt.show()
def load_image(img_path, target_size=(500, 300)):
    image = cv2.imread(img_path)
    return cv2.resize(image, target_size)
print("Loading and resizing images...")
images = [load_image(img_path) for img_path in image_paths]
images = np.array(images)
labels = np.array(labels)
train_size = int(0.9 * len(images))
train_images, val_images = images[:train_size], images[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)  
    ])
    return model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = build_cnn_model(input_shape=input_shape)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', mae_intensity]
)
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=4),
    validation_data=(val_images, val_labels),
    epochs=10,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)
model_save_path = 'welding_intensity_model_cnn_improved.keras'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")