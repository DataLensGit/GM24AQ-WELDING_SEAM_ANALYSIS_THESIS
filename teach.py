import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import json
import glob
import cv2
input_shape = (300, 500, 3)
num_outputs = 2  
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA GPU detected:", gpus)
else:
    print("No GPU detected, check CUDA installation.")
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_outputs)  
    ])
    return model
@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    performance_pred, intensity_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    performance_true, intensity_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    mse_performance = tf.reduce_mean(tf.square(performance_pred - performance_true), name="mse_performance")
    mse_intensity = tf.reduce_mean(tf.square(intensity_pred - intensity_true), name="mse_intensity")
    total_loss = mse_performance + mse_intensity
    return total_loss
def mse_performance(y_true, y_pred):
    performance_true = tf.split(y_true, num_or_size_splits=2, axis=-1)[0]
    performance_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)[0]
    return tf.reduce_mean(tf.square(performance_pred - performance_true))
def mse_intensity(y_true, y_pred):
    intensity_true = tf.split(y_true, num_or_size_splits=2, axis=-1)[1]
    intensity_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)[1]
    return tf.reduce_mean(tf.square(intensity_pred - intensity_true))
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=custom_loss,
    metrics=['mae', mse_performance, mse_intensity]
)
def normalize_labels(labels):
    performance_min, performance_max = 720.0, 4760.0
    intensity_min, intensity_max = 8.0, 238.0
    performance_norm = (labels[0] - performance_min) / (performance_max - performance_min)
    intensity_norm = (labels[1] - intensity_min) / (intensity_max - intensity_min)
    return [performance_norm, intensity_norm]
def load_image_and_label(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 500])
    image = image / 255.0  
    normalized_label = normalize_labels(label)
    return image, normalized_label
image_paths = []
labels = []
base_dir = 'images'  
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        print(f"Warning: Missing JSON file in {seam_path}")
        continue
    with open(json_file) as f:
        info = json.load(f)
        label = [
            float(info['Current']) * float(info['Voltage']),
            float(info['Current']) * float(info['Voltage']) / (float(info['Speed']) + 1e-6)
        ]
    for image_file in glob.glob(os.path.join(seam_path, '*augmented*.jpg')):
        image_paths.append(image_file)
        labels.append(label)
print(f"Found {len(image_paths)} augmented images with complete data.")
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(lambda x, y: load_image_and_label(x, y))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE).repeat()
train_size = int(0.9 * len(image_paths))
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)
print(f"Total images: {len(image_paths)}")
print(f"Training images: {train_size}")
print(f"Validation images: {len(image_paths) - train_size}")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=6,
    steps_per_epoch=train_size // 32,  
    validation_steps=(len(image_paths) - train_size) // 32,
    verbose=1
)
model_save_path = 'welding_model_performance_intensity2.keras'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
