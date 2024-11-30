import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16  
import os
import json
import glob
num_outputs = 3  
def load_feature_extractor(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    return Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
def extract_features(feature_extractor, image_paths):
    features = []
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))  
        image = tf.keras.applications.vgg16.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        feature = feature_extractor.predict(image)
        features.append(feature.flatten())
    return np.array(features)
def normalize_labels(labels):
    current_min, current_max = 40.0, 200.0
    voltage_min, voltage_max = 18.0, 23.8
    speed_min, speed_max = 20.0, 90.0
    current_norm = (labels[0] - current_min) / (current_max - current_min)
    voltage_norm = (labels[1] - voltage_min) / (voltage_max - voltage_min)
    speed_norm = (labels[2] - speed_min) / (speed_max - speed_min)
    return [current_norm, voltage_norm, speed_norm]
image_paths = []
labels = []
base_dir = 'images'
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        continue
    with open(json_file) as f:
        info = json.load(f)
        label = normalize_labels([info['Current'], info['Voltage'], info['Speed']])
    for image_file in glob.glob(os.path.join(seam_path, '*augmented*.jpg')):
        image_paths.append(image_file)
        labels.append(label)
labels = np.array(labels)
feature_extractor = load_feature_extractor((224, 224, 3))
features = extract_features(feature_extractor, image_paths)
train_size = int(0.9 * len(features))
train_features, val_features = features[:train_size], features[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_outputs)  
    ])
    return model
model = build_model(input_shape=(train_features.shape[1],))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history = model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    epochs=10,
    batch_size=32,
    verbose=1
)
model_save_path = 'welding_feature_model.keras'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
