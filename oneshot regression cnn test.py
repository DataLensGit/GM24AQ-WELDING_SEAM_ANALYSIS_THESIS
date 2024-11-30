import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
images_dir = 'images'  
model_save_path = 'oneshot_cnn_intensity_model.h5'  
img_size = (168, 375, 3)  
model = load_model(model_save_path)
def evaluate_on_training_images(images_dir, model):
    errors = []
    print("Starting evaluation on training images...")
    for folder_name in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder_name)
        json_path = os.path.join(folder_path, 'archive_info.json')
        if not os.path.exists(json_path):
            print(f"Missing JSON for {folder_name}. Skipping...")
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            actual_intensity = json_data.get('energy_intensity')
            if actual_intensity is None:
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
            img = np.expand_dims(img, axis=0)
            predicted_intensity = model.predict(img)[0][0]
            error = abs(predicted_intensity - actual_intensity)
            errors.append(error)
            print(f"Folder: {folder_name} | Image: {image_file} | Actual Intensity: {actual_intensity}, Predicted Intensity: {predicted_intensity}, Error: {error:.2f}")
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.figure(figsize=(12, 6))
    plt.bar(["Training Images"], [mean_error], yerr=[std_error], capsize=5, color='blue')
    plt.title("Mean Absolute Error for Training Images")
    plt.xlabel("Dataset")
    plt.ylabel("Mean Absolute Error")
    plt.show()
evaluate_on_training_images(images_dir, model)
