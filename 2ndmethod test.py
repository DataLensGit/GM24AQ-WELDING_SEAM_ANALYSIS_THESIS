import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
image_path = 'panorama.jpg'  
model_path = 'welding_intensity_model_cnn_improved.keras'  
output_image = "prediction_report.png"  
def mae_intensity(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))
model = tf.keras.models.load_model(model_path, custom_objects={
    'mae_intensity': mae_intensity
})
def denormalize_intensity(intensity_norm):
    intensity_min, intensity_max = 8.0, 238.0
    return intensity_norm * (intensity_max - intensity_min) + intensity_min
panorama = cv2.imread(image_path)
if panorama is None:
    print(f"Error: Could not load image from path {image_path}")
    exit()
print(f"Panorama dimensions: {panorama.shape}")
crop_width = 300
crop_height = 1500
num_sections = 5
image_height, image_width = panorama.shape[:2]
x_positions = [(image_width - num_sections * crop_width) // 2 + i * crop_width for i in range(num_sections)]
sections = []
for i, x_start in enumerate(x_positions):
    section = panorama[image_height // 2 - crop_height // 2:image_height // 2 + crop_height // 2, x_start:x_start + crop_width]
    if section.size == 0:
        print(f"Warning: Section {i} is empty.")
        continue
    sections.append(section)
predictions = []
for idx, section in enumerate(sections):
    section_resized = cv2.resize(section, (500, 300)) / 255.0
    section_input = np.expand_dims(section_resized, axis=0)  
    intensity_norm = model.predict(section_input)[0][0]  
    predictions.append(intensity_norm)
    print(f"Section {idx} - Predicted Intensity: {intensity_norm:.2f}")
fig, axes = plt.subplots(1, len(sections), figsize=(20, 5))
fig.suptitle('Predicted Intensity for Each Column', fontsize=16)
for i, (section, pred_int) in enumerate(zip(sections, predictions)):
    ax = axes[i]
    ax.imshow(cv2.cvtColor(section, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(f"Predicted Intensity={pred_int:.2f}", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_image, dpi=300)
plt.show()
