import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from skimage.feature import graycomatrix, graycoprops, hog
import matplotlib.pyplot as plt
performance_min, performance_max = 720.0, 4760.0
intensity_min, intensity_max = 8.0, 238.0
def denormalize(predicted_values, min_value, max_value):
    return predicted_values  
def extract_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(grayscale)
    avg_color = np.mean(image, axis=(0, 1))
    edges = np.sum(cv2.Canny(grayscale, 100, 200) > 0)
    lines = cv2.HoughLines(cv2.Canny(grayscale, 50, 150), 1, np.pi / 180, 100)
    line_count = len(lines) if lines is not None else 0
    glcm = graycomatrix(grayscale, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    h, w = grayscale.shape
    vert_symmetry = np.mean(grayscale[:, :w // 2] - np.flip(grayscale[:, w // 2:], axis=1))
    horiz_symmetry = np.mean(grayscale[:h // 2, :] - np.flip(grayscale[h // 2:, :], axis=0))
    hog_mean = np.mean(hog(grayscale, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False))
    return [
        contrast, *avg_color, edges, line_count,
        texture_contrast, texture_homogeneity,
        vert_symmetry, horiz_symmetry, hog_mean
    ]
def build_feature_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2)  
    ])
    return model
model = build_feature_model(input_shape=11)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
def predict_performance_intensity(image, model):
    features = np.array([extract_features(image)])
    refined_params = model.predict(features)[0]
    performance = denormalize(refined_params[0], performance_min, performance_max)
    intensity = denormalize(refined_params[1], intensity_min, intensity_max)
    return performance, intensity
def process_and_display_segments(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from path {image_path}")
        return []
    rect_width, rect_height = 4500, 1500
    start_x = (image.shape[1] - rect_width) // 2
    start_y = (image.shape[0] - rect_height) // 2
    central_crop = image[start_y:start_y + rect_height, start_x:start_x + rect_width]
    segment_width = rect_width // 5
    results = []
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(5):
        x_start = start_x + i * segment_width
        segment = central_crop[:, i * segment_width:(i + 1) * segment_width]
        performance, intensity = predict_performance_intensity(segment, model)
        results.append((performance, intensity))
        axes[i].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Performance={performance:.2f}\nIntensity={intensity:.2f}")
    plt.tight_layout()
    plt.show()
    return results
image_path = 'panorama.jpg'
results = process_and_display_segments(image_path)
for i, (performance, intensity) in enumerate(results, 1):
    print(f"Segment {i}: Performance={performance}, Intensity={intensity}")
