import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
performance_min, performance_max = 720.0, 4760.0
intensity_min, intensity_max = 8.0, 238.0
current_min, current_max = 40.0, 200.0
voltage_min, voltage_max = 18.0, 23.8
speed_min, speed_max = 20.0, 90.0
def denormalize(predicted_values, min_value, max_value):
    return predicted_values * (max_value - min_value) + min_value
def mae_current(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred[:, 0] - y_true[:, 0]))
def mae_voltage(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred[:, 1] - y_true[:, 1]))
def mae_speed(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred[:, 2] - y_true[:, 2]))
def custom_loss(y_true, y_pred):
    performance_pred, intensity_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    performance_true, intensity_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    mse_performance = tf.reduce_mean(tf.square(performance_pred - performance_true), name="mse_performance")
    mse_intensity = tf.reduce_mean(tf.square(intensity_pred - intensity_true), name="mse_intensity")
    return mse_performance + mse_intensity
def mse_performance(y_true, y_pred):
    performance_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    performance_true, _ = tf.split(y_true, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.square(performance_pred - performance_true))
def mse_intensity(y_true, y_pred):
    _, intensity_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    _, intensity_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.square(intensity_pred - intensity_true))
welding_model = tf.keras.models.load_model(
    'welding_model_performance_intensity.keras',
    custom_objects={
        'custom_loss': custom_loss,
        'mse_performance': mse_performance,
        'mse_intensity': mse_intensity
    }
)
feature_model = tf.keras.models.load_model(
    'welding_feature_model_complex.keras',
    custom_objects={
        'mae_current': mae_current,
        'mae_voltage': mae_voltage,
        'mae_speed': mae_speed
    }
)
def extract_features(image):
    image_resized = cv2.resize(image, (250, 150))
    grayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    contrast = np.std(grayscale)
    avg_color = np.mean(image_resized, axis=(0, 1))
    edges = np.sum(cv2.Canny(grayscale, 100, 200) > 0)
    lines = cv2.HoughLines(cv2.Canny(grayscale, 50, 150), 1, np.pi / 180, 100)
    line_count = len(lines) if lines is not None else 0
    glcm = graycomatrix(grayscale, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    vert_symmetry = np.mean(grayscale[:, :125] - np.flip(grayscale[:, 125:], axis=1))
    horiz_symmetry = np.mean(grayscale[:75, :] - np.flip(grayscale[75:, :], axis=0))
    hog_mean = np.mean(hog(grayscale, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False))
    return [
        contrast, *avg_color, edges, line_count,
        texture_contrast, texture_homogeneity,
        vert_symmetry, horiz_symmetry, hog_mean
    ]
def predict_on_segment(image, welding_model, feature_model):
    intensity, power = welding_model.predict(np.expand_dims(image, axis=0))[0]
    features = np.array([extract_features(image)])
    refined_params = feature_model.predict(features)[0]
    current, voltage, speed = refined_params
    return intensity, power, current, voltage, speed
def process_panorama(image_path):
    image = cv2.imread(image_path)
    segments = [image[0:1500, i * 1500:(i + 1) * 1500] for i in range(5)]
    results = []
    for segment in segments:
        segment_resized = cv2.resize(segment, (250, 150))  
        intensity, power, current, voltage, speed = predict_on_segment(segment_resized, welding_model, feature_model)
        intensity = denormalize(intensity, intensity_min, intensity_max)
        power = denormalize(power, performance_min, performance_max)
        current = denormalize(current, current_min, current_max)
        voltage = denormalize(voltage, voltage_min, voltage_max)
        speed = denormalize(speed, speed_min, speed_max)
        results.append((intensity, power, current, voltage, speed))
    return results
image_path = 'panorama.jpg'
results = process_panorama(image_path)
for i, (intensity, power, current, voltage, speed) in enumerate(results, 1):
    print(f"Segment {i}: Intensity={intensity}, Power={power}, Current={current}, Voltage={voltage}, Speed={speed}")
