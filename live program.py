import os
import time

import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops, hog
from flask import Flask, jsonify, send_file
import threading
from io import BytesIO

app = Flask(__name__)

knn_cluster_boundaries = {
    0: (0, 30),
    1: (30, 60),
    2: (60, 90),
    3: (90, 130),
    4: (130, 170),
    5: (170, float('inf'))
}

rf_models_dir = 'intensity_models'
cnn_models_dir = 'cnn_intensity_models'
knn_model_path = 'knn_model.pkl'
knn_model = joblib.load(knn_model_path)
oneshot_cnn_path = 'cnn_model_highres.h5'
img_size = (168, 375)
cropped_resolution = (900, 2000)
original_resolution = (3840, 2160)

oneshot_cnn_model = load_model(oneshot_cnn_path)
latest_intensity = {"intensity": 0.0}
latest_frame = None


def extract_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(grayscale)
    avg_color = np.mean(image, axis=(0, 1))
    edges = np.sum(cv2.Canny(grayscale, 100, 200) > 0)
    glcm = graycomatrix(grayscale, distances=[5], angles=[0], symmetric=True, normed=True)
    texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
    texture_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    resized_for_hog = cv2.resize(grayscale, (500, 225))
    hog_features = hog(resized_for_hog, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False)
    hog_mean = np.mean(hog_features)
    return [contrast, *avg_color, edges, texture_contrast, texture_homogeneity, hog_mean]


def knn_cluster_classification(intensity):
    for cluster, (lower, upper) in knn_cluster_boundaries.items():
        if lower <= intensity < upper:
            return cluster
    return None


def load_cluster_models(cluster_id):
    rf_model_path = os.path.join(rf_models_dir, f"intensity_model_cluster_{cluster_id}.joblib")
    cnn_model_path = os.path.join(cnn_models_dir, f"cnn_intensity_model_cluster_{cluster_id}.h5")
    rf_model = joblib.load(rf_model_path)
    cnn_model = load_model(cnn_model_path)
    return rf_model, cnn_model


def preprocess_image_for_model(image, model):
    input_shape = model.input_shape[1:]
    height, width, *channels = input_shape
    resized_image = cv2.resize(image, (width, height)) / 255.0
    if len(channels) == 1 and channels[0] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        resized_image = np.expand_dims(resized_image, axis=-1)
    return np.expand_dims(resized_image, axis=0)


def combine_method(segment):
    TTT = 10
    def reposition_and_classify(segment):
        h, w = original_resolution
        step_size_x = max(1, (w - cropped_resolution[1]) // 4)
        step_size_y = max(1, (h - cropped_resolution[0]) // 4)
        for x in range(0, w - cropped_resolution[1] + 1, step_size_x):
            for y in range(0, h - cropped_resolution[0] + 1, step_size_y):
                crop = segment[y:y + cropped_resolution[0], x:x + cropped_resolution[1]]
                if crop.size > 0:
                    img_resized = preprocess_image_for_model(crop, oneshot_cnn_model)
                    intensity = oneshot_cnn_model.predict(img_resized)[0][0]
                    knn_cluster = knn_cluster_classification(intensity)
                    yield crop, intensity, knn_cluster

    print("Extracting features for KNN model...")
    features = np.array(extract_features(segment)).reshape(1, -1)
    knn_cluster = knn_model.predict(features)[0]
    print(f"KNN model predicted cluster: {knn_cluster}")

    for crop, oneshot_intensity, oneshot_cluster in reposition_and_classify(segment):
        print(f"OneShot CNN predicted intensity: {oneshot_intensity}")
        print(f"OneShot intensity maps to cluster: {oneshot_cluster}")

        if knn_cluster != oneshot_cluster:
            print(f"Cluster mismatch: KNN ({knn_cluster}) != OneShot ({oneshot_cluster}). Retrying...")
            continue

        print("Loading cluster models...")
        rf_model, cnn_model = load_cluster_models(knn_cluster)

        print("Preprocessing image for CNN model...")
        img_resized_cnn = preprocess_image_for_model(crop, cnn_model)
        ICNN = cnn_model.predict(img_resized_cnn)[0][0]
        print(f"CNN model predicted intensity: {ICNN}")

        print("Using Random Forest model...")
        features = np.array(extract_features(crop)).reshape(1, -1)
        IRF = rf_model.predict(features)[0]
        print(f"Random Forest model predicted intensity: {IRF}")

        IOneShot = oneshot_intensity

        print("Calculating differences...")
        delta_rf_cnn = abs(IRF - ICNN)
        delta_rf_oneshot = abs(IRF - IOneShot)
        delta_cnn_oneshot = abs(ICNN - IOneShot)
        print(f"Differences - RF-CNN: {delta_rf_cnn}, RF-OneShot: {delta_rf_oneshot}, CNN-OneShot: {delta_cnn_oneshot}")

        if all(delta <= TTT for delta in [delta_rf_cnn, delta_rf_oneshot, delta_cnn_oneshot]):
            final_intensity = (IRF + ICNN + IOneShot) / 3
            print(f"Agreement between models. Final intensity: {final_intensity}")
        else:
            total_differences = {
                "RF": delta_rf_cnn + delta_rf_oneshot,
                "CNN": delta_rf_cnn + delta_cnn_oneshot,
                "OneShot": delta_rf_oneshot + delta_cnn_oneshot,
            }
            final_intensity_model = min(total_differences, key=total_differences.get)
            final_intensity = eval(final_intensity_model)
            print(f"Disagreement. Using {final_intensity_model} model's intensity: {final_intensity}")

        if max(delta_rf_cnn, delta_rf_oneshot, delta_cnn_oneshot) > 2 * TTT:
            print("Significant disagreement detected. Returning None.")
            return None

        print(f"Final intensity result: {final_intensity}")
        return final_intensity

    print("No agreement reached after repositioning attempts.")
    return None



@app.route('/get_intensity', methods=['GET'])
def get_intensity():
    return jsonify(latest_intensity)


@app.route('/get_image', methods=['GET'])
def get_image():
    global latest_frame
    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 404
    _, buffer = cv2.imencode('.jpg', latest_frame)
    return send_file(BytesIO(buffer), mimetype='image/jpeg', as_attachment=False)


def live_camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        start_x = (w - 4500) // 2
        start_y = (h - 1500) // 2
        central_crop = frame[start_y:start_y + 1500, start_x:start_x + 4500]
        intensity = combine_method(central_crop)
        if intensity is not None:
            latest_intensity["intensity"] = float(intensity)
            input()
        latest_frame = cv2.resize(frame, (640, 480))
        middle_y = frame.shape[0] // 2
        cv2.line(latest_frame, (0, middle_y), (frame.shape[1], middle_y), (0, 255, 255), 2)
        cv2.imshow('Live Camera Feed', latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_flask():
    app.run(debug=False, port=5000, use_reloader=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    live_camera_thread()
