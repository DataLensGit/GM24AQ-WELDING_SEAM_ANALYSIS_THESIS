import time

from flask import Flask, render_template, request, jsonify
import serial
import cv2
import os
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

serial_port = "/dev/ttyUSB0"
baud_rate = 9600
arduino = serial.Serial(serial_port, baud_rate, timeout=1)

camera_index = 0
camera = cv2.VideoCapture(camera_index)

calibration_data_path = "calibration_data.npz"
calibration_data = np.load(calibration_data_path)
camera_matrix = calibration_data["mtx"]
dist_coeffs = calibration_data["dist"]

output_dir = "archive"
os.makedirs(output_dir, exist_ok=True)

def undistort_image(image):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def save_to_archive(seam_id, stitched_image, subimages, metadata):
    seam_dir = os.path.join(output_dir, f"Seam_{seam_id}")
    os.makedirs(seam_dir, exist_ok=True)
    stitched_path = os.path.join(seam_dir, "panorama.jpg")
    cv2.imwrite(stitched_path, stitched_image)
    for idx, img in enumerate(subimages):
        cv2.imwrite(os.path.join(seam_dir, f"subimage_{idx}.jpg"), img)
    metadata_path = os.path.join(seam_dir, "archive_info.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

@app.route('/')
def index():
    return render_template('controller.html')

@app.route('/move', methods=['POST'])
def move():
    direction = request.json.get("direction")
    steps = request.json.get("steps")
    command = f"MOVE{direction}{steps}\n"
    arduino.write(command.encode())
    return jsonify({"status": "success"})

@app.route('/home', methods=['POST'])
def home():
    arduino.write("HOME\n".encode())
    time.sleep(20)
    return jsonify({"status": "success"})

@app.route('/capture', methods=['POST'])
def capture():
    ret, frame = camera.read()
    if not ret:
        return jsonify({"error": "Camera capture failed"}), 500

    undistorted_frame = undistort_image(frame)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(output_dir, f"capture_{timestamp}.jpg")
    cv2.imwrite(img_path, undistorted_frame)
    return jsonify({"image_path": img_path})

@app.route('/stitch', methods=['POST'])
def stitch():
    image_paths = request.json.get("image_paths")
    images = [cv2.imread(path) for path in image_paths]
    stitched_image = cv2.hconcat(images)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stitched_path = os.path.join(output_dir, f"stitched_{timestamp}.jpg")
    cv2.imwrite(stitched_path, stitched_image)
    return jsonify({"stitched_image": stitched_path})
@app.route('/start_capturing', methods=['POST'])
def start_capturing():
    line_id = request.json.get("line_id")
    num_images = int(request.json.get("num_images"))
    x_start = int(request.json.get("x_start"))
    y_position = int(request.json.get("y_position"))
    spacing = int(request.json.get("spacing"))

    subimages = []
    metadata = {
        "line_id": line_id,
        "num_images": num_images,
        "x_start": x_start,
        "y_position": y_position,
        "spacing": spacing,
        "timestamps": []
    }
    home()
    arduino.write(f"MOVEX{x_start}\n".encode())
    time.sleep(5)
    arduino.write(f"MOVEY{y_position}\n".encode())
    time.sleep(5)
    for i in range(num_images):
        ret, frame = camera.read()
        if not ret:
            return jsonify({"error": f"Camera capture failed at image {i+1}"}), 500

        undistorted_frame = undistort_image(frame)
        corrected_frame = apply_clahe(undistorted_frame)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(output_dir, f"line_{line_id}_image_{i+1}_{timestamp}.jpg")
        cv2.imwrite(img_path, corrected_frame)
        subimages.append(corrected_frame)
        metadata["timestamps"].append(timestamp)
        next_x_position = x_start + (i + 1) * spacing
        arduino.write(f"MOVEX{next_x_position}\n".encode())
        time.sleep(5)

    stitched_image = cv2.hconcat(subimages)

    save_to_archive(line_id, stitched_image, subimages, metadata)

    return jsonify({"status": "success", "stitched_image": "Stitching and saving complete."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
