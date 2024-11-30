import os
import json
from flask import Flask, render_template, jsonify, request, send_file
from PIL import Image
import hashlib

app = Flask(__name__)

DEFAULT_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "images")
THUMBNAIL_FOLDER = os.path.join(os.path.dirname(__file__), "thumbnails")

if not os.path.exists(THUMBNAIL_FOLDER):
    os.makedirs(THUMBNAIL_FOLDER)


def create_thumbnail(image_path):
    relative_path = os.path.relpath(image_path, DEFAULT_IMAGE_FOLDER)
    hashed_name = hashlib.md5(relative_path.encode()).hexdigest() + ".jpg"
    thumbnail_path = os.path.join(THUMBNAIL_FOLDER, hashed_name)
    if not os.path.exists(thumbnail_path):
        with Image.open(image_path) as img:
            img.thumbnail((300, 300))
            img.save(thumbnail_path, format="JPEG")
    return thumbnail_path


@app.route("/")
def index():
    return render_template("filter.html")


@app.route("/load_images", methods=["POST"])
def load_images():
    filters = request.json.get("filters", {})
    filtered_images = []

    for root, _, files in os.walk(DEFAULT_IMAGE_FOLDER):
        if "archive_info.json" in files:
            json_path = os.path.join(root, "archive_info.json")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except:
                continue

            if all(data.get(key) is not None and data[key] >= filters.get(f"{key}_min", -float("inf"))
                   and data[key] <= filters.get(f"{key}_max", float("inf"))
                   for key in ["Current", "Voltage", "Speed"]):
                for file in files:
                    if file.lower() == "panorama.jpg":
                        full_image_path = os.path.join(root, file)
                        thumbnail_path = create_thumbnail(full_image_path)
                        filtered_images.append({
                            "thumbnail": os.path.relpath(thumbnail_path, THUMBNAIL_FOLDER),
                            "full": os.path.relpath(full_image_path, DEFAULT_IMAGE_FOLDER)
                        })

    return jsonify({"images": filtered_images})


@app.route("/thumbnail/<path:image_path>")
def get_thumbnail(image_path):
    full_path = os.path.join(THUMBNAIL_FOLDER, image_path)
    if os.path.exists(full_path):
        return send_file(full_path)
    return "Thumbnail not found", 404


@app.route("/image/<path:image_path>")
def get_image(image_path):
    full_path = os.path.join(DEFAULT_IMAGE_FOLDER, image_path)
    if os.path.exists(full_path):
        return send_file(full_path)
    return "Image not found", 404


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_IMAGE_FOLDER):
        os.makedirs(DEFAULT_IMAGE_FOLDER)

    app.run(debug=True)
