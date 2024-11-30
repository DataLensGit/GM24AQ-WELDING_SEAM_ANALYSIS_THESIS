import cv2
import os
import json
import glob
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
def augment_image(image):
    flip_x = cv2.flip(image, 0)
    flip_y = cv2.flip(image, 1)
    return flip_x, flip_y
def delete_existing_augmented_images(seam_path):
    for file_path in glob.glob(os.path.join(seam_path, "*_augmented_*.jpg")):
        try:
            os.remove(file_path)
            print(f"Deleted existing augmented image: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
def create_augmented_images(base_dir):
    for seam_folder in os.listdir(base_dir):
        seam_path = os.path.join(base_dir, seam_folder)
        if os.path.isdir(seam_path):
            delete_existing_augmented_images(seam_path)
            json_file = os.path.join(seam_path, 'archive_info.json')
            image_path = os.path.join(seam_path, 'panorama.jpg')
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping...")
                continue
            with open(json_file) as f:
                info = json.load(f)
            panorama = cv2.imread(image_path)
            if panorama is None:
                print(f"Warning: Unable to load image at {image_path}. Skipping...")
                continue
            panorama = apply_clahe(panorama)
            rect_width, rect_height = 4500, 2000
            if panorama.shape[0] < rect_height or panorama.shape[1] < rect_width:
                print(f"Warning: Image dimensions are smaller than required rectangle size. Skipping...")
                continue
            start_x = (panorama.shape[1] - rect_width) // 2
            start_y = (panorama.shape[0] - rect_height) // 2
            for i in range(5):
                crop = panorama[start_y:start_y + rect_height,
                                start_x + (i * (rect_width // 5)):start_x + ((i + 1) * (rect_width // 5))]
                if crop.size == 0:
                    print(f"Warning: Crop for index {i} is empty. Skipping...")
                    continue
                flip_x, flip_y = augment_image(crop)
                cv2.imwrite(os.path.join(seam_path, f"{seam_folder}_augmented_{i}_flip_x.jpg"), flip_x)
                print(f"Image saved: {seam_folder}_augmented_{i}_flip_x.jpg")
                cv2.imwrite(os.path.join(seam_path, f"{seam_folder}_augmented_{i}_flip_y.jpg"), flip_y)
                print(f"Image saved: {seam_folder}_augmented_{i}_flip_y.jpg")
base_dir = "images"
create_augmented_images(base_dir)
