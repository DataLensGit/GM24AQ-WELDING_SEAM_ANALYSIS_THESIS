import cv2
import numpy as np
import matplotlib.pyplot as plt
def visualize_horizontal_cropped_areas(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    full_width, full_height = 3840, 2160
    crop_width, crop_height = 900, 2000  
    vertical_center = (full_height - crop_height) // 2
    start_x = 100  
    horizontal_step = (full_width - start_x - crop_width) // 4  
    for i in range(4):  
        x = start_x + i * horizontal_step
        y = vertical_center
        rect_image = image.copy()
        cv2.rectangle(rect_image, (x, y), (x + crop_width, y + crop_height), (255, 0, 0), 3)
        plt.figure(figsize=(10, 6))
        plt.imshow(rect_image)
        plt.title(f"Horizontal Crop Position {i + 1}")
        plt.show()
visualize_horizontal_cropped_areas('images/Welding_seam1_1/saved_image_3.jpg')
