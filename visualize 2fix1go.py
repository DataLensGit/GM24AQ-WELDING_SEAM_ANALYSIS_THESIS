import os
import json
import cv2
import matplotlib.pyplot as plt
base_dir = 'images'  
thumb_size = (150, 150)  
output_image = "parameter_influence_report.png"  
def create_thumbnail(image, thumb_size):
    return cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
def collect_variations(base_dir, max_samples=5):
    variation_data = {
        'Vary_Speed': [],  
        'Vary_Current': [],  
        'Vary_Voltage': []  
    }
    used_folders = set()  
    for seam_folder in os.listdir(base_dir):
        if seam_folder in used_folders:
            continue  
        seam_path = os.path.join(base_dir, seam_folder)
        json_file = os.path.join(seam_path, 'archive_info.json')
        image_file = os.path.join(seam_path, 'panorama.jpg')
        if not os.path.exists(json_file) or not os.path.exists(image_file):
            continue
        with open(json_file) as f:
            info = json.load(f)
            current = info.get("Current")
            voltage = info.get("Voltage")
            speed = info.get("Speed")
        image = cv2.imread(image_file)
        if image is None:
            continue
        thumbnail = create_thumbnail(image, thumb_size)
        if current == 102.0 and voltage == 20.1:
            variation_data['Vary_Speed'].append((thumbnail, speed, f"Voltage={voltage}, Current={current}"))
            used_folders.add(seam_folder)
        elif current == 108.0 and voltage == 20.3:
            variation_data['Vary_Speed'].append((thumbnail, speed, f"Voltage={voltage}, Current={current}"))
            used_folders.add(seam_folder)
        elif current == 114.0 and voltage == 20.6:
            variation_data['Vary_Speed'].append((thumbnail, speed, f"Voltage={voltage}, Current={current}"))
            used_folders.add(seam_folder)
    for key in variation_data:
        variation_data[key].sort(key=lambda x: x[1])  
        variation_data[key] = variation_data[key][:max_samples]  
    return variation_data
variation_data = collect_variations(base_dir, max_samples=5)
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('Influence of Each Parameter on Welding Seam', fontsize=16)
row_titles = [
    "Fixed Voltage & Current, Varying Speed",
    "Fixed Voltage & Speed, Varying Current",
    "Fixed Current & Speed, Varying Voltage"
]
for i, (key, row_title) in enumerate(zip(['Vary_Speed', 'Vary_Current', 'Vary_Voltage'], row_titles)):
    axes[i, 0].set_ylabel(row_title, fontsize=12, rotation=0, labelpad=60, ha='right')
    for j, (thumb, value, fixed_params) in enumerate(variation_data[key]):
        ax = axes[i, j]
        ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f"{fixed_params}\n{key.split('_')[1]}={value}", fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_image, dpi=300)
plt.show()
