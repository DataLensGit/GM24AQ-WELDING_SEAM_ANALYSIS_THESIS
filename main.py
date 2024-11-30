import os
import json
import pandas as pd
import matplotlib.pyplot as plt
base_dir = "Hegesztési varratképek"
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    if os.path.isdir(seam_path):
        json_file = os.path.join(seam_path, 'archive_info.json')
        with open(json_file) as f:
            info = json.load(f)
        if info['Current'] == 500 and info['Voltage'] == 500 and info['Speed'] == 500:
            continue  
        if 'performance' not in info or 'energy_intensity' not in info:
            performance = info['Voltage'] * info['Current']  
            energy_intensity = performance / info['Speed']  
            info['performance'] = performance
            info['energy_intensity'] = energy_intensity
            with open(json_file, 'w') as f:
                json.dump(info, f, indent=4)
data = []
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    if os.path.isdir(seam_path):
        json_file = os.path.join(seam_path, 'archive_info.json')
        with open(json_file) as f:
            info = json.load(f)
        if info['Current'] == 500 and info['Voltage'] == 500 and info['Speed'] == 500:
            continue  
        data.append({
            'Seam': seam_folder,
            'Current (A)': info['Current'],
            'Voltage (V)': info['Voltage'],
            'Speed (mm/s)': info['Speed'],
            'Performance (W)': info.get('performance', 0),  
            'Energy Intensity (W/mm/s)': info.get('energy_intensity', 0)  
        })
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
plt.bar(df['Seam'], df['Performance (W)'], color='blue')
plt.title('Welding Seam Performance')
plt.xlabel('Welding Seam')
plt.ylabel('Performance (W)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.bar(df['Seam'], df['Energy Intensity (W/mm/s)'], color='orange')
plt.title('Welding Seam Energy Intensity')
plt.xlabel('Welding Seam')
plt.ylabel('Energy Intensity (W/mm/s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
