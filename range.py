import os
import json
base_dir = 'images'
ranges = {
    'Current': {'min': float('inf'), 'max': float('-inf')},
    'Voltage': {'min': float('inf'), 'max': float('-inf')},
    'Speed': {'min': float('inf'), 'max': float('-inf')},
    'Performance': {'min': float('inf'), 'max': float('-inf')},
    'Intensity': {'min': float('inf'), 'max': float('-inf')}
}
for seam_folder in os.listdir(base_dir):
    seam_path = os.path.join(base_dir, seam_folder)
    json_file = os.path.join(seam_path, 'archive_info.json')
    if not os.path.exists(json_file):
        print(f"Warning: Missing JSON file in {seam_path}")
        continue
    with open(json_file) as f:
        data = json.load(f)
        current = data['Current']
        voltage = data['Voltage']
        speed = data['Speed']
        performance = current * voltage
        intensity = performance / (speed + 1e-6)  
        for param, value in zip(['Current', 'Voltage', 'Speed', 'Performance', 'Intensity'],
                                [current, voltage, speed, performance, intensity]):
            if value < ranges[param]['min']:
                ranges[param]['min'] = value
            if value > ranges[param]['max']:
                ranges[param]['max'] = value
for param, range_vals in ranges.items():
    print(f"{param}: min = {range_vals['min']}, max = {range_vals['max']}")
