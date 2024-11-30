import os
import json
from collections import defaultdict
base_dir = 'images'
combinations = defaultdict(set)
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                current = data.get('Current')
                voltage = data.get('Voltage')
                speed = data.get('Speed')
                if current is None or voltage is None or speed is None:
                    print(f"Hiányzó paraméterek a következő fájlban: {json_path}")
                    continue
                combinations[('Current', 'Voltage', current, voltage)].add(speed)
                combinations[('Current', 'Speed', current, speed)].add(voltage)
                combinations[('Voltage', 'Speed', voltage, speed)].add(current)
print("Eredmények:")
for key, varying_values in combinations.items():
    param_fixed1, param_fixed2, fixed1_val, fixed2_val = key
    num_varying_values = len(varying_values)
    if num_varying_values > 1:
        varying_param_values = ', '.join(map(str, sorted(varying_values)))
        print(f"Ha {param_fixed1} = {fixed1_val} és {param_fixed2} = {fixed2_val}, "
              f"akkor a harmadik paraméter {num_varying_values} különböző értéket vesz fel: {varying_param_values}.")
