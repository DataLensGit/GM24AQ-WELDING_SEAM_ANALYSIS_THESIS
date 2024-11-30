import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(0)
data = pd.DataFrame({
    'Current': np.random.uniform(50, 200, 100),
    'Voltage': np.random.uniform(18, 25, 100),
    'Speed': np.random.uniform(10, 100, 100)
})
data['Intensity'] = data['Current'] * data['Voltage'] / data['Speed']  
data['Performance'] = data['Current'] * data['Voltage']  
data['Cluster'] = np.random.randint(1, 4, data.shape[0])  
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Current'], data['Voltage'], data['Speed'],
                     c=data['Intensity'], cmap='viridis', alpha=0.8)
ax.set_xlabel('Current')
ax.set_ylabel('Voltage')
ax.set_zlabel('Speed')
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Intensity')
plt.title("3D Scatter Plot of Current, Voltage, and Speed with Intensity as Color")
plt.show()
fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=False)  
for idx, param in enumerate(['Current', 'Voltage', 'Speed', 'Intensity', 'Performance']):
    sns.boxplot(x='Cluster', y=param, data=data, ax=axes[idx], palette="Set2")
    axes[idx].set_title(f'{param}')
plt.suptitle("Box Plots of Parameters by Cluster", fontsize=16)
plt.show()
sns.pairplot(data[['Current', 'Voltage', 'Speed', 'Intensity', 'Performance']], kind='kde', height=2)
plt.suptitle("Pairwise KDE Plots of Parameters", fontsize=16, y=1.02)
plt.show()
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x='Current', y='Voltage', fill=True, cmap='viridis', thresh=0.05)
plt.scatter(data['Current'], data['Voltage'], c=data['Intensity'], cmap='viridis', edgecolor='k', s=30, alpha=0.6)
plt.colorbar(label='Intensity')
plt.title("2D Density Plot of Current vs Voltage with Intensity as Color")
plt.xlabel("Current")
plt.ylabel("Voltage")
plt.show()
