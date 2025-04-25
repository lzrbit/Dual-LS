import pickle
import matplotlib.pyplot as plt
import numpy as np

model_name = 'plastic' 
file_path = f'./processed_data/model_update_random_value_all_tasks_{model_name}.pkl'
if model_name == 'plastic':
    threshold = 0.9  
else:
    threshold = 0.7

# 读取文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

downsample_rate = 100
data = data[::downsample_rate]  # Apply downsampling

steps = np.arange(0, len(data))

above_threshold = [i for i, val in zip(steps, data) if val > threshold]
below_threshold = [i for i, val in zip(steps, data) if val <= threshold]
above_vals = [val for val in data if val > threshold]
below_vals = [val for val in data if val <= threshold]

above_threshold_original = [i * downsample_rate for i in above_threshold]
below_threshold_original = [i * downsample_rate for i in below_threshold]

plt.figure(figsize=(7.16, 3))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 15})

plt.scatter(above_threshold_original, above_vals, color='red', label='Above Threshold', s=1.2)

plt.scatter(below_threshold_original, below_vals, color='gray', label='Below or Equal Threshold', s=0.5)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
if model_name == 'plastic':
    plt.text(0.1, 0.8, 'Threshold=0.9', horizontalalignment='center',color='red',  verticalalignment='center', transform=plt.gca().transAxes)
else:
    plt.text(0.1, 0.6, 'Threshold=0.7', horizontalalignment='center',color='red',  verticalalignment='center', transform=plt.gca().transAxes)

plt.xlabel('training step (batch number)')
plt.ylabel('random values')
plt.xlim(0, len(data) * downsample_rate)  # Set the xlim to the full range of the original data

# Adjust x-ticks to match the downsampled data
xticks = np.arange(0, len(data) * downsample_rate, downsample_rate*50)
plt.xticks(xticks, [str(x) for x in xticks])

# plt.grid(True)
plt.tight_layout()
# plt.savefig(f'./plot_figures/model_update_{model_name}_ds_{downsample_rate}.pdf')

plt.show()
