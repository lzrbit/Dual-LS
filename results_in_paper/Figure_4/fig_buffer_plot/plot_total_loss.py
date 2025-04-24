import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

loss_type_index = 0
file_path = f'./processed_data/loss_all_tasks_type_index_{loss_type_index}.pkl'

with open(file_path, 'rb') as file:
    loss_data = pickle.load(file)

downsample_rate = 1

# Predefined task update step counts
task_step_num = [4181, 8282, 425, 1925, 988, 1142, 10205, 4439]

# Divide loss data by tasks
loss_divided_1 = loss_data[:task_step_num[0]]
loss_divided_2 = loss_data[task_step_num[0]:sum(task_step_num[:2])]
loss_divided_3 = loss_data[sum(task_step_num[:2]):sum(task_step_num[:3])]
loss_divided_4 = loss_data[sum(task_step_num[:3]):sum(task_step_num[:4])]
loss_divided_5 = loss_data[sum(task_step_num[:4]):sum(task_step_num[:5])]
loss_divided_6 = loss_data[sum(task_step_num[:5]):sum(task_step_num[:6])]
loss_divided_7 = loss_data[sum(task_step_num[:6]):sum(task_step_num[:7])]
loss_divided_8 = loss_data[sum(task_step_num[:7]):sum(task_step_num[:8])]

# Combine downsampled data
ds_list = []
for kk in range(1, 9):
    ds_list.extend(eval(f'loss_divided_{kk}')[::downsample_rate])

# Create x-axis data
x = range(1, len(ds_list)+1)

# Smoothing
window_size = 20
if len(ds_list) >= window_size:
    smoothed_ds_list = np.convolve(ds_list, np.ones(window_size)/window_size, mode='valid')
else:
    smoothed_ds_list = np.array([])  # Handle case where window_size > len(ds_list)

# Plotting
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 20})

if len(smoothed_ds_list) > 0:
    plt.plot(x[window_size-1:], smoothed_ds_list, label='Smoothed Loss', color='midnightblue', linewidth=1)

plt.xlabel('training step (batch number)')
plt.ylabel('loss')
plt.xlim(1, len(ds_list))
plt.ylim(0, 600)

# Calculate task boundaries
boudary_id = [0]
for i in range(8):
    boudary_id.append(boudary_id[-1] + task_step_num[i])

intervals = [(boudary_id[i]+1, boudary_id[i+1]) for i in range(8)]
print(intervals)

# Background colors for tasks
colors = ['#f1f1f1', '#e6f7ff', '#d3d3d3', '#d0eaff', '#fff3e6',
          '#fbe2e2', '#e6f7e6', '#f5e6f9']
alpha = 1

for (start, end), color in zip(intervals, colors):
    plt.axvspan(start, end, color=color, alpha=alpha, linewidth=0)

plt.grid(True)
plt.tight_layout()
plt.show()

# Export to Excel
if len(smoothed_ds_list) > 0:
    smoothed_data = np.concatenate([np.full(window_size-1, np.nan), smoothed_ds_list])
else:
    smoothed_data = np.full(len(ds_list), np.nan)

# Create task ID column
task_id = np.empty(len(x), dtype=object)
for i, (start, end) in enumerate(intervals, 1):
    task_id[(np.array(x) >= start) & (np.array(x) <= end)] = i

data_dict = {
    'Training Step': x,
    'Raw Loss': ds_list,
    'Smoothed Loss (window=20)': smoothed_data[:len(x)],  # Ensure same length
    'Task ID': task_id  # Single column with task IDs 1-8
}

df = pd.DataFrame(data_dict)

# Metadata
metadata = {
    'Description': [
        f'Loss data for type_index={loss_type_index}',
        f'Downsample rate: {downsample_rate}',
        f'Smoothed with window size {window_size}',
        'Task intervals:',
        *[f'Task {i}: steps {intervals[i-1][0]}-{intervals[i-1][1]}' for i in range(1,9)]
    ]
}

# Save to Excel
with pd.ExcelWriter(f'raw_data_of_total_loss.xlsx') as writer:
    df.to_excel(writer, sheet_name='Loss Data', index=False)
    pd.DataFrame(metadata).to_excel(
        writer, 
        sheet_name='Metadata',
        index=False,
        header=False
    )

print(f"Data saved to loss_data_type_{loss_type_index}.xlsx")