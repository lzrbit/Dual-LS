import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


model_name = 'reservoir'  
# model_name = 'gss'  

fontsize = 18
file_path = f'./processed_data/buffer_task_id_all_tasks_{model_name}.pkl'


with open(file_path, 'rb') as file:
    data = pickle.load(file)

downsampling_rate = 100
data = data[::downsampling_rate]  # Downsample the data


num_steps = len(data)


stats_matrix = np.zeros((8, num_steps), dtype=int)

for j, batch in enumerate(data):
    for task_id in batch:
        stats_matrix[task_id - 1, j] += 1

print(stats_matrix.shape)


plt.figure(figsize=(20, 4))  
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})

if model_name == 'gss':
    cmap = sns.color_palette("Reds", as_cmap=True) 
else:
    cmap = sns.color_palette("Blues", as_cmap=True) 

ax = sns.heatmap(stats_matrix, cmap=cmap, cbar_kws={'label': 'count number'}, annot=False, fmt="d", 
                 linewidths=0.1, linecolor='white')  # General linecolor for the grid

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fontsize)


# for _, spine in ax.spines.items():  # Iterate over the grid line (spines)
#     spine.set_visible(True)
#     spine.set_alpha(0.1)  # Set transparency for grid lines (0 = fully transparent, 1 = fully opaque)

# Ensure spines are visible and adjust the color and linewidth
for spine in ax.spines.values():
    spine.set_visible(True)  # Make sure the spines are visible
    spine.set_linewidth(1.5)  # Set linewidth to 1.5
    spine.set_color('black')  # Set the color of the spines to black
    # set the font size of color bar
    



if model_name == 'gss':
    plt.title('Gradient-based diversity sampling strategy')
else:
    plt.title('Reservoir sampling strategy')

plt.xlabel('Training step (batch number)', fontsize = fontsize)
plt.ylabel('Task ID', fontsize = fontsize)
plt.gca().invert_yaxis()  # Invert Y-axis to have Task ID 1 at the top

ax.set_yticks(np.arange(8) + 0.5) 
ax.set_yticklabels(np.arange(1, 9))  # Task IDs (1 to 8)

# Original indices: multiply downsampled index by downsampling_rate
original_x_ticks = np.arange(1, num_steps + 1,50) * downsampling_rate  # Scale by downsampling rate
ax.set_xticks(np.arange(1, num_steps + 1,50) + 0.5)  # Set tick positions in the center of each cell
ax.set_xticklabels(original_x_ticks)  # Set the tick labels to the original indices

plt.tight_layout()
plt.show()
# save the fig as PDF
# plt.savefig(f'./plot_figures/task_id_past_{model_name}.pdf', dpi=300)







# # ==================== Export Matrix Data to Excel ====================
# import pandas as pd

# # Create DataFrame from stats_matrix with proper labels
# df = pd.DataFrame(stats_matrix)
# df.index = [f'Task {i+1}' for i in range(8)]  # Row labels (Task IDs)
# df.columns = [f'Step {(i+1)*downsampling_rate}' for i in range(num_steps)]  # Column labels (Training steps)

# # Export to Excel
# excel_filename = f'task_id_distribution_{model_name}.xlsx'
# df.to_excel(excel_filename, sheet_name='Task Distribution')

# print(f"Matrix data exported to {excel_filename}")
