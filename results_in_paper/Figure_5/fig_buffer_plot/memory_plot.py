import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd

buffer_name = "reservoir"
# buffer_name = "gss"


with open(f'./processed_data/8_{buffer_name}_buffer_memory', 'rb') as f:
    loaded_file = pickle.load(f)

# print((loaded_file[1]))

# create a figure with figsize
fig = plt.figure(figsize=(10, 2))


replay_buffer = []

for i in range(len(loaded_file)):
    replay_buffer.append(loaded_file[i])

replay_buffer = np.array(replay_buffer)
print(replay_buffer.shape)




# rank the replay buffer for each row
replay_buffer_ranked = np.zeros(replay_buffer.shape)
for i in tqdm(range(replay_buffer.shape[0]), desc="Ranking rows"):
    buffer_line = replay_buffer[i]
    replay_buffer_ranked[i] = buffer_line[np.argsort(buffer_line)]


# build a dict 
my_dict = {}
frequency_all = []
down_sample = 100
all_num = replay_buffer.shape[0]

for i in tqdm(range(0,all_num, down_sample), desc="Plotting histograms"):
    # first calculate the frequency of each number in the row
    frequency = np.zeros(9)
    for j in range(9):
        frequency[j] = np.sum(replay_buffer_ranked[i]==j)
    frequency_all.append(frequency)
    
# convert the list to a numpy array
frequency_all = np.array(frequency_all)
print(frequency_all.shape)

bar_num = frequency_all.shape[0]
for i in range(bar_num):
    my_dict[i] = frequency_all[i]

# plot the my_dict with sns.barplot

df = pd.DataFrame(my_dict)
# df.columns 

print(df)
df.columns = [str(i) for i in range(bar_num)]




fontsize = 13
bar_width = 1.0

# Original colors
colors = ['#c8c8c8', '#b3d4e6', '#a0a0a0', '#99c8e6', '#e6c7b3', '#f5b3b3', '#b3d4b3', '#d6b3d9']

# Add transparency (alpha = 80, which is ~50% opacity)
transparent_colors = [color + 'FF' for color in colors]

for i in range(1, 9):
    if i > 1:
        plt.bar(df.columns, df.loc[i], color = colors[i-1], width=bar_width, bottom=df.loc[:i-1].sum())
    else:
        plt.bar(df.columns, df.loc[i], color = colors[i-1], width=bar_width)
  
# 设置轴标签
if buffer_name == 'gss':
    plt.legend([f'task {i+1}' for i in range(8)], fontsize=fontsize-5, ncol=2)



plt.xlabel('training steps', fontsize = fontsize)
plt.ylabel('memory samples', fontsize = fontsize)
plt.xlim(0, bar_num)

if buffer_name == 'gss':
    plt.title('Gradient-based diversity sampling strategy')
else:
    plt.title('Reservoir sampling strategy')



# add several xticks from 1 to all_num
x_ticks_positions = np.arange(0, bar_num, 8)  
x_tick_labels = 100*np.arange(0, len(x_ticks_positions))  


x_ticks_display_positions = x_ticks_positions[::50]  
x_ticks_display_labels = x_tick_labels[::50]  

plt.xticks(x_ticks_display_positions, x_ticks_display_labels, fontsize = fontsize)
plt.yticks(fontsize = fontsize)


fig_path = f'./plot_figures/buffer_vis_{buffer_name}_ds_{down_sample}.pdf'
plt.tight_layout()
# plt.savefig(fig_path, bbox_inches = 'tight')
plt.show()



# ==================== Export Bar Plot Data to Excel ====================
# Create DataFrame with proper labels
export_df = pd.DataFrame({
    'Training Step': [i*down_sample for i in range(bar_num)],
    **{f'Task {i} Count': frequency_all[:, i] for i in range(1, 9)}
})

# Calculate percentages for each task
for i in range(1, 9):
    export_df[f'Task {i} %'] = export_df[f'Task {i} Count'] / export_df[[f'Task {j} Count' for j in range(1, 9)]].sum(axis=1) * 100

# Export to Excel with multiple sheets
excel_filename = f'buffer_memory_distribution_{buffer_name}.xlsx'
with pd.ExcelWriter(excel_filename) as writer:
    # Raw counts
    export_df[[col for col in export_df.columns if 'Count' in col or 'Training Step' in col]].to_excel(
        writer, sheet_name='Raw Counts', index=False)
    
    # Percentages
    export_df[[col for col in export_df.columns if '%' in col or 'Training Step' in col]].to_excel(
        writer, sheet_name='Percentages', index=False)
    
    # Combined view
    export_df.to_excel(writer, sheet_name='Combined View', index=False)

print(f"Bar plot data exported to {excel_filename}")
