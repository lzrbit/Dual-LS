import os

import matplotlib.pyplot as plt
import numpy as np
from utils import *




def plot_metrics_bar_chart(group_index, experiment_index, metric_key, metric_name, figsize=(8, 6), font_size=12):

   
    
    metrics = []
    selected_folders = [f"result_{i}" for i in experiment_index]
    for folder in selected_folders:
        file_path = os.path.join(folder, "log", "8_CL_tasks_clser_bf_1000.txt")
        if os.path.exists(file_path):
            metrics_data = extract_metrics(file_path) 
            metrics.append(metrics_data.get(metric_key, np.nan))  
        else:
            metrics.append(np.nan)
            print(f"File not found: {file_path}")
    

    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(selected_folders))  
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    bar_width = 0.5  
    bars = ax.bar(x, metrics, color='white', 
              edgecolor='black', 
              width=bar_width)
    
    for i, b in enumerate(bars):
        b.set_facecolor(colors[i % len(colors)]) 
        b.set_hatch('/')                        
    
    # add value annotation
    for bar in bars:
        yval = bar.get_height()
        if yval >= 0:
            va = 'bottom'
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va=va, fontsize=font_size)
        else:
            # plot at the y=0
            va = 'bottom'
            ax.text(bar.get_x() + bar.get_width()/2, 0, round(yval, 2), ha='center', va=va, fontsize=font_size)
        
        

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['(1,0.5)', '(0.5,1)', '(1,1)'], fontsize=font_size-2) 

    ax.set_ylabel(metric_name, fontsize=font_size)
    ax.set_ylim(-0.2, 0.25)
    

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    output_file = f"{metric_key}_bar_{group_index}.pdf"
    plt.savefig('PDF_files/'+output_file, format='pdf',dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_file}")
    # plt.show()



if __name__ == "__main__":

    metric_key = 'FDE_BWT' 
    metric_name = 'FDE-BWT/m'  

    
    
    for group_index in range(1, 10):  # 假设有9组，每组包含3个实验
        experiment_index = [3 * (group_index - 1) + i for i in range(1, 4)]
        plot_metrics_bar_chart(group_index, experiment_index, metric_key, metric_name, figsize=(4, 3), font_size=18)

    print("All charts saved successfully!")
