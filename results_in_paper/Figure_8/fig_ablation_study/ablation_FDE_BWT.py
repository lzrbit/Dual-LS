import os

import matplotlib.pyplot as plt
import numpy as np
from utils import *




def plot_metrics_bar_chart(group_index, experiment_index, metric_key, metric_name, figsize=(8, 6), font_size=12):

    """
    绘制柱状图
    :param selected_folders: 选择的文件夹列表，例如 ['result_1', 'result_2', 'result_3']
    :param metric_key: 要绘制的指标键，例如 'averaged_FDE'
    :param metric_name: 要绘制的指标名称，用于Y轴标签，例如 'Averaged FDE'
    :param params: 每个result的超参数列表，形如 ['α = 1, β = 0.5', 'α = 0.5, β = 1', 'α = 1, β = 1']
    :param figsize: 图的大小，默认为 (8, 6)
    :param font_size: 字体大小，默认为 12
    """
    
    
    metrics = []
    selected_folders = [f"result_{i}" for i in experiment_index]
    for folder in selected_folders:
        file_path = os.path.join(folder, "log", "8_CL_tasks_clser_bf_1000.txt")
        if os.path.exists(file_path):
            metrics_data = extract_metrics(file_path)  # 使用之前的extract_metrics函数
            metrics.append(metrics_data.get(metric_key, np.nan))  # 提取指定的metric_key对应的值
        else:
            metrics.append(np.nan)
            print(f"File not found: {file_path}")
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(selected_folders))  # 横轴位置
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    # 为每个柱状条指定不同的颜色
    bar_width = 0.5  # 指定柱状条的宽度
    bars = ax.bar(x, metrics, color='white',  # 用白色或其他颜色作为“底色”
              edgecolor='black',  # 设置边框颜色，方便看清 hatch
              width=bar_width)
    
    for i, b in enumerate(bars):
        b.set_facecolor(colors[i % len(colors)])  # 循环使用配色
        b.set_hatch('/')                         # 设置斜线填充
    
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
    # 设置Y轴标签和标题
    ax.set_ylabel(metric_name, fontsize=font_size)
    ax.set_ylim(-0.2, 0.25)
    
    # 显示网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    
    plt.yticks(fontsize=font_size)
    # 保存图片
    plt.tight_layout()
    output_file = f"{metric_key}_bar_{group_index}.pdf"
    plt.savefig('PDF_files/'+output_file, format='pdf',dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_file}")
    # plt.show()



if __name__ == "__main__":

    metric_key = 'FDE_BWT'  # 选择的指标键
    metric_name = 'FDE-BWT/m'  # Y轴标签名称

    
    
    for group_index in range(1, 10):  # 假设有9组，每组包含3个实验
        experiment_index = [3 * (group_index - 1) + i for i in range(1, 4)]
        plot_metrics_bar_chart(group_index, experiment_index, metric_key, metric_name, figsize=(4, 3), font_size=18)

    print("All charts saved successfully!")