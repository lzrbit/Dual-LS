import os
import re
import numpy as np
from utils import *





# 配置参数
base_dir = "."  # 基础目录路径
experiment_idx = 0  # 实验组索引
task_num = 5  # 任务数
model = "clser"  # 模型方法名称  derppgssrev
buffer_size = 2000  # buffer大小

# 收集所有实验组的数据
avg_fdes, avg_mrs, fde_bts, mr_bts = collect_metrics(base_dir, experiment_idx, task_num, model, buffer_size)

# 计算并输出各项指标的Mean值和standard deviation
if avg_fdes:
    mean_avg_fde, std_avg_fde = calculate_statistics(avg_fdes)
    print(f"Mean FDE of all tasks: {mean_avg_fde:.4f} m, standard deviation: {std_avg_fde:.4f}")
if avg_mrs:
    mean_avg_mr, std_avg_mr = calculate_statistics(avg_mrs)
    print(f"Mean Missing Rate of all tasks: {mean_avg_mr:.4f} %, standard deviation: {std_avg_mr:.4f}")
if fde_bts:
    mean_fde_bt, std_fde_bt = calculate_statistics(fde_bts)
    print(f"FDE backward transfer: {mean_fde_bt:.4f}, standard deviation: {std_fde_bt:.4f}")
if mr_bts:
    mean_mr_bt, std_mr_bt = calculate_statistics(mr_bts)
    print(f"Missing Rate backward transfer: {mean_mr_bt:.4f}, standard deviation: {std_mr_bt:.4f}")
