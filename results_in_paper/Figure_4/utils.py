import os
import re
import numpy as np


def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    avg_fde_match = re.search(r"The averaged FDE of all tasks: ([\d\.]+) m", content)
    avg_mr_match = re.search(r"The averaged Missing Rate of all tasks: ([\d\.]+) %", content)
    fde_bt_match = re.search(r"FDE backward transfer:(-?[\d\.]+e?-?\d*)", content)
    mr_bt_match = re.search(r"Missing Rate backward transfer: (-?[\d\.]+e?-?\d*)", content)
    

    if avg_fde_match and avg_mr_match and fde_bt_match and mr_bt_match:
        avg_fde = float(avg_fde_match.group(1))
        avg_mr = float(avg_mr_match.group(1))
        fde_bt = float(fde_bt_match.group(1))
        mr_bt = float(mr_bt_match.group(1))
        return avg_fde, avg_mr, fde_bt, mr_bt
    else:
        print(f"Failed to parse metrics from {file_path}")
        return None

def collect_metrics(base_dir, num_experiments, task_num, model, buffer_size):
    avg_fdes = []
    avg_mrs = []
    fde_bts = []
    mr_bts = []
    
    for i in range(0, num_experiments+1):
        file_path = os.path.join(base_dir, str(i), "results", "logs", f"{task_num}_CL_tasks_{model}_bf_{buffer_size}.txt")
        if os.path.exists(file_path):
            metrics = parse_file(file_path)
            if metrics:
                avg_fde, avg_mr, fde_bt, mr_bt = metrics
                avg_fdes.append(avg_fde)
                avg_mrs.append(avg_mr)
                fde_bts.append(fde_bt)
                mr_bts.append(mr_bt)
        else:
            print(f"File {file_path} does not exist.")
    
    return avg_fdes, avg_mrs, fde_bts, mr_bts

def calculate_statistics(values):
    if len(values) == 0:
        return None, None
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    return mean_val, std_val




def extract_results(base_dir, num_experiments, task_num, model, buffer_size):
    tlist_fde = []
    tlist_mr = []
    for ii in range(0, task_num):#依次存放task1-tasknum的repeat数据
        tlist_fde.append([])
        tlist_mr.append([])
    
    for i in range(0, num_experiments):
        file_path = os.path.join(base_dir, str(i), "results", "logs", f"{task_num}_CL_tasks_{model}_bf_{buffer_size}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()

            pattern = re.compile(r'minFDE: ([\d.]+)m\nMR: ([\d.]+)%')
            matches = pattern.findall(content)

            for j in range(0, task_num):
                tlist_fde[j].append(float(matches[j][0]))
                tlist_mr[j].append(float(matches[j][1]))
    
    minFDE_values = []
    minFDE_se = []
    MR_values = []
    MR_se = []
    for jj in range(0,task_num):
        tmp_fde_mean, tmp_fde_se = calculate_statistics(tlist_fde[jj])
        tmp_mr_mean, tmp_mr_se = calculate_statistics(tlist_mr[jj])
        minFDE_values.append(tmp_fde_mean)
        minFDE_se.append(tmp_fde_se)
        MR_values.append(tmp_mr_mean)
        MR_se.append(tmp_mr_se)


    return minFDE_values, MR_values, minFDE_se, MR_se
