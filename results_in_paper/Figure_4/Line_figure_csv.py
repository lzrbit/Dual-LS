import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.gridspec as gridspec
import pandas as pd  # 添加pandas库用于CSV操作
from overall_statistic import*


task_num = 8
if_fde = True
base_dir = '.'
experiment_idx = 10  # index of exp group
bf = 500


# clser
buffer_size = int(bf/2) 
model = 'clser'
fde_clser, mr_clser, fde_clser_se, mr_clser_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)


buffer_size = int(bf)
#A-GEM
model = 'agem'
fde_agem, mr_agem, fde_agem_se, mr_agem_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)

#GEM
model = 'gem'
fde_gem, mr_gem, fde_gem_se, mr_gem_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)


#DER
model = 'der'
fde_der, mr_der, fde_der_se, mr_der_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)


#GSS
model = 'gss'
fde_gss, mr_gss, fde_gss_se, mr_gss_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)


#Vanilla
buffer_size = 0
model = 'vanilla'
fde_vanilla, mr_vanilla, fde_vanilla_se, mr_vanilla_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)


#Joint
base_dir = './joint_repeat/'
model = 'vanilla'
fde_joint, mr_joint, fde_joint_se, mr_joint_se = extract_results(base_dir, experiment_idx, task_num, model, buffer_size)

def add_and_substr(list_base, list_se):
    result_list_u = [i+j for i,j in zip(list_base, list_se)]
    result_list_d = [i-j for i,j in zip(list_base,list_se)]
    return result_list_u, result_list_d

fde_vanilla_upper_se, fde_vanilla_down_se = add_and_substr(fde_vanilla, fde_vanilla_se)
fde_agem_upper_se, fde_agem_down_se = add_and_substr(fde_agem, fde_agem_se)
fde_gem_upper_se, fde_gem_down_se = add_and_substr(fde_gem, fde_gem_se)
fde_joint_upper_se, fde_joint_down_se = add_and_substr(fde_joint, fde_joint_se)
fde_der_upper_se, fde_der_down_se = add_and_substr(fde_der, fde_der_se)
fde_gss_upper_se, fde_gss_down_se = add_and_substr(fde_gss, fde_gss_se)
fde_clser_upper_se, fde_clser_down_se = add_and_substr(fde_clser, fde_clser_se)


mr_vanilla_upper_se, mr_vanilla_down_se = add_and_substr(mr_vanilla, mr_vanilla_se)
mr_agem_upper_se, mr_agem_down_se = add_and_substr(mr_agem, mr_agem_se)
mr_gem_upper_se, mr_gem_down_se = add_and_substr(mr_gem, mr_gem_se)
mr_joint_upper_se, mr_joint_down_se = add_and_substr(mr_joint, mr_joint_se)
mr_der_upper_se, mr_der_down_se = add_and_substr(mr_der, mr_der_se)
mr_gss_upper_se, mr_gss_down_se = add_and_substr(mr_gss, mr_gss_se)
mr_clser_upper_se, mr_clser_down_se = add_and_substr(mr_clser, mr_clser_se)

fde_data = {
    'Task': list(range(1, task_num+1)),
    'Vanilla': fde_vanilla,
    'Vanilla_upper_se': fde_vanilla_upper_se,
    'Vanilla_down_se': fde_vanilla_down_se,
    'A-GEM': fde_agem,
    'A-GEM_upper_se': fde_agem_upper_se,
    'A-GEM_down_se': fde_agem_down_se,
    'GSS': fde_gss,
    'GSS_upper_se': fde_gss_upper_se,
    'GSS_down_se': fde_gss_down_se,
    'DER': fde_der,
    'DER_upper_se': fde_der_upper_se,
    'DER_down_se': fde_der_down_se,
    'Joint': fde_joint,
    'Joint_upper_se': fde_joint_upper_se,
    'Joint_down_se': fde_joint_down_se,
    'Dual-LS': fde_clser,
    'Dual-LS_upper_se': fde_clser_upper_se,
    'Dual-LS_down_se': fde_clser_down_se
}

# 创建DataFrame来存储MR数据
mr_data = {
    'Task': list(range(1, task_num+1)),
    'Vanilla': mr_vanilla,
    'Vanilla_upper_se': mr_vanilla_upper_se,
    'Vanilla_down_se': mr_vanilla_down_se,
    'A-GEM': mr_agem,
    'A-GEM_upper_se': mr_agem_upper_se,
    'A-GEM_down_se': mr_agem_down_se,
    'GSS': mr_gss,
    'GSS_upper_se': mr_gss_upper_se,
    'GSS_down_se': mr_gss_down_se,
    'DER': mr_der,
    'DER_upper_se': mr_der_upper_se,
    'DER_down_se': mr_der_down_se,
    'Joint': mr_joint,
    'Joint_upper_se': mr_joint_upper_se,
    'Joint_down_se': mr_joint_down_se,
    'Dual-LS': mr_clser,
    'Dual-LS_upper_se': mr_clser_upper_se,
    'Dual-LS_down_se': mr_clser_down_se
}

fde_df = pd.DataFrame(fde_data)
mr_df = pd.DataFrame(mr_data)

fde_df.to_csv('fde_results_buffer_'+str(bf)+'.csv', index=False)
mr_df.to_csv('mr_results_buffer_'+str(bf)+'.csv', index=False)

#========================================Drawing===================================================#
font_size = 15
tasks = [f"task{i}" for i in range(1, task_num+1)]
x = range(1, task_num+1)
plt.rcParams['font.family'] = 'Arial'
linewid = 0.8
figsize=(7, 5)
cmap = plt.cm.viridis
colors = [cmap(i) for i in np.linspace(0, 1, 6)]



# turn off the grid
# ax1.grid(False)
fig1, ax1 = plt.subplots(figsize=figsize)
# FDE
ax1.plot(x, fde_vanilla, marker='o', linestyle='--', label = 'Vanilla', linewidth=linewid, clip_on=False, markersize=3, color='red')
ax1.fill_between(x, fde_vanilla_upper_se, fde_vanilla_down_se,  facecolor='red', alpha=0.1)

ax1.plot(x, fde_agem, marker='^', linestyle='-', label = 'A-GEM', linewidth=linewid, clip_on=False, markersize=3, color='darkorange')
ax1.fill_between(x, fde_agem_upper_se, fde_agem_down_se,  facecolor='darkorange', alpha=0.1)

ax1.plot(x, fde_gss, marker='d', linestyle='-', label = 'GSS', linewidth=linewid, clip_on=False, markersize=3, color='darkcyan')
ax1.fill_between(x, fde_gss_upper_se, fde_gss_down_se,  facecolor='darkcyan', alpha=0.1)

ax1.plot(x, fde_der, marker='s', linestyle='-', label = 'DER', linewidth=linewid, clip_on=False, markersize=3, color='navy')
ax1.fill_between(x, fde_der_upper_se, fde_der_down_se,  facecolor='navy', alpha=0.1)


ax1.plot(x, fde_joint, marker='x', linestyle='--', label = 'Joint training', linewidth=linewid, clip_on=False, markersize=6, color='darkviolet')
ax1.fill_between(x, fde_joint_upper_se, fde_joint_down_se,  facecolor='darkviolet', alpha=0.1)


ax1.plot(x, fde_clser, marker='x', linestyle='--', label = 'Dual-LS', linewidth=linewid, clip_on=False, markersize=6, color='black')
ax1.fill_between(x, fde_clser_upper_se, fde_clser_down_se,  facecolor='darkviolet', alpha=0.1)

ax1.set_ylabel('FDE (m)', fontsize=font_size)
ax1.set_xlabel('Task', fontsize=font_size)
ax1.set_xlim(1,8)
ax1.set_ylim(0, 5)
ax1.set_xticks(x, x)
ax1.grid(True, color='lightgray', linewidth=0.5, zorder=1)
# set the font size of x and y axis
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

# ax2.grid(False)
handles, labels = plt.gca().get_legend_handles_labels()
# plt.figlegend(handles, labels, loc = 'upper right')
plt.figlegend(handles, labels, bbox_to_anchor=(0.97, 0.95), ncol=1, frameon=True, fontsize=font_size)
plt.tight_layout()
# plt.savefig('./plot_pdf/BF_4000/line_8_tasks_FDE.pdf')
plt.show()


# MR
fig2, ax2 = plt.subplots(figsize=figsize)
ax2.plot(x, mr_vanilla, marker='o', linestyle='--', label = 'Vanilla', linewidth=linewid, clip_on=False, markersize=3, color='red')
ax2.fill_between(x, mr_vanilla_upper_se, mr_vanilla_down_se,  facecolor='red', alpha=0.1)

ax2.plot(x, mr_agem, marker='^', linestyle='-', label = 'A-GEM', linewidth=linewid, clip_on=False, markersize=3, color='darkorange')
ax2.fill_between(x, mr_agem_upper_se, mr_agem_down_se,  facecolor='darkorange', alpha=0.1)

ax2.plot(x, mr_gss, marker='d', linestyle='-', label = 'GSS', linewidth=linewid, clip_on=False, markersize=3, color='darkcyan')
ax2.fill_between(x, mr_gss_upper_se, mr_gss_down_se,  facecolor='darkcyan', alpha=0.1)

ax2.plot(x, mr_der, marker='s', linestyle='-', label = 'DER', linewidth=linewid, clip_on=False, markersize=3, color='navy')
ax2.fill_between(x, mr_der_upper_se, mr_der_down_se,  facecolor='navy', alpha=0.1)


ax2.plot(x, mr_joint, marker='x', linestyle='--', label = 'Joint training', linewidth=linewid, clip_on=False, markersize=6, color='darkviolet')
ax2.fill_between(x, mr_joint_upper_se, mr_joint_down_se,  facecolor='darkviolet', alpha=0.1)

ax2.plot(x, mr_clser, marker='x', linestyle='--', label = 'Dual-LS', linewidth=linewid, clip_on=False, markersize=6, color='black')
ax2.fill_between(x, mr_clser_upper_se, mr_clser_down_se,  facecolor='darkviolet', alpha=0.1)


ax2.set_ylabel('Miss Rate (%)', fontsize=font_size)
ax2.set_xlabel('Task', fontsize=font_size)
ax2.set_xlim(1,8)
ax2.set_ylim(0, 70)
ax2.set_xticks(x, x)
ax2.grid(True, color='lightgray', linewidth=0.5, zorder=1)

# set the font size of x and y axis
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

# ax2.grid(False)
handles, labels = plt.gca().get_legend_handles_labels()
plt.figlegend(handles, labels, bbox_to_anchor=(0.97, 0.95), ncol=1, frameon=True, fontsize=font_size)
plt.tight_layout()
# plt.savefig('./plot_pdf/BF_4000/line_8_tasks_MR.pdf')
plt.show()