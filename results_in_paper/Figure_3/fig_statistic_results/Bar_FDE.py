import numpy as np
import matplotlib.pyplot as plt
import re
from utils import *

base_dir = "."  # 
num_experiments = 9  # 
task_num = 8  # 


index_bf =[500, 1000, 2000, 4000]

avg_fde_bwt_clser = []
se_fde_bwt_clser = []
avg_fde_clser = []
se_fde_clser = []


avg_fde_bwt_derppgssrev = []
se_fde_bwt_derppgssrev = []
avg_fde_derppgssrev = []
se_fde_derppgssrev = []

avg_fde_bwt_agem = []
avg_fde_agem = []
se_fde_bwt_agem = []
se_fde_agem = []

avg_fde_bwt_gem = []
avg_fde_gem = []
se_fde_bwt_gem = []
se_fde_gem = []

avg_fde_bwt_der = []
avg_fde_der = []
se_fde_bwt_der = []
se_fde_der = []

avg_fde_bwt_gss = []
avg_fde_gss = []
se_fde_bwt_gss = []
se_fde_gss = []




fde_list_clser = []
mr_list_clser = []
fde_bwt_list_clser = []
mr_bwt_list_clser = []

fde_list_agem = []
mr_list_agem = []
fde_bwt_list_agem = []
mr_bwt_list_agem = []

fde_list_gem = []
mr_list_gem = []
fde_bwt_list_gem = []
mr_bwt_list_gem = []

fde_list_der = []
mr_list_der = []
fde_bwt_list_der = []
mr_bwt_list_der = []

fde_list_gss = []
mr_list_gss = []
fde_bwt_list_gss = []
mr_bwt_list_gss = []





for bf in index_bf:

    #clser
    buffer_size = int(bf/2) # derppgssrev real buffer size = buffer_size*2 (due to the dual-memory)
    model = 'clser'
    fde_clser, mr_clser, fde_bwt_clser, mr_bwt_clser = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_clser.append(fde_clser)
    mr_list_clser.append(mr_clser)
    fde_bwt_list_clser.append(fde_bwt_clser)
    mr_bwt_list_clser.append(mr_bwt_clser)

    mean_fde_bwt_clser, std_fde_bwt_clser = calculate_statistics(fde_bwt_clser)
    mean_fde_clser, std_fde_clser = calculate_statistics(fde_clser)


    avg_fde_bwt_clser.append(mean_fde_bwt_clser)
    se_fde_bwt_clser.append(std_fde_bwt_clser)
    avg_fde_clser.append(mean_fde_clser)
    se_fde_clser.append(std_fde_clser) 



    buffer_size = bf
    #A-GEM
    model = 'agem'
    fde_agem , mr_agem, fde_bwt_agem, mr_bwt_agem = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_agem.append(fde_agem)
    mr_list_agem.append(mr_agem)
    fde_bwt_list_agem.append(fde_bwt_agem)
    mr_bwt_list_agem.append(mr_bwt_agem)


    mean_fde_bwt_agem, std_fde_bwt_agem = calculate_statistics(fde_bwt_agem)
    mean_fde_agem, std_fde_agem = calculate_statistics(fde_agem)
    avg_fde_bwt_agem.append(mean_fde_bwt_agem)
    se_fde_bwt_agem.append(std_fde_bwt_agem)
    avg_fde_agem.append(mean_fde_agem)
    se_fde_agem.append(std_fde_agem) 

    #GEM
    model = 'gem'
    fde_gem , mr_gem, fde_bwt_gem, mr_bwt_gem = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_gem.append(fde_gem)
    mr_list_gem.append(mr_gem)
    fde_bwt_list_gem.append(fde_bwt_gem)
    mr_bwt_list_gem.append(mr_bwt_gem)


    mean_fde_bwt_gem, std_fde_bwt_gem = calculate_statistics(fde_bwt_gem)
    mean_fde_gem, std_fde_gem = calculate_statistics(fde_gem)
    avg_fde_bwt_gem.append(mean_fde_bwt_gem)
    se_fde_bwt_gem.append(std_fde_bwt_gem)
    avg_fde_gem.append(mean_fde_gem)
    se_fde_gem.append(std_fde_gem) 

    #DER
    model = 'der'
    fde_der, mr_der, fde_bwt_der, mr_bwt_der = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_der.append(fde_der)
    mr_list_der.append(mr_der)
    fde_bwt_list_der.append(fde_bwt_der)
    mr_bwt_list_der.append(mr_bwt_der)

    mean_fde_bwt_der, std_fde_bwt_der = calculate_statistics(fde_bwt_der)
    mean_fde_der, std_fde_der = calculate_statistics(fde_der)
    avg_fde_bwt_der.append(mean_fde_bwt_der)
    se_fde_bwt_der.append(std_fde_bwt_der)
    avg_fde_der.append(mean_fde_der)
    se_fde_der.append(std_fde_der) 



    #GSS
    model = 'gss'
    fde_gss, mr_gss, fde_bwt_gss, mr_bwt_gss = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_gss.append(fde_gss)
    mr_list_gss.append(mr_gss)
    fde_bwt_list_gss.append(fde_bwt_gss)
    mr_bwt_list_gss.append(mr_bwt_gss)


    mean_fde_bwt_gss, std_fde_bwt_gss = calculate_statistics(fde_bwt_gss)
    mean_fde_gss, std_fde_gss = calculate_statistics(fde_gss)
    avg_fde_bwt_gss.append(mean_fde_bwt_gss)
    se_fde_bwt_gss.append(std_fde_bwt_gss)
    avg_fde_gss.append(mean_fde_gss)
    se_fde_gss.append(std_fde_gss) 

#========================================PLOT===============================================#

# Adjust plt.rcParams only once
figsize=(6, 4)
font_size = 15
plt.rcParams['font.family'] = 'Arial'
bar_width = 0.15
offset = 1 * bar_width
index_group = [1.3, 2.3, 3.3, 4.3]
error_bar_capsize = 4.0
error_bar_elinewidth = 0.3
error_bar_capthick = 0.3

# barchart_8_tasks_FDE_BWT
fig1, ax1 = plt.subplots(figsize=figsize)

ax1.bar(index_group, avg_fde_bwt_agem, width=bar_width, label='A-GEM', zorder=2, edgecolor ='gold',  color='gold')
ax1.errorbar(index_group, avg_fde_bwt_agem, yerr=se_fde_bwt_agem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# mr_list_agem is a list with for element, each element is a group of scatter, help me plot the scatter  of for each group
for i in range(len(fde_bwt_list_agem)):
    x_axis = [index_group[i]]*len(fde_bwt_list_agem[i])
    ax1.scatter(x_axis, fde_bwt_list_agem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2) 



ax1.bar([size - offset for size in index_group], avg_fde_bwt_gem, width=bar_width, align='center', label='GEM', zorder=2, color='orange')
ax1.errorbar([size - offset for size in index_group], avg_fde_bwt_gem, yerr=se_fde_bwt_gem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_bwt_list_gem)):
    x_axis = [index_group[i]-offset]*len(fde_bwt_list_gem[i])
    ax1.scatter(x_axis, fde_bwt_list_gem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax1.bar([size - 2*offset for size in index_group], avg_fde_bwt_der, width=bar_width, align='center', label='DER', zorder=2, color='steelblue')
ax1.errorbar([size - 2*offset for size in index_group], avg_fde_bwt_der, yerr=se_fde_bwt_der, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_bwt_list_der)):
    x_axis = [index_group[i]-2*offset]*len(fde_bwt_list_der[i])
    ax1.scatter(x_axis, fde_bwt_list_der[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax1.bar([size + offset for size in index_group], avg_fde_bwt_gss, width=bar_width, align='center', label='GSS', zorder=2, color='slateblue')
ax1.errorbar([size + offset for size in index_group], avg_fde_bwt_gss, yerr=se_fde_bwt_gss, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_bwt_list_gss)):
    x_axis = [index_group[i]+offset]*len(fde_bwt_list_gss[i])
    ax1.scatter(x_axis, fde_bwt_list_gss[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax1.bar([size + 2*offset for size in index_group], avg_fde_bwt_clser, width=bar_width, align='center', label='Dual-LS', zorder=2, color='limegreen')
ax1.errorbar([size + 2*offset for size in index_group], avg_fde_bwt_clser, yerr=se_fde_bwt_clser, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(fde_bwt_list_clser)):
    x_axis = [index_group[i]+2*offset]*len(fde_bwt_list_clser[i])
    ax1.scatter(x_axis, fde_bwt_list_clser[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)



ax1.set_xlabel('Buffer Size', font='Arial', fontsize=font_size)
ax1.set_ylabel('FDE-BWT', font='Arial', fontsize=font_size)
# ax1.set_ylim(0, 0.6)
# ytick1 = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# ax1.set_yticks(ytick1)
# ax1.set_yticklabels(['0','0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40'])
ax1.tick_params(axis='y', labelsize=font_size)
ax1.set_xticks(index_group)
ax1.set_xticklabels(['500', '1,000', '2,000', '4,000'])
ax1.tick_params(axis='x', labelsize=font_size)
ax1.grid(False)  
ax1.set_ylim(-0.2, 2.8)
ax1.yaxis.grid(True, color='lightgray', linewidth=0.5)

handles, labels = ax1.get_legend_handles_labels()
fig1.legend(handles, labels,  loc='upper right', bbox_to_anchor=(0.97, 0.97), shadow=False, ncol=3)
plt.tight_layout()
# plt.savefig('./plot_pdf/bar_8_tasks_FDE_BWT.pdf')
plt.show()



# barchart_8_tasks_MR
fig2, ax2 = plt.subplots(figsize=figsize)

ax2.bar(index_group, avg_fde_agem, width=bar_width, label='A-GEM', zorder=2, color='gold')
ax2.errorbar(index_group, avg_fde_agem, yerr=se_fde_agem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_list_agem)):
    x_axis = [index_group[i]]*len(fde_list_agem[i])
    ax2.scatter(x_axis, fde_list_agem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax2.bar([size - offset for size in index_group], avg_fde_gem, width=bar_width, align='center', label='GEM', zorder=2, color='orange')
ax2.errorbar([size - offset for size in index_group], avg_fde_gem, yerr=se_fde_gem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_list_gem)):
    x_axis = [index_group[i]-offset]*len(fde_list_gem[i])
    ax2.scatter(x_axis, fde_list_gem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax2.bar([size - 2*offset for size in index_group], avg_fde_der, width=bar_width, align='center', label='DER', zorder=2, color='steelblue')
ax2.errorbar([size - 2*offset for size in index_group], avg_fde_der, yerr=se_fde_der, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(fde_list_der)):
    x_axis = [index_group[i]-2*offset]*len(fde_list_der[i])
    ax2.scatter(x_axis, fde_list_der[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax2.bar([size + offset for size in index_group], avg_fde_gss, width=bar_width, align='center', label='GSS', zorder=2, color='slateblue')
ax2.errorbar([size + offset for size in index_group], avg_fde_gss, yerr=se_fde_gss, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(fde_list_gss)):
    x_axis = [index_group[i]+offset]*len(fde_list_gss[i])
    ax2.scatter(x_axis, fde_list_gss[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)



ax2.bar([size + 2*offset for size in index_group], avg_fde_clser, width=bar_width, align='center', label='Dual-LS', zorder=2, color='limegreen')
ax2.errorbar([size + 2*offset for size in index_group], avg_fde_clser, yerr=se_fde_clser, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(fde_list_clser)):
    x_axis = [index_group[i]+2*offset]*len(fde_list_clser[i])
    ax2.scatter(x_axis, fde_list_clser[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax2.set_xlabel('Buffer Size', font='Arial', fontsize=font_size)
ax2.set_ylabel(r'$\text{FDE}_{\text{Ave}} $(m)', fontname='Arial', fontsize=font_size)
ax2.set_ylim(0, 3.5)
# ytick2 = [0, 10, 20, 30, 40, 50, 60, 70]
# ax2.set_yticks(ytick2)
ax2.tick_params(axis='y', labelsize=font_size)
ax2.set_xticks(index_group)
ax2.set_xticklabels(['500', '1,000', '2,000', '4,000'])
ax2.tick_params(axis='x', labelsize=font_size)
ax2.grid(False)
ax2.yaxis.grid(True, color='lightgray', linewidth=0.5, zorder=1)

handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels,  loc='upper right', bbox_to_anchor=(0.97, 0.95), shadow=False, ncol=3)
plt.tight_layout()
# plt.savefig('./plot_pdf/bar_8_tasks_FDE.pdf')
plt.show()

import pandas as pd
from pandas import ExcelWriter



# ============================ Excel（Sheets by Buffer Sizes） ============================
with ExcelWriter('fde_bars_results.xlsx') as writer:
    for i, bf in enumerate(index_bf):
        fde_bwt_data = {
            'Model': ['A-GEM', 'GEM', 'DER', 'GSS', 'Dual-LS'],
            'Average': [
                avg_fde_bwt_agem[i],
                avg_fde_bwt_gem[i],
                avg_fde_bwt_der[i],
                avg_fde_bwt_gss[i],
                avg_fde_bwt_clser[i]
            ],
            'Std_Error': [
                se_fde_bwt_agem[i],
                se_fde_bwt_gem[i],
                se_fde_bwt_der[i],
                se_fde_bwt_gss[i],
                se_fde_bwt_clser[i]
            ],
            'Raw_Data': [
                fde_bwt_list_agem[i],
                fde_bwt_list_gem[i],
                fde_bwt_list_der[i],
                fde_bwt_list_gss[i],
                fde_bwt_list_clser[i]
            ]
        }
        
        fde_data = {
            'Model': ['A-GEM', 'GEM', 'DER', 'GSS', 'Dual-LS'],
            'Average': [
                avg_fde_agem[i],
                avg_fde_gem[i],
                avg_fde_der[i],
                avg_fde_gss[i],
                avg_fde_clser[i]
            ],
            'Std_Error': [
                se_fde_agem[i],
                se_fde_gem[i],
                se_fde_der[i],
                se_fde_gss[i],
                se_fde_clser[i]
            ],
            'Raw_Data': [
                fde_list_agem[i],
                fde_list_gem[i],
                fde_list_der[i],
                fde_list_gss[i],
                fde_list_clser[i]
            ]
        }
        
        fde_bwt_df = pd.DataFrame(fde_bwt_data)
        fde_df = pd.DataFrame(fde_data)
        
        sheet_name = f"BufferSize_{bf}"
        fde_bwt_df.to_excel(writer, sheet_name=f"{sheet_name}_FDE_BWT", index=False)
        fde_df.to_excel(writer, sheet_name=f"{sheet_name}_FDE", index=False)
