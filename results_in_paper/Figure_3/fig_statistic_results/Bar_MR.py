import numpy as np
import matplotlib.pyplot as plt
import re
from utils import *

base_dir = "."  # 
num_experiments = 9  # 
task_num = 8  

# avg_fdes, avg_mrs, fde_bts, mr_bts = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

index_bf =[500, 1000, 2000, 4000]

avg_mr_bwt_clser = []
se_mr_bwt_clser = []
avg_mr_clser = []
se_mr_clser = []


avg_mr_bwt_derppgssrev = []
se_mr_bwt_derppgssrev = []
avg_mr_derppgssrev = []
se_mr_derppgssrev = []

avg_mr_bwt_agem = []
avg_mr_agem = []
se_mr_bwt_agem = []
se_mr_agem = []

avg_mr_bwt_gem = []
avg_mr_gem = []
se_mr_bwt_gem = []
se_mr_gem = []

avg_mr_bwt_der = []
avg_mr_der = []
se_mr_bwt_der = []
se_mr_der = []

avg_mr_bwt_gss = []
avg_mr_gss = []
se_mr_bwt_gss = []
se_mr_gss = []


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

    mean_mr_bwt_clser, std_mr_bwt_clser = calculate_statistics(mr_bwt_clser)
    mean_mr_clser, std_mr_clser = calculate_statistics(mr_clser)


    avg_mr_bwt_clser.append(mean_mr_bwt_clser)
    se_mr_bwt_clser.append(std_mr_bwt_clser)
    avg_mr_clser.append(mean_mr_clser)
    se_mr_clser.append(std_mr_clser) 



    buffer_size = bf
    #A-GEM
    model = 'agem'
    fde_agem , mr_agem, fde_bwt_agem, mr_bwt_agem = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_agem.append(fde_agem)
    mr_list_agem.append(mr_agem)
    fde_bwt_list_agem.append(fde_bwt_agem)
    mr_bwt_list_agem.append(mr_bwt_agem)


    mean_mr_bwt_agem, std_mr_bwt_agem = calculate_statistics(mr_bwt_agem)
    mean_mr_agem, std_mr_agem = calculate_statistics(mr_agem)
    avg_mr_bwt_agem.append(mean_mr_bwt_agem)
    se_mr_bwt_agem.append(std_mr_bwt_agem)
    avg_mr_agem.append(mean_mr_agem)
    se_mr_agem.append(std_mr_agem) 

    #GEM
    model = 'gem'
    fde_gem , mr_gem, fde_bwt_gem, mr_bwt_gem = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_gem.append(fde_gem)
    mr_list_gem.append(mr_gem)
    fde_bwt_list_gem.append(fde_bwt_gem)
    mr_bwt_list_gem.append(mr_bwt_gem)


    mean_mr_bwt_gem, std_mr_bwt_gem = calculate_statistics(mr_bwt_gem)
    mean_mr_gem, std_mr_gem = calculate_statistics(mr_gem)
    avg_mr_bwt_gem.append(mean_mr_bwt_gem)
    se_mr_bwt_gem.append(std_mr_bwt_gem)
    avg_mr_gem.append(mean_mr_gem)
    se_mr_gem.append(std_mr_gem) 

    #DER
    model = 'der'
    fde_der, mr_der, fde_bwt_der, mr_bwt_der = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_der.append(fde_der)
    mr_list_der.append(mr_der)
    fde_bwt_list_der.append(fde_bwt_der)
    mr_bwt_list_der.append(mr_bwt_der)

    mean_mr_bwt_der, std_mr_bwt_der = calculate_statistics(mr_bwt_der)
    mean_mr_der, std_mr_der = calculate_statistics(mr_der)
    avg_mr_bwt_der.append(mean_mr_bwt_der)
    se_mr_bwt_der.append(std_mr_bwt_der)
    avg_mr_der.append(mean_mr_der)
    se_mr_der.append(std_mr_der) 



    #GSS
    model = 'gss'
    fde_gss, mr_gss, fde_bwt_gss, mr_bwt_gss = collect_metrics(base_dir, num_experiments, task_num, model, buffer_size)

    fde_list_gss.append(fde_gss)
    mr_list_gss.append(mr_gss)
    fde_bwt_list_gss.append(fde_bwt_gss)
    mr_bwt_list_gss.append(mr_bwt_gss)


    mean_mr_bwt_gss, std_mr_bwt_gss = calculate_statistics(mr_bwt_gss)
    mean_mr_gss, std_mr_gss = calculate_statistics(mr_gss)
    avg_mr_bwt_gss.append(mean_mr_bwt_gss)
    se_mr_bwt_gss.append(std_mr_bwt_gss)
    avg_mr_gss.append(mean_mr_gss)
    se_mr_gss.append(std_mr_gss) 

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

# barchart_8_tasks_MR_BWT
fig1, ax1 = plt.subplots(figsize=figsize)

ax1.bar(index_group, avg_mr_bwt_agem, width=bar_width, label='A-GEM', zorder=2, edgecolor ='gold',  color='gold')
ax1.errorbar(index_group, avg_mr_bwt_agem, yerr=se_mr_bwt_agem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# mr_list_agem is a list with for element, each element is a group of scatter, help me plot the scatter  of for each group
for i in range(len(mr_bwt_list_agem)):
    x_axis = [index_group[i]]*len(mr_bwt_list_agem[i])
    ax1.scatter(x_axis, mr_bwt_list_agem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2) 



ax1.bar([size - offset for size in index_group], avg_mr_bwt_gem, width=bar_width, align='center', label='GEM', zorder=2, color='orange')
ax1.errorbar([size - offset for size in index_group], avg_mr_bwt_gem, yerr=se_mr_bwt_gem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_bwt_list_gem)):
    x_axis = [index_group[i]-offset]*len(mr_bwt_list_gem[i])
    ax1.scatter(x_axis, mr_bwt_list_gem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax1.bar([size - 2*offset for size in index_group], avg_mr_bwt_der, width=bar_width, align='center', label='DER', zorder=2, color='steelblue')
ax1.errorbar([size - 2*offset for size in index_group], avg_mr_bwt_der, yerr=se_mr_bwt_der, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_bwt_list_der)):
    x_axis = [index_group[i]-2*offset]*len(mr_bwt_list_der[i])
    ax1.scatter(x_axis, mr_bwt_list_der[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax1.bar([size + offset for size in index_group], avg_mr_bwt_gss, width=bar_width, align='center', label='GSS', zorder=2, color='slateblue')
ax1.errorbar([size + offset for size in index_group], avg_mr_bwt_gss, yerr=se_mr_bwt_gss, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_bwt_list_gss)):
    x_axis = [index_group[i]+offset]*len(mr_bwt_list_gss[i])
    ax1.scatter(x_axis, mr_bwt_list_gss[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax1.bar([size + 2*offset for size in index_group], avg_mr_bwt_clser, width=bar_width, align='center', label='Dual-LS', zorder=2, color='limegreen')
ax1.errorbar([size + 2*offset for size in index_group], avg_mr_bwt_clser, yerr=se_mr_bwt_clser, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(mr_bwt_list_clser)):
    x_axis = [index_group[i]+2*offset]*len(mr_bwt_list_clser[i])
    ax1.scatter(x_axis, mr_bwt_list_clser[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)



ax1.set_xlabel('Buffer Size', font='Arial', fontsize=font_size)
ax1.set_ylabel('MR-BWT', font='Arial', fontsize=font_size)
# ax1.set_ylim(0, 0.6)
# ytick1 = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# ax1.set_yticks(ytick1)
# ax1.set_yticklabels(['0','0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40'])
ax1.tick_params(axis='y', labelsize=font_size)
ax1.set_xticks(index_group)
ax1.set_xticklabels(['500', '1,000', '2,000', '4,000'])
ax1.tick_params(axis='x', labelsize=font_size)
ax1.grid(False)  
ax1.yaxis.grid(True, color='lightgray', linewidth=0.5)

handles, labels = ax1.get_legend_handles_labels()
fig1.legend(handles, labels,  loc='upper right', bbox_to_anchor=(0.8, 0.97), shadow=False, ncol=3)
plt.tight_layout()
# plt.savefig('./plot_pdf/bar_8_tasks_MR_BWT.pdf')
plt.show()



# barchart_8_tasks_MR
fig2, ax2 = plt.subplots(figsize=figsize)

ax2.bar(index_group, avg_mr_agem, width=bar_width, label='A-GEM', zorder=2, color='gold')
ax2.errorbar(index_group, avg_mr_agem, yerr=se_mr_agem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_list_agem)):
    x_axis = [index_group[i]]*len(mr_list_agem[i])
    ax2.scatter(x_axis, mr_list_agem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax2.bar([size - offset for size in index_group], avg_mr_gem, width=bar_width, align='center', label='GEM', zorder=2, color='orange')
ax2.errorbar([size - offset for size in index_group], avg_mr_gem, yerr=se_mr_gem, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_list_gem)):
    x_axis = [index_group[i]-offset]*len(mr_list_gem[i])
    ax2.scatter(x_axis, mr_list_gem[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)


ax2.bar([size - 2*offset for size in index_group], avg_mr_der, width=bar_width, align='center', label='DER', zorder=2, color='steelblue')
ax2.errorbar([size - 2*offset for size in index_group], avg_mr_der, yerr=se_mr_der, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(mr_list_der)):
    x_axis = [index_group[i]-2*offset]*len(mr_list_der[i])
    ax2.scatter(x_axis, mr_list_der[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax2.bar([size + offset for size in index_group], avg_mr_gss, width=bar_width, align='center', label='GSS', zorder=2, color='slateblue')
ax2.errorbar([size + offset for size in index_group], avg_mr_gss, yerr=se_mr_gss, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')
# same as above
for i in range(len(mr_list_gss)):
    x_axis = [index_group[i]+offset]*len(mr_list_gss[i])
    ax2.scatter(x_axis, mr_list_gss[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)



ax2.bar([size + 2*offset for size in index_group], avg_mr_clser, width=bar_width, align='center', label='Dual-LS', zorder=2, color='limegreen')
ax2.errorbar([size + 2*offset for size in index_group], avg_mr_clser, yerr=se_mr_clser, fmt='none', 
             capsize=error_bar_capsize, elinewidth=error_bar_elinewidth, capthick=error_bar_capthick, color='k')

# same as above
for i in range(len(mr_list_clser)):
    x_axis = [index_group[i]+2*offset]*len(mr_list_clser[i])
    ax2.scatter(x_axis, mr_list_clser[i],  color='white' , edgecolors='k', marker='o', s=3, zorder=2)

ax2.set_xlabel('Buffer Size', font='Arial', fontsize=font_size)
ax2.set_ylabel(r'$\text{MR}_{\text{Ave}} (\%)$',  font='Arial', fontsize=font_size)
ax2.set_ylim(0, 50)
ytick2 = [0, 10, 20, 30, 40, 50]
ax2.set_yticks(ytick2)
ax2.tick_params(axis='y', labelsize=font_size)
ax2.set_xticks(index_group)
ax2.set_xticklabels(['500', '1,000', '2,000', '4,000'])
ax2.tick_params(axis='x', labelsize=font_size)
ax2.grid(False)
ax2.yaxis.grid(True, color='lightgray', linewidth=0.5, zorder=1)

handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels,  loc='upper right', bbox_to_anchor=(0.97, 0.94), shadow=False, ncol=3)
plt.tight_layout()
# plt.savefig('./plot_pdf/bar_8_tasks_MR.pdf')
plt.show()


# ============================ Export Data to Excel ============================
import pandas as pd
from pandas import ExcelWriter

# Create Excel writer
with ExcelWriter('mr_results.xlsx') as writer:
    # Process each buffer size
    for i, bf in enumerate(index_bf):
        # MR-BWT data for current buffer size
        mr_bwt_data = {
            'Model': ['A-GEM', 'GEM', 'DER', 'GSS', 'Dual-LS'],
            'Average': [
                avg_mr_bwt_agem[i],
                avg_mr_bwt_gem[i],
                avg_mr_bwt_der[i],
                avg_mr_bwt_gss[i],
                avg_mr_bwt_clser[i]
            ],
            'Std_Error': [
                se_mr_bwt_agem[i],
                se_mr_bwt_gem[i],
                se_mr_bwt_der[i],
                se_mr_bwt_gss[i],
                se_mr_bwt_clser[i]
            ],
            'Raw_Data': [
                mr_bwt_list_agem[i],
                mr_bwt_list_gem[i],
                mr_bwt_list_der[i],
                mr_bwt_list_gss[i],
                mr_bwt_list_clser[i]
            ]
        }
        
        # MR data for current buffer size
        mr_data = {
            'Model': ['A-GEM', 'GEM', 'DER', 'GSS', 'Dual-LS'],
            'Average': [
                avg_mr_agem[i],
                avg_mr_gem[i],
                avg_mr_der[i],
                avg_mr_gss[i],
                avg_mr_clser[i]
            ],
            'Std_Error': [
                se_mr_agem[i],
                se_mr_gem[i],
                se_mr_der[i],
                se_mr_gss[i],
                se_mr_clser[i]
            ],
            'Raw_Data': [
                mr_list_agem[i],
                mr_list_gem[i],
                mr_list_der[i],
                mr_list_gss[i],
                mr_list_clser[i]
            ]
        }
        
        # Convert to DataFrames
        mr_bwt_df = pd.DataFrame(mr_bwt_data)
        mr_df = pd.DataFrame(mr_data)
        
        # Write to Excel sheets
        sheet_prefix = f"Buffer_{bf}"
        mr_bwt_df.to_excel(writer, sheet_name=f"{sheet_prefix}_MR_BWT", index=False)
        mr_df.to_excel(writer, sheet_name=f"{sheet_prefix}_MR", index=False)

print("MR data exported to mr_results.xlsx")
