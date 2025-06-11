import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import gridspec
from utils import *

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
font_size = 8
plot_fde = False   # False   True

def plot_matrix(matrix, method_name, ax, show_colorbar=False, ii=0, plotfde=True):
    n = len(matrix)

    cmap = plt.get_cmap('viridis')

    if plotfde:
        norm = Normalize(vmin=0, vmax=3)
    else:
        norm = Normalize(vmin=0, vmax=52)
    mask = (matrix == 0.0)
    mask[-1,-1]=False
    cmap.set_bad(color='whitesmoke')

    cax = ax.imshow(np.ma.masked_array(matrix, mask=mask), cmap=cmap, norm=norm, origin='upper')
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'Task {i+1}' for i in range(n)], fontname='Arial', rotation=45, ha='center', fontsize=font_size)

    if ii % 3 == 0:  
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'After Task {i+1}' for i in range(n)], fontname='Arial', fontsize=font_size)
    else:
        ax.set_yticks([])  #

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                ax.text(j, i, '{:.1f}'.format(matrix[i,j]), va='center', ha='center', color='white', weight='bold', fontsize=font_size-2)
            else:
                ax.text(j, i, '', va='center', ha='center', color='black', weight='bold', fontsize=font_size)  

    if show_colorbar:
        if plotfde:
            cbar = plt.colorbar(cax, ax=ax)  # , ticks=np.arange(0, 10, 1)
            cbar.set_label('FDE (m)', fontsize=font_size)
        else:
            cbar = plt.colorbar(cax, ax=ax, ticks=np.arange(0, 51, 10))
            cbar.set_label('Miss Rate (%)', fontsize=font_size)
        
    ax.text(0.5, -0.32, f'({chr(ord("a") + ii)}) {method_name}', ha='center', fontsize=font_size, fontname='Times New Roman', transform=ax.transAxes)





# main
#methods: der_gssp_mixed  der  derpp  gem  agem  vanilla  
def to_obtain_matrix(method_name, if_fde=True, task_number=5, buffer=500):
    task_num = task_number
    method = method_name
    buffer_size = buffer
    listFDE = []
    listMR = []
    for i in range(1, task_num+1):
        minFDE_values, MR_values, _, _ = extract_results(base_dir='.', num_experiments=10, task_num=i, model=method, buffer_size=buffer_size)
        listFDE.append(minFDE_values)
        listMR.append(MR_values)

        if minFDE_values is not None and MR_values is not None:
            print("-------------task {:.0f}---------------".format(i))
            print(f"minFDE values: {minFDE_values}")
            print(f"MR values: {MR_values}")
        else:
            print("Failed to read the file.")


    matrix_FDE = np.zeros((task_num, task_num))
    matrix_MR = np.zeros((task_num, task_num))
    for ii in range(0, task_num):
        for jj in range(0, ii+1):
            matrix_FDE[ii, jj] =round(listFDE[ii][jj], 2)
            matrix_MR[ii, jj] = round(listMR[ii][jj], 2)
    print("\n\n\n matrix FDE\n", matrix_FDE)
    print("\n matrix MR\n", matrix_MR)
    if if_fde:
        return matrix_FDE
    else:
        return matrix_MR




fig = plt.figure(figsize=(7.16, 4.4))
# gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], left=0.08, bottom=0.13, right=0.96, top=0.97, wspace=0, hspace=0.4)
# 调用函数进行绘图

tasK_num = 8
bf = 500
ax = plt.subplot(gs[0, 0])
plot_matrix(to_obtain_matrix("vanilla", if_fde=plot_fde, task_number=tasK_num, buffer=0), "Vanilla", ax, ii=0, plotfde= plot_fde)  #
ax = plt.subplot(gs[0, 1])
plot_matrix(to_obtain_matrix("agem", if_fde=plot_fde, task_number=tasK_num, buffer=bf), "A-GEM", ax, ii=1, plotfde=plot_fde)  # 
ax = plt.subplot(gs[0, 2])
plot_matrix(to_obtain_matrix("gem", if_fde=plot_fde, task_number=tasK_num, buffer=bf), "GEM", ax, show_colorbar=True, ii=2, plotfde=plot_fde)  # 
ax = plt.subplot(gs[1, 0])
plot_matrix(to_obtain_matrix("der", if_fde=plot_fde, task_number=tasK_num, buffer=bf), "DER", ax, ii=3,plotfde= plot_fde)  # 
ax = plt.subplot(gs[1, 1])
plot_matrix(to_obtain_matrix("gss", if_fde=plot_fde, task_number=tasK_num, buffer=bf), "GSS", ax, ii=4, plotfde=plot_fde)  # 
ax = plt.subplot(gs[1, 2])
plot_matrix(to_obtain_matrix("clser", if_fde=plot_fde, task_number=tasK_num, buffer=int(bf/2)), "Dual-LS", ax, show_colorbar=True, ii=5, plotfde=plot_fde)  # 

#derppgssrev

plt.tight_layout()
# if plot_fde:
#     plt.savefig("./plot_pdf/BF_4000/matrix_8_tasks_FDE.pdf")
# else:
#     plt.savefig('./plot_pdf/BF_4000/matrix_8_tasks_MR.pdf')
plt.show()




# ============================ Export Matrix Data to Excel ============================
import pandas as pd
from pandas import ExcelWriter

def export_matrices_to_excel():
    methods = [
        ("vanilla", 0),
        ("agem", bf),
        ("gem", bf),
        ("der", bf),
        ("gss", bf),
        ("clser", int(bf/2))
    ]
    
    with ExcelWriter('matrix_results_buffer_{:.0f}.xlsx'.format(bf)) as writer:
        # Export FDE matrices
        for method_name, buffer in methods:
            matrix = to_obtain_matrix(method_name, if_fde=True, task_number=tasK_num, buffer=buffer)
            df = pd.DataFrame(matrix)
            df.columns = [f'Task {i+1}' for i in range(tasK_num)]
            df.index = [f'After Task {i+1}' for i in range(tasK_num)]
            df.to_excel(writer, sheet_name=f'{method_name}_FDE')
        
        # Export MR matrices
        for method_name, buffer in methods:
            matrix = to_obtain_matrix(method_name, if_fde=False, task_number=tasK_num, buffer=buffer)
            df = pd.DataFrame(matrix)
            df.columns = [f'Task {i+1}' for i in range(tasK_num)]
            df.index = [f'After Task {i+1}' for i in range(tasK_num)]
            df.to_excel(writer, sheet_name=f'{method_name}_MR')

    print("Matrix data exported to matrix_results.xlsx")

# Call the export function after your plotting code
export_matrices_to_excel()