import numpy as np
from utils import * 
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def statistic_interaction_mutli(dataset_name):

    data_path = f'./cl_dataset/{dataset_name}.npz'
    data = np.load(data_path)

    traj = data['trajectory']#(cases, 26, 9, 8)
    nb_agents = data['nbagents']#(cases,) max agents in each case

    dis_list = []
    casesnb = len(traj)
    invid_case = 0

    for caseid in tqdm(range(0, casesnb), desc="Processing all cases"):
        traj_case = traj[caseid]
        if nb_agents[caseid] > 1:
            for timestamp in range(0, 9): 
                ego_x = traj_case[0][timestamp][0]
                ego_y = traj_case[0][timestamp][1]
                for sur_vehicle_id in range(1, nb_agents[caseid]):
                    sur_vehicle_x = traj_case[sur_vehicle_id][timestamp][0]
                    sur_vehicle_y = traj_case[sur_vehicle_id][timestamp][1]
                    dis_list.append(euclidean_distance(ego_x, ego_y, sur_vehicle_x, sur_vehicle_y))
        else:
            # print("Case ", caseid, " has only one agent, so no interaction")
            invid_case += 1
            pass
    return dis_list







if __name__ == '__main__':
    # dataset_name_list is the list of dataset names in fold cl_dataset
    dataset_name_list = ['train_FT', 'train_ZS2','train_GL']
    
        
    dis_list_1 = statistic_interaction_mutli(dataset_name_list[0])
    dis_list_2 = statistic_interaction_mutli(dataset_name_list[1])
    dis_list_3 = statistic_interaction_mutli(dataset_name_list[2])

    # plot the histogram of the distance
    # plt.hist(dis_list, bins=100, color='darkred') # , edgecolor='black'

    color_list = ['darkred','darkblue','darkorange']
    fontsize =  15

    # create a figure with figsize
    # plt.figure(figsize=(5,6))

    bin_width = 0.1
    bins = np.arange(0, max(max(dis_list_1), max(dis_list_2), max(dis_list_3)) + bin_width, bin_width)

    legend_list = ['Roundabout', 'Merging', 'Intersection']
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xpos = np.concatenate([bins[:-1]] * 3)
    ypos = np.repeat([0, 0.5, 1], len(bins) - 1)
    zpos = np.zeros_like(xpos)

    for i, data in enumerate([dis_list_1, dis_list_2, dis_list_3]):
        label = legend_list[i]
        kde = gaussian_kde(data, bw_method=0.3)  # 
        x_vals = np.linspace(bins[0], bins[-1], 100)  #
        y_vals = kde(x_vals)  # 
        ax.plot(x_vals, np.full_like(x_vals, i), y_vals, color=color_list[i], lw=2, label=label)
        # 
        verts = [(x, i, 0) for x in x_vals] + [(x, i, z) for x, z in zip(x_vals[::-1], y_vals[::-1])]
        poly = Poly3DCollection([verts], color=color_list[i], alpha=0.3)
        ax.add_collection3d(poly)
    


    
    
    plt.legend(['Roundabout', 'Merging', 'Intersection'], fontsize=fontsize)
    ax.set_xlabel('Distance (m)', fontsize=fontsize)
    ax.set_zlabel('Density', fontsize=fontsize)
    ax.set_yticks([0, 1, 2])

    # set x stick fontsize
    ax.tick_params(axis='x', labelsize=fontsize-5)
    ax.tick_params(axis='z', labelsize=fontsize-5)
    # add z limit
    ax.set_zlim(0, 0.1)

    ax.set_yticklabels([])
    ax.set_yticks([])
    # ax.grid(False)

    
    ax.set_facecolor('none')  # 
    # ax.legend(loc='lower right', fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize, bbox_to_anchor=(0.95, 0.75))

    fig_path = f'./outputs/histogram_distance_for_3_sces.pdf'
    plt.tight_layout()
    plt.savefig(fig_path,dpi = 500)
    plt.show()