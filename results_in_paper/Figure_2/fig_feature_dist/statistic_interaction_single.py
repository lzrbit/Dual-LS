import numpy as np
from utils import * 
from tqdm import tqdm
import matplotlib.pyplot as plt

def statistic_interaction(dataset_name):
    fontsize = 15
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

    print("------------number of cases with only one agent:", invid_case)
    avg_dist = np.mean(dis_list)#the final result
    print("------------averge distance:", avg_dist, ' m')




    #plot the histogram of percentage
    plt.hist(dis_list, bins=50, color='darkred', density=True) 
    # add x limit and y limit 
    plt.xlim(0, 70)
    plt.ylim(0, 0.075)
    plt.xlabel('Distance (m)', fontsize=fontsize)
    plt.ylabel('Distribution', fontsize=fontsize)

    # change the font size of the ticks
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # plt.title('Histogram of Distance between Ego Vehicle and Surrounding Vehicles')
    # plt.show()
    fig_path = f'./outputs/histogram_distance_{dataset_name}.pdf'
    plt.tight_layout()
    plt.savefig(fig_path)
    # clear the plot
    plt.clf()


if __name__ == '__main__':

    # dataset_name_list is the list of dataset names in fold cl_dataset
    dataset_name_list = ['train_MA', 'train_FT','train_LN', 'train_ZS2', 
                         'train_OF', 'train_EP0','train_GL', 'train_ZS0']
    
    # dataset_name_list = ['train_FT', 'train_ZS2','train_GL']
    
    for dataset_name in dataset_name_list:
        print("------------Start processing ", dataset_name)
        statistic_interaction(dataset_name)
        print("------------Finish processing ", dataset_name)