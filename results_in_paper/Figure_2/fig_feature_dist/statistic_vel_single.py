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

    for caseid in tqdm(range(0, casesnb), desc="Processing all cases"):
        traj_case = traj[caseid]
        for timestamp in range(0, 9): 
            ego_vx = traj_case[0][timestamp][4]
            ego_vy = traj_case[0][timestamp][5]
            # calculate the sqrt of ego_vx and ego_vy
            speed = np.sqrt(ego_vx**2 + ego_vy**2)
            dis_list.append(speed)



    #plot the histogram of percentage
    plt.hist(dis_list, bins=50, color='darkred', density=True)   # 

    max_speed = max(dis_list)

    # add x limit and y limit 
    plt.xlim(1, max_speed)
    plt.ylim(0, 0.31)
    plt.xlabel('Speed (m/s)', fontsize=fontsize)
    plt.ylabel('Distribution', fontsize=fontsize)

    # change the font size of the ticks
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # plt.title('Histogram of Distance between Ego Vehicle and Surrounding Vehicles')
    # plt.show()
    fig_path = f'./outputs/histogram_vel_{dataset_name}.pdf'
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