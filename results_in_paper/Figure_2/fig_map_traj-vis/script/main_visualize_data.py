
from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils
import csv
import pandas as pd
import tqdm
import multiprocessing
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def visualize_data(sce):

    fontsize = 20
    alphas = 0.3

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                                                        "files)",default= sce, nargs="?")
    # parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="vehicle",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, default=100, nargs="?")
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()

    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, args.scenario_name + lanelet_map_ending)
    scenario_dir = os.path.join(tracks_dir, args.scenario_name)
    track_file_name = os.path.join(scenario_dir+"_train" + ".csv")

    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Did not find map file directory \"" + tracks_dir + "\"\n"
    # if not os.path.isdir(scenario_dir):
    #     error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Did not find track file \"" + track_file_name + "\"\n"

    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    # create a figure
    fig, axes = plt.subplots(1, 1)

    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    print("Loading map...")

    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin, alphas)
    # open track_file_name as dataframe:  
    df = pd.read_csv(track_file_name)
    case_ids = df['case_id'].unique()
    for case_id in tqdm.tqdm(case_ids, desc=f'Processing cases  for {sce}'):
        # get data for each case_id
        df_case_id = df[df['case_id'] == case_id]
        # get unique track_id
        track_ids = df_case_id['track_id'].unique()
        # loop for each track_id
        for track_id in track_ids:
            # print(f'case_id: {case_id}--track_id: {track_id}')
            # get data for each track_id
            data = df_case_id[df_case_id['track_id'] == track_id]
            
            # if agent_type is 'car'
            if data['agent_type'].iloc[0] == 'car':
                # plot the data
                plt.plot(data['x'], data['y'], color='r', linewidth=0.05, alpha=0.03) 
    plt.xlabel('X/m', fontsize=fontsize)
    plt.ylabel('X/m', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    output_filename = f'outputs_CN/Traj_Map_{sce}.png'  
    # tightly 
    plt.tight_layout()
    plt.savefig(output_filename, dpi=500, bbox_inches='tight')



if __name__ == '__main__':
    scenarios = [
        'DR_USA_Roundabout_FT',
        'DR_USA_Intersection_MA',
        'DR_USA_Intersection_GL',
        'DR_USA_Intersection_EP0',
        'DR_DEU_Roundabout_OF',
        'DR_CHN_Roundabout_LN',
        'DR_CHN_Merging_ZS2',
        'DR_CHN_Merging_ZS0'
    ]

    with multiprocessing.Pool() as pool:
        pool.map(visualize_data, scenarios)

    print('Done')
