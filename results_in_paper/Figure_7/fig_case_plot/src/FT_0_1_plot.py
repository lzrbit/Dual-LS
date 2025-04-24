import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import math
from visualization_utils import map_vis_without_lanelet
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
font_size = 10

def rotation_matrix(rad):
    psi = rad - math.pi/2
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])

def get_polygon_cars(center, width, length, radian):
    lowleft = (- length / 2., - width / 2.)
    lowright = ( + length / 2., - width / 2.)
    upright = ( + length / 2., + width / 2.)
    upleft = ( - length / 2., + width / 2.)
    rotate_ = rotation_matrix(radian)
    return (np.array([lowleft, lowright, upright, upleft])).dot(rotate_)+center

# Settings
case_id = 0  # Must match the extracted case_id
test_sce = 'FT'
mapfile = './mapfiles/DR_USA_Roundabout_FT.osm'
case_data_file = f'./case_data/{test_sce}_case_{case_id}_1.npz'

# Load complete case data
data = np.load(case_data_file, allow_pickle=True)
obs_traj_all_agents = data['trajectory']
obs_pose_all_agents = data['pose_shape']
raw_tv_origin_x = data['x']
raw_tv_origin_y = data['y']

# Extract target vehicle information
tv_traj_x = obs_traj_all_agents[0][:,2] 
tv_traj_y = obs_traj_all_agents[0][:,3]
tv_origin_x = obs_traj_all_agents[0][-1,2]
tv_origin_y = obs_traj_all_agents[0][-1,3]
tv_heading = obs_pose_all_agents[0][-1,0]
tv_width = obs_pose_all_agents[0][-1,1]
tv_length = obs_pose_all_agents[0][-1,2]

# Create figure
fig = plt.figure(figsize=(7.16, 5))
gs = gridspec.GridSpec(2, 4, figure=fig, left=0.048, right=0.98, top=0.95, bottom=0.01, wspace=0.04, hspace=0)

# Predefine legend handles
handles = []

# Iterate through all 8 configurations
for i in range(8):
    # Get prediction data for current configuration
    prediction = data['predictions'][i]
    ground_truth = data['gts'][i]
    pred_heatmap = data['heatmaps'][i]
    config = data['configs'][i]
    
    # Create subplot
    ax = fig.add_subplot(gs[i])
    
    # Map visualization
    raw_origin = (raw_tv_origin_x, raw_tv_origin_y)
    origin = (tv_origin_x, tv_origin_y)
    rotate = rotation_matrix(tv_heading)
    xrange = [-30, 20]
    yrange = [-10, 45]
    
    map_vis_without_lanelet.draw_map_without_lanelet(mapfile, ax, raw_origin, rotate, xrange, yrange)
    
    # Heatmap
    x = np.arange(-22.75, 23.25, 0.5)
    y = np.arange(-11.75, 75.25, 0.5)
    s = pred_heatmap/np.amax(pred_heatmap)
    s[s<0.006]=np.nan
    ax.pcolormesh(x, y, s.transpose(), cmap='Reds', zorder=0)
    
    # Target vehicle
    bbox_tv = get_polygon_cars(origin, tv_width, tv_length, 0)
    rect_tv = matplotlib.patches.Polygon(bbox_tv, closed=True, facecolor='r', edgecolor='r', linewidth=1, alpha=0.5, zorder=20)
    ax.add_patch(rect_tv)
    
    # Surrounding vehicles
    for ii in range(1, 26):
        sv_traj_x = obs_traj_all_agents[ii][:,2]
        sv_traj_y = obs_traj_all_agents[ii][:,3]
        if np.all(sv_traj_x == 0) and np.all(sv_traj_y == 0): 
            continue
        elif sv_traj_x[0] == 0: 
            sv_traj_x = obs_traj_all_agents[ii][1:,2]
            sv_traj_y = obs_traj_all_agents[ii][1:,3]
        ax.plot(sv_traj_x, sv_traj_y, color='b')
        sv_current_x = obs_traj_all_agents[ii][8, 2]
        sv_current_y = obs_traj_all_agents[ii][8, 3]
        sv_current_head_relative = -(obs_pose_all_agents[ii][8, 0] - tv_heading)
        sv_width = obs_pose_all_agents[ii][8, 1]
        sv_length = obs_pose_all_agents[ii][8, 2]
        if sv_length == 0 and sv_width == 0:
            h_sv, = ax.scatter(sv_current_x, sv_current_y, marker='o', color='r', label='SV')
        else:
            bbox_sv = get_polygon_cars((sv_current_x, sv_current_y), sv_width, sv_length, sv_current_head_relative)
            rect_sv = matplotlib.patches.Polygon(bbox_sv, closed=True, facecolor='b', edgecolor='b', linewidth=1, alpha=0.5, zorder=20)
            ax.add_patch(rect_sv)
            h_sv, = ax.plot([], [], color='b', label='Surrounding Vehicle Trajectory')
    
    # Trajectory and prediction points
    h_tv, = ax.plot(tv_traj_x, tv_traj_y, label='Target Vehicle Trajectory', color='r')
    h_pred = ax.scatter(prediction[:, 0], prediction[:, 1], marker='*', s=8, color='gold', label='Predicted Points', zorder=200)
    h_gt = ax.scatter(ground_truth[0], ground_truth[1], marker='^', s=8, color='green', label='Ground Truth', zorder=199)
    
    # Set titles and axes
    if i not in [0, 4]:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    
    if i < 4:
        ax.set_title(f'Vanilla: After Task {config["learned_task"]}', fontsize=font_size, pad=2)
    else:
        ax.set_title(f'Dual-LS: After Task {config["learned_task"]}', fontsize=font_size, pad=2)
    
    # Collect legend handles
    if i == 0:
        handles = [h_tv, h_sv, h_gt, h_pred]

# Add unified legend
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=font_size, frameon=False)

plt.tight_layout()
plt.savefig(f'./outputs/{test_sce}_case_{case_id}_2.pdf', dpi=500, bbox_inches='tight')
plt.show()