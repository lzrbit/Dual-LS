# Dual-LS
Code for "Complementary Learning System Empowers Online Continual Learning of Deep Neural Network", which has been submitted to **_Nature Machine Intelligence_**.




# File Structure
```text
Dual-LS/
├── README.md
├── cl_data_stream
│   ├── joint_dataset.py
│   ├── seq_dataset.py
│   └── traj_dataset.py
├── cl_model
│   ├── __init__.py
│   ├── agem.py
│   ├── continual_model.py
│   ├── der.py
│   ├── derppgssrev.py
│   ├── dual_ls.py
│   ├── gem.py
│   ├── gss.py
│   └── vanilla.py
├── experiments
│   ├── joint_training.py
│   ├── seq_training_all_task.py
│   └── testing_1_task.py
├── logging
│   └── original_reference
│       ├── EP0_target.npz
│       ├── FT_target.npz
│       ├── GL_target.npz
│       ├── LN_target.npz
│       ├── MA_target.npz
│       ├── MT_target.npz
│       ├── OF_target.npz
│       ├── SR_target.npz
│       ├── ZS0_target.npz
│       ├── ZS2_target.npz
│       └── val_index.pickle
├── mapfiles
│   ├── DR_CHN_Merging_ZS0.osm
│   ├── DR_CHN_Merging_ZS0.osm_xy
│   ├── DR_CHN_Merging_ZS2.osm
│   ├── DR_CHN_Merging_ZS2.osm_xy
│   ├── DR_CHN_Roundabout_LN.osm
│   ├── DR_CHN_Roundabout_LN.osm_xy
│   ├── DR_DEU_Merging_MT.osm
│   ├── DR_DEU_Merging_MT.osm_xy
│   ├── DR_DEU_Roundabout_OF.osm
│   ├── DR_DEU_Roundabout_OF.osm_xy
│   ├── DR_Intersection_CM.osm
│   ├── DR_Intersection_CM.osm_xy
│   ├── DR_LaneChange_ET0.osm
│   ├── DR_LaneChange_ET0.osm_xy
│   ├── DR_LaneChange_ET1.osm
│   ├── DR_LaneChange_ET1.osm_xy
│   ├── DR_Merging_TR0.osm
│   ├── DR_Merging_TR0.osm_xy
│   ├── DR_Merging_TR1.osm
│   ├── DR_Merging_TR1.osm_xy
│   ├── DR_Roundabout_RW.osm
│   ├── DR_Roundabout_RW.osm_xy
│   ├── DR_USA_Intersection_EP0.osm
│   ├── DR_USA_Intersection_EP0.osm_xy
│   ├── DR_USA_Intersection_EP1.osm
│   ├── DR_USA_Intersection_EP1.osm_xy
│   ├── DR_USA_Intersection_GL.osm
│   ├── DR_USA_Intersection_GL.osm_xy
│   ├── DR_USA_Intersection_MA.osm
│   ├── DR_USA_Intersection_MA.osm_xy
│   ├── DR_USA_Roundabout_EP.osm
│   ├── DR_USA_Roundabout_EP.osm_xy
│   ├── DR_USA_Roundabout_FT.osm
│   ├── DR_USA_Roundabout_FT.osm_xy
│   ├── DR_USA_Roundabout_SR.osm
│   └── DR_USA_Roundabout_SR.osm_xy
├── results_in_paper
│   ├── Figure_1
│   │   └── Figure_1_slides.pptx
│   ├── Figure_2
│   │   ├── fig_feature_dist
│   │   │   ├── README.md
│   │   │   ├── outputs
│   │   │   │   ├── histogram_distance_for_3_sces.pdf
│   │   │   │   ├── histogram_distance_train_EP0.pdf
│   │   │   │   ├── histogram_distance_train_FT.pdf
│   │   │   │   ├── histogram_distance_train_GL.pdf
│   │   │   │   ├── histogram_distance_train_LN.pdf
│   │   │   │   ├── histogram_distance_train_MA.pdf
│   │   │   │   ├── histogram_distance_train_OF.pdf
│   │   │   │   ├── histogram_distance_train_ZS0.pdf
│   │   │   │   ├── histogram_distance_train_ZS2.pdf
│   │   │   │   ├── histogram_vel_for_3_sces.pdf
│   │   │   │   ├── histogram_vel_train_EP0.pdf
│   │   │   │   ├── histogram_vel_train_FT.pdf
│   │   │   │   ├── histogram_vel_train_GL.pdf
│   │   │   │   ├── histogram_vel_train_LN.pdf
│   │   │   │   ├── histogram_vel_train_MA.pdf
│   │   │   │   ├── histogram_vel_train_OF.pdf
│   │   │   │   ├── histogram_vel_train_ZS0.pdf
│   │   │   │   └── histogram_vel_train_ZS2.pdf
│   │   │   ├── statistic_interaction_multi.py
│   │   │   ├── statistic_interaction_single.py
│   │   │   ├── statistic_vel_multi.py
│   │   │   ├── statistic_vel_single.py
│   │   │   └── utils.py
│   │   └── fig_map_traj-vis
│   │       ├── README.md
│   │       ├── maps
│   │       │   ├── DR_CHN_Merging_ZS0.osm
│   │       │   ├── DR_CHN_Merging_ZS2.osm
│   │       │   ├── DR_CHN_Merging_ZS2.osm_xy
│   │       │   ├── DR_CHN_Roundabout_LN.osm
│   │       │   ├── DR_CHN_Roundabout_LN.osm_xy
│   │       │   ├── DR_DEU_Merging_MT.osm
│   │       │   ├── DR_DEU_Merging_MT.osm_xy
│   │       │   ├── DR_DEU_Roundabout_OF.osm
│   │       │   ├── DR_DEU_Roundabout_OF.osm_xy
│   │       │   ├── DR_Intersection_CM.osm
│   │       │   ├── DR_Intersection_CM.osm_xy
│   │       │   ├── DR_LaneChange_ET0.osm
│   │       │   ├── DR_LaneChange_ET0.osm_xy
│   │       │   ├── DR_LaneChange_ET1.osm
│   │       │   ├── DR_LaneChange_ET1.osm_xy
│   │       │   ├── DR_Merging_TR0.osm
│   │       │   ├── DR_Merging_TR0.osm_xy
│   │       │   ├── DR_Merging_TR1.osm
│   │       │   ├── DR_Merging_TR1.osm_xy
│   │       │   ├── DR_Roundabout_RW.osm
│   │       │   ├── DR_Roundabout_RW.osm_xy
│   │       │   ├── DR_USA_Intersection_EP0.osm
│   │       │   ├── DR_USA_Intersection_EP0.osm_xy
│   │       │   ├── DR_USA_Intersection_EP1.osm
│   │       │   ├── DR_USA_Intersection_EP1.osm_xy
│   │       │   ├── DR_USA_Intersection_GL.osm
│   │       │   ├── DR_USA_Intersection_GL.osm_xy
│   │       │   ├── DR_USA_Intersection_MA.osm
│   │       │   ├── DR_USA_Intersection_MA.osm_xy
│   │       │   ├── DR_USA_Roundabout_EP.osm
│   │       │   ├── DR_USA_Roundabout_EP.osm_xy
│   │       │   ├── DR_USA_Roundabout_FT.osm
│   │       │   ├── DR_USA_Roundabout_FT.osm_xy
│   │       │   ├── DR_USA_Roundabout_SR.osm
│   │       │   └── DR_USA_Roundabout_SR.osm_xy
│   │       ├── outputs
│   │       │   ├── Traj_Map_DR_CHN_Merging_ZS0.png
│   │       │   ├── Traj_Map_DR_CHN_Merging_ZS2.png
│   │       │   ├── Traj_Map_DR_CHN_Roundabout_LN.png
│   │       │   ├── Traj_Map_DR_DEU_Roundabout_OF.png
│   │       │   ├── Traj_Map_DR_USA_Intersection_EP0.png
│   │       │   ├── Traj_Map_DR_USA_Intersection_GL.png
│   │       │   ├── Traj_Map_DR_USA_Intersection_MA.png
│   │       │   └── Traj_Map_DR_USA_Roundabout_FT.png
│   │       └── script
│   │           ├── __init__.py
│   │           ├── main_load_track_file.py
│   │           ├── main_visualize_data.py
│   │           └── utils
│   │               ├── __init__.py
│   │               ├── check_imports.py
│   │               ├── dataset_reader.py
│   │               ├── dataset_types.py
│   │               ├── dict_utils.py
│   │               ├── map_vis_lanelet2.py
│   │               ├── map_vis_without_lanelet.py
│   │               └── tracks_vis.py
│   ├── Figure_3
│   │   └── fig_statistic_results
│   │       ├── Bar_FDE.py
│   │       ├── Bar_MR.py
│   │       ├── Line_figure.py
│   │       ├── Line_figure_csv.py
│   │       ├── README.md
│   │       ├── matrix_figure.py
│   │       ├── overall_statistic.py
│   │       ├── plot_pdf
│   │       │   ├── BF_1000
│   │       │   │   ├── line_8_tasks_FDE.pdf
│   │       │   │   ├── line_8_tasks_MR.pdf
│   │       │   │   ├── matrix_8_tasks_FDE.pdf
│   │       │   │   └── matrix_8_tasks_MR.pdf
│   │       │   ├── BF_2000
│   │       │   │   ├── line_8_tasks_FDE.pdf
│   │       │   │   ├── line_8_tasks_MR.pdf
│   │       │   │   ├── matrix_8_tasks_FDE.pdf
│   │       │   │   └── matrix_8_tasks_MR.pdf
│   │       │   ├── BF_4000
│   │       │   │   ├── line_8_tasks_FDE.pdf
│   │       │   │   ├── line_8_tasks_MR.pdf
│   │       │   │   ├── matrix_8_tasks_FDE.pdf
│   │       │   │   └── matrix_8_tasks_MR.pdf
│   │       │   ├── BF_500
│   │       │   │   ├── line_8_tasks_FDE.pdf
│   │       │   │   ├── line_8_tasks_MR.pdf
│   │       │   │   ├── matrix_8_tasks_FDE.pdf
│   │       │   │   └── matrix_8_tasks_MR.pdf
│   │       │   ├── bar_8_tasks_FDE.pdf
│   │       │   ├── bar_8_tasks_FDE_BWT.pdf
│   │       │   ├── bar_8_tasks_MR.pdf
│   │       │   └── bar_8_tasks_MR_BWT.pdf
│   │       ├── source_data
│   │       │   ├── bar_figures
│   │       │   │   ├── fde_bars_results.xlsx
│   │       │   │   └── mr_bar_results.xlsx
│   │       │   ├── line_figures
│   │       │   │   ├── fde_results_buffer_1000.csv
│   │       │   │   ├── fde_results_buffer_2000.csv
│   │       │   │   ├── fde_results_buffer_4000.csv
│   │       │   │   ├── fde_results_buffer_500.csv
│   │       │   │   ├── mr_results_buffer_1000.csv
│   │       │   │   ├── mr_results_buffer_2000.csv
│   │       │   │   ├── mr_results_buffer_4000.csv
│   │       │   │   └── mr_results_buffer_500.csv
│   │       │   └── matrix_figures
│   │       │       ├── matrix_results_buffer_1000.xlsx
│   │       │       ├── matrix_results_buffer_2000.xlsx
│   │       │       ├── matrix_results_buffer_4000.xlsx
│   │       │       └── matrix_results_buffer_500.xlsx
│   │       └── utils.py
│   ├── Figure_4
│   │   ├── buffer_visualize.pptx
│   │   └── fig_buffer_plot
│   │       ├── README.md
│   │       ├── memory_plot.py
│   │       ├── plot_figures
│   │       │   ├── buffer_vis_gss_ds_100.pdf
│   │       │   ├── buffer_vis_reservoir_ds_100.pdf
│   │       │   ├── loss_type_0_ds_1.pdf
│   │       │   ├── model_update_plastic_ds_100.pdf
│   │       │   ├── model_update_stable_ds_100.pdf
│   │       │   ├── task_id_past_gss.pdf
│   │       │   └── task_id_past_reservoir.pdf
│   │       ├── plot_total_loss.py
│   │       ├── preprocess.py
│   │       ├── random_scatter.py
│   │       ├── sampled_matrix.py
│   │       └── source_data
│   │           ├── Figure4a
│   │           │   ├── buffer_memory_distribution_gss.xlsx
│   │           │   └── buffer_memory_distribution_reservoir.xlsx
│   │           ├── Figure4b
│   │           │   ├── task_id_distribution_gradient.xlsx
│   │           │   └── task_id_distribution_reservoir.xlsx
│   │           └── Figure4c
│   │               └── raw_data_of_total_loss.xlsx
│   ├── Figure_7
│   │   └── fig_case_plot
│   │       ├── FT_v1
│   │       │   ├── FT_case_1-1.pdf
│   │       │   ├── FT_case_1-2.pdf
│   │       │   ├── FT_case_2-1.pdf
│   │       │   ├── FT_case_2-2.pdf
│   │       │   ├── clser_minFDE_8.png
│   │       │   ├── clser_minMR.png
│   │       │   ├── clser_minMR_8.png
│   │       │   ├── vanilla_maxFDE.png
│   │       │   └── vanilla_maxMR_8.png
│   │       ├── MA_v1
│   │       │   ├── MA_case_1-1.pdf
│   │       │   ├── MA_case_1-2.pdf
│   │       │   ├── MA_case_2-1.pdf
│   │       │   ├── MA_case_2-2.pdf
│   │       │   ├── MA_case_3-1.pdf
│   │       │   ├── MA_case_3-2.pdf
│   │       │   ├── clser_minFDE.png
│   │       │   ├── clser_minFDE_.png
│   │       │   ├── clser_minMR.png
│   │       │   ├── clser_minMR_.png
│   │       │   ├── vanilla_maxMR.png
│   │       │   └── vanilla_maxMR_.png
│   │       ├── OF_v1
│   │       │   ├── OF_case_1-1.pdf
│   │       │   ├── OF_case_2-1.pdf
│   │       │   ├── clser_minFDE.png
│   │       │   ├── clser_minMR.png
│   │       │   ├── vanilla_maxFDE.png
│   │       │   └── vanilla_maxMR.png
│   │       ├── README.md
│   │       ├── case_study_figures
│   │       │   ├── FT_case_1-1.pdf
│   │       │   ├── FT_case_1-2.pdf
│   │       │   ├── FT_case_2-1.pdf
│   │       │   ├── FT_case_2-2.pdf
│   │       │   ├── MA_case_1-1.pdf
│   │       │   ├── MA_case_1-2.pdf
│   │       │   ├── MA_case_2-1.pdf
│   │       │   ├── MA_case_2-2.pdf
│   │       │   ├── MA_case_3-1.pdf
│   │       │   ├── MA_case_3-2.pdf
│   │       │   ├── OF_case_1-1.pdf
│   │       │   └── OF_case_2-1.pdf
│   │       └── src
│   │           ├── FT_0_1_plot.py
│   │           ├── FT_0_2_plot.py
│   │           ├── FT_2_1_plot.py
│   │           ├── FT_2_2_plot.py
│   │           ├── MA_0_1_plot.py
│   │           ├── MA_0_2_plot.py
│   │           ├── MA_698_1_plot.py
│   │           ├── MA_698_2_plot.py
│   │           ├── MA_721_1_plot.py
│   │           ├── MA_721_2_plot.py
│   │           ├── OF_0_2_plot.py
│   │           ├── OF_551_2_plot.py
│   │           ├── case_data
│   │           │   ├── FT_case_0_1.npz
│   │           │   ├── FT_case_0_2.npz
│   │           │   ├── FT_case_2_1.npz
│   │           │   ├── FT_case_2_2.npz
│   │           │   ├── MA_case_0_1.npz
│   │           │   ├── MA_case_0_2.npz
│   │           │   ├── MA_case_698_1.npz
│   │           │   ├── MA_case_698_2.npz
│   │           │   ├── MA_case_721_1.npz
│   │           │   ├── MA_case_721_2.npz
│   │           │   ├── OF_case_0_2.npz
│   │           │   └── OF_case_551_2.npz
│   │           ├── mapfiles
│   │           │   ├── DR_CHN_Merging_ZS0.osm
│   │           │   ├── DR_CHN_Merging_ZS0.osm_xy
│   │           │   ├── DR_CHN_Merging_ZS2.osm
│   │           │   ├── DR_CHN_Merging_ZS2.osm_xy
│   │           │   ├── DR_CHN_Roundabout_LN.osm
│   │           │   ├── DR_CHN_Roundabout_LN.osm_xy
│   │           │   ├── DR_DEU_Merging_MT.osm
│   │           │   ├── DR_DEU_Merging_MT.osm_xy
│   │           │   ├── DR_DEU_Roundabout_OF.osm
│   │           │   ├── DR_DEU_Roundabout_OF.osm_xy
│   │           │   ├── DR_Intersection_CM.osm
│   │           │   ├── DR_Intersection_CM.osm_xy
│   │           │   ├── DR_LaneChange_ET0.osm
│   │           │   ├── DR_LaneChange_ET0.osm_xy
│   │           │   ├── DR_LaneChange_ET1.osm
│   │           │   ├── DR_LaneChange_ET1.osm_xy
│   │           │   ├── DR_Merging_TR0.osm
│   │           │   ├── DR_Merging_TR0.osm_xy
│   │           │   ├── DR_Merging_TR1.osm
│   │           │   ├── DR_Merging_TR1.osm_xy
│   │           │   ├── DR_Roundabout_RW.osm
│   │           │   ├── DR_Roundabout_RW.osm_xy
│   │           │   ├── DR_USA_Intersection_EP0.osm
│   │           │   ├── DR_USA_Intersection_EP0.osm_xy
│   │           │   ├── DR_USA_Intersection_EP1.osm
│   │           │   ├── DR_USA_Intersection_EP1.osm_xy
│   │           │   ├── DR_USA_Intersection_GL.osm
│   │           │   ├── DR_USA_Intersection_GL.osm_xy
│   │           │   ├── DR_USA_Intersection_MA.osm
│   │           │   ├── DR_USA_Intersection_MA.osm_xy
│   │           │   ├── DR_USA_Roundabout_EP.osm
│   │           │   ├── DR_USA_Roundabout_EP.osm_xy
│   │           │   ├── DR_USA_Roundabout_FT.osm
│   │           │   ├── DR_USA_Roundabout_FT.osm_xy
│   │           │   ├── DR_USA_Roundabout_SR.osm
│   │           │   └── DR_USA_Roundabout_SR.osm_xy
│   │           ├── outputs
│   │           │   ├── FT_case_0_2.pdf
│   │           │   ├── MA_case_0_2.pdf
│   │           │   └── MA_case_698_2.pdf
│   │           └── visualization_utils
│   │               ├── dict_utils.py
│   │               ├── dictionary.py
│   │               ├── extract_original_tv_info.py
│   │               ├── map_vis_without_lanelet copy.py
│   │               └── map_vis_without_lanelet.py
│   └── Figure_8
│       └── fig_ablation_study
│           ├── PDF_files
│           │   ├── FDE-AVE_bar_1.pdf
│           │   ├── FDE-AVE_bar_2.pdf
│           │   ├── FDE-AVE_bar_3.pdf
│           │   ├── FDE-AVE_bar_4.pdf
│           │   ├── FDE-AVE_bar_5.pdf
│           │   ├── FDE-AVE_bar_6.pdf
│           │   ├── FDE-AVE_bar_7.pdf
│           │   ├── FDE-AVE_bar_8.pdf
│           │   ├── FDE-AVE_bar_9.pdf
│           │   ├── FDE_BWT_bar_1.pdf
│           │   ├── FDE_BWT_bar_2.pdf
│           │   ├── FDE_BWT_bar_3.pdf
│           │   ├── FDE_BWT_bar_4.pdf
│           │   ├── FDE_BWT_bar_5.pdf
│           │   ├── FDE_BWT_bar_6.pdf
│           │   ├── FDE_BWT_bar_7.pdf
│           │   ├── FDE_BWT_bar_8.pdf
│           │   ├── FDE_BWT_bar_9.pdf
│           │   ├── MR-AVE_bar_1.pdf
│           │   ├── MR-AVE_bar_2.pdf
│           │   ├── MR-AVE_bar_3.pdf
│           │   ├── MR-AVE_bar_4.pdf
│           │   ├── MR-AVE_bar_5.pdf
│           │   ├── MR-AVE_bar_6.pdf
│           │   ├── MR-AVE_bar_7.pdf
│           │   ├── MR-AVE_bar_8.pdf
│           │   ├── MR-AVE_bar_9.pdf
│           │   ├── MR_BWT_bar_1.pdf
│           │   ├── MR_BWT_bar_2.pdf
│           │   ├── MR_BWT_bar_3.pdf
│           │   ├── MR_BWT_bar_4.pdf
│           │   ├── MR_BWT_bar_5.pdf
│           │   ├── MR_BWT_bar_6.pdf
│           │   ├── MR_BWT_bar_7.pdf
│           │   ├── MR_BWT_bar_8.pdf
│           │   └── MR_BWT_bar_9.pdf
│           ├── README.md
│           ├── ablation_FDE_AVE.py
│           ├── ablation_FDE_BWT.py
│           ├── ablation_MR_AVE.py
│           ├── ablation_MR_BWT.py
│           ├── parameter_list.xlsx
│           ├── raw_data.xlsx
│           └── utils.py
├── test_CL.py
├── test_joint.py
├── train_CL.py
├── train_joint.py
├── traj_predictor
│   ├── __init__.py
│   ├── baselayers.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── interaction_model.py
│   ├── losses.py
│   ├── traj_para.py
│   └── utils.py
├── utils
│   ├── args_loading.py
│   ├── buffer_loss_gss.py
│   ├── buffer_loss_reservoir.py
│   ├── create_log_dir.py
│   ├── current_task_loss.py
│   ├── dual_structure_func.py
│   ├── gss_buffer.py
│   ├── metrics.py
│   ├── model_weights_loading.py
│   ├── model_weights_saving.py
│   └── reservoir_buffer.py
└── visualization_utils
    ├── dict_utils.py
    ├── dictionary.py
    ├── extract_original_tv_info.py
    └── map_vis_without_lanelet.py
```






# Processed Data
The experiments in this work are based on [INTERACTION dataset](https://interaction-dataset.com/).
The processed data is available in this link for [download](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).

# Implementations
## Enviroment
1. System: The codes can be run in **Ubuntu 22.04 LTS**.
2. **Python = 3.9**
3. **Pytorch = 2.0.0**
4. Other required packages are provided in "**requirements.txt**":
```
 pip install -r requirements.txt
```
## Configurations
1. Before running codes, please revise "**root_dir**" and "**data_dir**" in "_./utils/args_loading.py_" to your local paths.
2. Parameters for the networks can be also revised in "_./utils/args_loading.py_".


## Key Parameters for running the experiments
1. **--model**: the method you want to train and test. 
2. **--buffer_size**: the memory size of the continual learning methods to run, and set as 0 when using the vanilla method.
3. **--dataset**: set as "seq-interaction" when continual training, set as "joint-interaction" when joint training.
4. **--train_task_num**: the number of tasks in continual training.
5. **--debug_mode**: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.  
6. **--num_tasks**: the number of continual tasks for testing.


# Running
## Simple usage of the bash file
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test.sh
```
