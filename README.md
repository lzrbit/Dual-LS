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
├── results_in_paper
│   ├── Figure_1
│   │   └── ...
│   ├── Figure_2
│   │   └── ...
│   ├── Figure_3
│   │   └── ...
│   ├── Figure_4
│   │   └── ...
│   ├── Figure_7
│   │   └── ...
│   └── Figure_8
│   │   └── ...
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
