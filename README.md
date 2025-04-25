# Dual-LS
Official Code for "**Complementary Learning System Empowers Online Continual Learning of Deep Neural Network**".


# Dataset
## Original Dataset
The experiments in this work are based on [INTERACTION dataset](https://interaction-dataset.com/).
## Processed Data
- The processed data is available in this link for [Google Drive](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).
- Please download the processed data in the direction ```./cl_dataset/```.

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
- Before running codes, please revise ```root_dir``` and ```data_dir``` in ```./utils/args_loading.py``` to your local paths.
- Parameters for the networks can be also revised in ```./utils/args_loading.py```.


## Key Parameters for running the experiments
- **--model**: the method you want to train and test.
- **--buffer_size**: the memory size of the continual learning methods to run, and set as 0 when using the vanilla method.
- **--dataset**: set as "seq-interaction" when continual training, set as "joint-interaction" when joint training.
- **--train_task_num**: the number of tasks in continual training.
- **--debug_mode**: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.
-  **--num_tasks**: the number of continual tasks for testing.


# Usage 
## Running
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test.sh
```
## Code
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
│       ├──...
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
- Main functions for training and testing are the python scripts in the root direction.
- The folder ```cl_model``` contains the proposed Dual-LS and compared CL baselines in experiments.
- The folder ```traj_predictor``` includes the files to construct the trajectory prediction model.
- Scripts for data pre-processing are provided in the folder ```cl_datastream```.


# Visualization of Experimental Results
- We have uploaded source data and scripts in the folder ```results_in_paper``` for re-visualization our experimental results in the paper.
- Each sub-folder contains the source data and scripts to visualize one figure in the paper.
- Please check README.md files in each sub-folder in ```results_in_paper``` and run the code to obtain the figures.


# Contact
If you have any questions or suggestions, feel free to contatct our main contributors:
- Zirui Li (ziruili.work.bit@gmail.com)
- Yunlong Lin (yunlonglin@bit.edu.cn)

