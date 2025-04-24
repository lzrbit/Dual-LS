import sys
import os
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
from argparse import ArgumentParser
import torch
import numpy  # needed (don't change it)
import importlib
from traj_predictor.traj_para import paralist
from utils.create_log_dir import create_dir



def args_loading():
    # basic parameters
    torch.set_num_threads(4)
    parser = ArgumentParser(description='CL for interactive behavior learning')
    parser.add_argument('--dataset', type=str, default= 'seq-interaction')   # 'joint-interaction'    'seq-interaction'
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')
    
    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')

    parser.add_argument('--gamma', type=float, default= 0.5,
                        help='the added constant to solve QP in GEM')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default= device)
    parser.add_argument('--root_dir', type=str, default='/home/lzr/TaskFree-CL-SU/', help='Root directory')
    parser.add_argument('--data_dir', type=str, default='/home/lzr/TaskFree-CL-SU/cl_dataset/', help='Data directory')


    # training parameters
    parser.add_argument('--model', type=str, default= 'dual_ls')
    parser.add_argument('--experiment_index', type=int, default= 1)
    parser.add_argument('--debug_mode', type=bool, default=False)
    parser.add_argument('--restart_training', type=bool, default=False)
    parser.add_argument('--restart_load_task_id_weight', type=int, default=1)
    parser.add_argument('--lr', type=float, default= 0.001, help='Learning rate.')
    parser.add_argument('--n_epochs', type=int, default= 1, help='n_epochs.')

    # CL parameters
    parser.add_argument('--scenario_info', type=dict, 
                        default={0:'MA', 1:'FT', 2:'LN', 3:'ZS2', 4:'OF', 5:'EP0', 6:'GL', 7:'ZS0', 8:'MT', 9:'SR'},
                        help='Scenario order information')
    parser.add_argument('--train_task_num', type=int, default=8, help='The Number of Continual Tasks for Training')

    parser.add_argument('--buffer_size', type=int,default= 1000)
    parser.add_argument('--batch_size', type=int, default= 8, help='Batch size.')
    parser.add_argument('--alpha', type=float, default= 1.0, help='Penalty weight.')
    parser.add_argument('--beta', type=float, default= 1.0, help='Penalty weight.')
    parser.add_argument('--replayed_rc', type=bool, default=True,
                        help='turn True for replayed data logging')
    parser.add_argument('--replayed_rc_path', type=str, default='/home/lzr/TaskFree-CL-SU/logging/replayed_memory/')
    
    # specific parameters
    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.7)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.9)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)


    args = parser.parse_args()
    args.paralist = paralist
    args.minibatch_size = args.batch_size
    args.gss_minibatch_size  = args.minibatch_size
    args.result_dir, args.saved_dir = create_dir(args)


    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    return args