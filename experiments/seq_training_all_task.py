import sys
import torch
from cl_model.continual_model import Continual_Model
import time
#import UQnet loss
from traj_predictor.losses import *
from traj_predictor.utils import *
from utils.args_loading import *
from utils.model_weights_loading import *
from utils.model_weights_saving import *

def train(model: Continual_Model,
          dataset,
          args):
    
    model.net.to(model.device)
    if args.replayed_rc:
        global replayed_data_recording
        replayed_data_recording = [1]*args.buffer_size

    print("The model for training:", args.model)
    if args.restart_training:
        task_num_pre = args.restart_load_task_id_weight
        print("Loaing weight from Scenario ", task_num_pre)
        print("You are in Scenario ", task_num_pre+1)
        model = model_weights_loading(model, args, task_num_pre)
        print("The trained weights loaded.")
        start_id = task_num_pre
    else:
        start_id = 0

    for t in range(start_id, args.train_task_num):
        train_loader = dataset.get_data_loaders(t)
        task_sample_num = len(train_loader)*args.batch_size
        total_batch_num = len(train_loader)
        for epoch in range(args.n_epochs):
            start_time = time.time()
            current = 0
            for i, data in enumerate(train_loader):
                is_last_sample = (i == total_batch_num - 1)
                
                current =current+args.batch_size

                if args.debug_mode and i == 10:
                    is_last_sample = True
                if args.debug_mode and i > 10:
                    print("\n >>>>>>>>>>>>In debuging>>>>>>>>>>>")
                    break
                traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data
                tensors_list = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
                tensors_list = [t.to(model.device) for t in tensors_list]
                inputs = (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
                labels = [ls, y]
                
                # to record replayed data for each task
                if args.replayed_rc:
                    # print("replayed data recording")
                    loss = model.observe(inputs, labels, i, t+1, is_last_sample)
                else:
                    loss = model.observe(inputs, labels, i)
                sys.stdout.write(f"\rTraining Progress:"
                                 f"  Epoch: {epoch+1}/{args.n_epochs}"
                                 f"    [{current:>6d}/{task_sample_num:>6d}]")
                sys.stdout.flush()
        
        # #write the logging text files for replayed data
        # if args.replayed_rc:
        #     with open("./logging/replayed_memory/"+str(args.model)+"_bf_"+str(args.buffer_size)+"_replayed_data_task{:.0f}.txt".format(t+1), 'w') as log_replay:
        #         log_replay.writelines("task"+str(t+1)+":")
        #         log_replay.writelines(str(replayed_data_recording))
        #     log_replay.close()

        model_weights_saving(args, epoch, t, model, dataset)


            
