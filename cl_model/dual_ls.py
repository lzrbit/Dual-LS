import torch
from torch.nn import functional as F
from torch import nn
from utils.gss_buffer import Buffer as Buffer_gss
from utils.reservoir_buffer import Buffer as Buffer_reservoir
from copy import deepcopy
from torch.optim import Adam
from utils.current_task_loss import current_task_loss
from utils.dual_structure_func import *
from utils.buffer_loss_gss import *
from utils.buffer_loss_reservoir import *
import pickle
import sys


class Dual_ls(nn.Module):
    NAME = 'dual_ls'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Dual_ls, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)


        self.buffer_gss = Buffer_gss(self.args.buffer_size, self.device, minibatch_size=self.args.minibatch_size, model_name=self.NAME, model=self)
        self.buffer_reservoir = Buffer_reservoir(self.args.buffer_size, self.args.minibatch_size, self.device, self.NAME)
        
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)

  
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
       
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.samples_seen = 0
        self.update_plastic_model_variables = update_plastic_model_variables
        self.update_stable_model_variables = update_stable_model_variables

        self.cal_buffer_loss_gss_logits =cal_buffer_loss_gss_logits
        self.cal_buffer_loss_gss_positions=cal_buffer_loss_gss_positions
        self.cal_buffer_loss_reservoir_logits=cal_buffer_loss_reservoir_logits
        self.cal_buffer_loss_reservoir_positions=cal_buffer_loss_reservoir_positions

        self.record_ls_sample_loss_gss = {}
        self.record_ls_sample_loss_reservoir = {}
        
        self.record_buffer_sample_task_id_gss = {}
        self.record_buffer_sample_task_id_reservoir = {}
        

        self.record_final_loss = {}
        self.plastic_model_update_random_dict = {}
        self.stable_model_update_random_dict = {}
    #gss function get_grads
    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        del loss
        torch.cuda.empty_cache()
        return grads


    def observe(self, inputs, labels, batch_id=None, task_id=None, is_last_sample=None):

        self.buffer_gss.drop_cache()
        self.buffer_gss.reset_fathom()
        self.opt.zero_grad()

        loss_current_task, heatmap_logits_current_task = current_task_loss(self, inputs, labels)



        # GSS buffer
        if not self.buffer_gss.is_empty():
            loss_gss_logits, buf_gss_inputs_past_1, buf_gss_labels_past_1, buf_gss_logits_past_1, _ = self.cal_buffer_loss_gss_logits(self)

            del buf_gss_inputs_past_1, buf_gss_labels_past_1, buf_gss_logits_past_1
            torch.cuda.empty_cache()
            loss_gss_positions, buf_gss_inputs_past_2, buf_gss_labels_past_2, buf_gss_logits_past_2, buf_gss_task_id_past_2 = self.cal_buffer_loss_gss_positions(self)

            self.record_buffer_sample_task_id_gss[batch_id] = buf_gss_task_id_past_2
            # dual structure loss for gss buffer
            loss_dual_gss, batch_sample_loss_record_gss = cal_dual_structure_loss(self, 
                                                                                    buf_gss_inputs_past_2, 
                                                                                    buf_gss_labels_past_2, 
                                                                                    buf_gss_logits_past_2)

            self.record_ls_sample_loss_gss[batch_id]=batch_sample_loss_record_gss

        # reservoir buffer
        if not self.buffer_reservoir.is_empty():
            loss_reservoir_logits, buf_reservoir_inputs_past_1, buf_reservoir_labels_past_1, buf_reservoir_logits_past_1, _  = self.cal_buffer_loss_reservoir_logits(self)
            del buf_reservoir_inputs_past_1, buf_reservoir_labels_past_1, buf_reservoir_logits_past_1
            torch.cuda.empty_cache()
            loss_reservoir_positions, buf_reservoir_inputs_past_2, buf_reservoir_labels_past_2, buf_reservoir_logits_past_2, buf_reservoir_task_id_past_2 = self.cal_buffer_loss_reservoir_positions(self)
            self.record_buffer_sample_task_id_reservoir[batch_id] = buf_reservoir_task_id_past_2

            # # dual structure loss for reservoir buffer
            loss_dual_reservoir, batch_sample_loss_reservoir = cal_dual_structure_loss(self, 
                                                                                        buf_reservoir_inputs_past_2, 
                                                                                        buf_reservoir_labels_past_2, 
                                                                                        buf_reservoir_logits_past_2)
            self.record_ls_sample_loss_reservoir[batch_id]=batch_sample_loss_reservoir

        if  self.buffer_gss.is_empty():
            loss_final = loss_current_task
            sys.stdout.write(f"Loss final: {loss_final:>.3f}")
            sys.stdout.flush()
        else:
            loss_final = loss_current_task + loss_gss_logits + loss_gss_positions + loss_reservoir_logits + loss_reservoir_positions + loss_dual_gss + loss_dual_reservoir
            self.record_final_loss[batch_id] = [loss_current_task.detach().cpu().numpy(),loss_gss_logits.detach().cpu().numpy(),loss_gss_positions.detach().cpu().numpy(),loss_reservoir_logits.detach().cpu().numpy(),loss_reservoir_positions.detach().cpu().numpy(),loss_dual_gss.detach().cpu().numpy(),loss_dual_reservoir.detach().cpu().numpy()]
            sys.stdout.write(f" final: {loss_final:>.3f}"
                             f" current: {loss_current_task:>.1f}"
                             f" gss logits: {loss_gss_logits:>.1f}"
                             f" gss pos: {loss_gss_positions:>.1f}"
                             f" res logits: {loss_reservoir_logits:>.1f}"
                             f" res positions: {loss_reservoir_positions:>.1f}"
                             f" dual gss: {loss_dual_gss:>.3f}"
                             f" dual res: {loss_dual_reservoir:>.1f}")
            sys.stdout.flush()
            
            
        # Update the plastic model and stable model
        self.global_step += 1
        plastic_model_update_random = torch.rand(1)
        self.plastic_model_update_random_dict[batch_id] = plastic_model_update_random
        if plastic_model_update_random < self.plastic_model_update_freq:
            self.update_plastic_model_variables(self)
        

        stable_model_update_random = torch.rand(1)
        self.stable_model_update_random_dict[batch_id] = stable_model_update_random
        if stable_model_update_random < self.stable_model_update_freq:
            self.update_stable_model_variables(self)
                
        loss_final.backward()
        
        self.opt.step()
        

        memory_replay_gss = self.buffer_gss.add_data(examples=inputs, labels=labels, logits=heatmap_logits_current_task.detach(),
                                 task_id=task_id, record_buffer = self.args.replayed_rc, batch_id=batch_id)
        
        memory_replay_reservoir = self.buffer_reservoir.add_data(examples=inputs, labels=labels, logits=heatmap_logits_current_task.detach(),
                                       task_id=task_id, record_buffer = self.args.replayed_rc, batch_id=batch_id)

        if self.args.replayed_rc and is_last_sample:
        # if self.args.replayed_rc and batch_id %10 ==0:    
            fold_path = self.args.replayed_rc_path

            save_name_gss_buffer_error = f'{task_id}_gss_buffer_error'
            save_path_gss_buffer_error = fold_path + save_name_gss_buffer_error
            save_name_reservoir_buffer_error = f'{task_id}_reservoir_buffer_error'
            save_path_reservoir_buffer_error = fold_path + save_name_reservoir_buffer_error

            save_name_gss_buffer_memory = f'{task_id}_gss_buffer_memory'
            save_path_gss_buffer_memory = fold_path + save_name_gss_buffer_memory
            save_name_reservoir_buffer_memory = f'{task_id}_reservoir_buffer_memory'
            save_path_reservoir_buffer_memory = fold_path + save_name_reservoir_buffer_memory

            save_name_loss_final = f'{task_id}_loss_final'
            save_path_loss_final = fold_path + save_name_loss_final

            save_name_plastic_model_update_random = f'{task_id}_plastic_model_update_random'
            save_path_plastic_model_update_random = fold_path + save_name_plastic_model_update_random

            save_name_stable_model_update_random = f'{task_id}_stable_model_update_random'
            save_path_stable_model_update_random = fold_path + save_name_stable_model_update_random
           
            save_name_gss_buffer_task_id = f'{task_id}_gss_buffer_task_id'
            save_path_gss_buffer_task_id = fold_path + save_name_gss_buffer_task_id
            save_name_reservoir_buffer_task_id = f'{task_id}_reservoir_buffer_task_id'
            save_path_reservoir_buffer_task_id = fold_path + save_name_reservoir_buffer_task_id


            with open(save_path_gss_buffer_error, 'wb') as f:
                pickle.dump(self.record_ls_sample_loss_gss, f)
            with open(save_path_reservoir_buffer_error, 'wb') as f:
                pickle.dump(self.record_ls_sample_loss_reservoir, f)

            if task_id == 8:
                with open(save_path_gss_buffer_memory, 'wb') as f:
                    pickle.dump(memory_replay_gss, f)
                with open(save_path_reservoir_buffer_memory, 'wb') as f:
                    pickle.dump(memory_replay_reservoir, f)

            with open(save_path_loss_final, 'wb') as f:
                pickle.dump(self.record_final_loss, f)

            with open(save_path_plastic_model_update_random, 'wb') as f:
                pickle.dump(self.plastic_model_update_random_dict, f)
            with open(save_path_stable_model_update_random, 'wb') as f:
                pickle.dump(self.stable_model_update_random_dict, f)

            with open(save_path_gss_buffer_task_id, 'wb') as f:
                pickle.dump(self.record_buffer_sample_task_id_gss, f)
            with open(save_path_reservoir_buffer_task_id, 'wb') as f:
                pickle.dump(self.record_buffer_sample_task_id_reservoir, f)

            self.record_ls_sample_loss_gss = {}
            self.record_ls_sample_loss_reservoir = {}

            self.record_final_loss = {}

            self.plastic_model_update_random_dict = {}
            self.stable_model_update_random_dict = {}

            self.record_buffer_sample_task_id_gss = {}
            self.record_buffer_sample_task_id_reservoir = {}
        return loss_final.item()
    


