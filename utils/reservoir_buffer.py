from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1




class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, minibatch_size, device, model_name, n_tasks=None, mode='reservoir'):
        assert mode in ('ring', 'reservoir')
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.model_name = model_name
        self.minibatch_size = minibatch_size
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        # create a dictionary to store the replayed data
        self.memory_data = {}
        # set the first step as 0
        self.memory_data[0] = [0]*self.buffer_size

        self.task_id_dict = {i: 0 for i in range(buffer_size)}

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: tuple, labels: torch.Tensor,
                     logits: tuple, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tuples containing tensor of historical trajectories and map information
        :param labels: the ground truth of predicted trajecotries
        :param logits: the predicted heatmap in UQnet
        :param task_labels: used in GEM method
        """

        for attr_str in self.attributes:
            attr = eval(attr_str) 
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                #setattr(对象，属性，所欲赋值)
                setattr(self, attr_str, list())
                if attr_str == 'examples':
                    for ii in range(len(attr)):
                        self.examples.append(torch.zeros((self.buffer_size,
                            *attr[ii].shape[1:]), dtype=typ, device=self.device))
                    self.examples = tuple(self.examples)
                
                if self.model_name == "derppgssrev" or self.model_name == "gem" or self.model_name == "agem" or self.model_name == "dual_ls":
                    if attr_str == 'labels':
                        for ii in range(len(attr)):
                            self.labels.append(torch.zeros((self.buffer_size,
                                *attr[ii].shape[1:]), dtype=typ, device=self.device))
                        self.labels = tuple(self.labels)          

                if self.model_name ==  "derppgssrev" or self.model_name == "der" or self.model_name == "dual_ls":
                    if attr_str == 'logits':
                        self.logits = torch.zeros((self.buffer_size,
                                *attr.shape[1:]), dtype=typ, device=self.device)
                        
                if self.model_name == "gem":
                        if attr_str == 'task_labels':
                            self.task_labels = torch.zeros((self.buffer_size,
                                *attr.shape[1:]), dtype=typ, device=self.device)



    def add_data(self, examples, labels=None, logits=None, task_labels=None, task_id=None, record_buffer=None, batch_id=None):

        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tuples containing tensor of historical trajectories and map information
        :param labels: the ground truth of predicted trajecotries
        :param logits: the predicted heatmap in UQnet
        :param task_labels: used in GEM method
        """

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
        for i in range(examples[0].shape[0]): 
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            
            # step = self.minibatch_size * batch_id + i + 1
            if index >= 0:
                # record the replayed data for further analysis
                self.memory_recording(index, self.num_seen_examples, task_id)
                self.task_id_dict[index] = task_id
                for ii in range(len(self.examples)):
                    self.examples[ii][index] = examples[ii][i].detach().to(self.device)
                
                if labels is not None:
                    for kk in range(len(self.labels)):
                        self.labels[kk][index] = labels[kk][i].detach().to(self.device)

                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)

                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
            else:
                self.memory_recording(index, self.num_seen_examples, task_id)
        if record_buffer:
            return self.memory_data
        else:
            return None



    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """

        if size > min(self.num_seen_examples, self.examples[0].shape[0]): 
            size = min(self.num_seen_examples, self.examples[0].shape[0]) 

        choice = np.random.choice(min(self.num_seen_examples, self.examples[0].shape[0]),
                                  size=size, replace=False)
        ret_task_id_list = [self.task_id_dict[int(key)] for key in choice]

        if transform is None:
            def transform(x): return x
        ret_list = [0 for _ in range(len(self.examples))] 
        for id_ex in range(len(self.examples)):
            ret_list[id_ex] = (torch.stack([transform(ee.cpu()) for ee in self.examples[id_ex][choice]]).to(self.device),)


        ret_tuple_tmp = tuple(ret_list)
        example_ret_tuple_tmp = ()
        for st in ret_tuple_tmp:
            example_ret_tuple_tmp += st

        label_ret_tuple_tmp = ()
        attr_str = self.attributes[1] #the first one is labels
        if hasattr(self, attr_str):
            attr = getattr(self, attr_str)
            for jj in range(len(self.labels)): 
                label_ret_tuple_tmp +=(attr[jj][choice],) 





        if self.model_name == "derppgssrev" or self.model_name == "dual_ls":
            ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp, self.logits[choice],ret_task_id_list)
            return ret_tuple
        elif self.model_name == "der":
            ret_tuple = (example_ret_tuple_tmp, self.logits[choice])
            return ret_tuple
        elif self.model_name == "gem":
            if self.task_labels is not None:
                ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp, self.task_labels[choice])
        
            if not return_index:
                return ret_tuple
            else:
                return (torch.tensor(choice).to(self.device), ) + ret_tuple
        elif self.model_name == "agem":
            if self.labels is not None:
                ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp)
        
            if not return_index:
                return ret_tuple
            else:
                return (torch.tensor(choice).to(self.device), ) + ret_tuple
        else:
            print("Model Error...!")

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def memory_recording(self, index, step, task_label):
        if index != -1:
            self.memory_data[step] = self.memory_data[step-1].copy()
            self.memory_data[step][index] = task_label
        else:
            self.memory_data[step] = self.memory_data[step-1].copy()


        # with open('./logging/replayed_memory/buffer_reservoir_'+self.model_name+'_bf_'+str(self.buffer_size)+'.txt', 'a') as f:
        #     f.write(f"Step {len(self.memory_data)-1}: {tmp_list_memory_in_this_step}\n")
