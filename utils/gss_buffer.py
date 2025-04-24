from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

    def __init__(self, buffer_size, device, minibatch_size, model_name, model = None,  mode='reservoir'):
        assert mode in ('ring', 'reservoir')
        self.buffer_size = buffer_size
        self.device = device
        self.minibatch_size = minibatch_size
        self.num_seen_examples = 0
        self.functional_index = eval(mode)

        self.attributes = ['examples', 'labels', 'logits']
        self.model = model

        self.cache = {}
        self.fathom = 0
        self.fathom_mask = None
        self.reset_fathom()

        self.model_name = model_name
        
        # create a dictionary to store the replayed data
        self.memory_data = {}
        # set the first step as 0
        self.memory_data[0] = [0]*self.buffer_size
        
        self.task_id_dict = {i: 0 for i in range(buffer_size)}


    def reset_fathom(self):
        self.fathom = 0
        self.fathom_mask = torch.randperm(min(self.num_seen_examples, self.examples[0].shape[0] if hasattr(self, 'examples') else self.num_seen_examples))

    def get_grad_score(self, x, y, X, Y, indices):
        indices = np.array(indices[0])
        g = self.model.get_grads(x, y) # input-groundtruth
        
        G = []
        # Compute the loss of the samples from buffer
        for bc in range(len(indices)):
            for idx in indices:
                if idx in self.cache:
                    grd = self.cache[idx]
                else:
                    bfx = tuple(tensor[bc].unsqueeze(0) for  tensor in X)
                    bfy = tuple(tensor[bc].unsqueeze(0) for tensor in Y)
                    grd = self.model.get_grads(bfx, bfy)
                    self.cache[idx] = grd
                G.append(grd)
        G = torch.cat(G).to(g.device)
        c_score = 0
        grads_at_a_time = 5
        
        # let's split this so your gpu does not melt. You're welcome.
        for it in range(int(np.ceil(G.shape[0] / grads_at_a_time))):
            tmp = F.cosine_similarity(g, G[it*grads_at_a_time: (it+1)*grads_at_a_time], dim=1).max().item() + 1 #Algorithm 2, row 7
            c_score = max(c_score, tmp)
        return c_score

    def functional_reservoir(self, x, y, batch_c, bigX=None, bigY=None, indices=None):
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples, batch_c

        elif batch_c < 1:
            crx = tuple(x_tmp.unsqueeze(0) for x_tmp in x)
            cry = tuple(y_tmp.unsqueeze(0) for y_tmp in y)
            single_c = self.get_grad_score(crx, cry, bigX, bigY, indices)
            s = self.scores.cpu().numpy()
            pp=s / s.sum()
            i = np.random.choice(np.arange(0, self.buffer_size), size=1, p=pp)[0]
            rand = np.random.rand(1)[0]
            # print(rand, s[i] / (s[i] + c))
            if rand < s[i] / (s[i] + single_c):
                return i, single_c

        return -1, 0
    
    def drop_cache(self):
        self.cache = {}

    #-----------------------------------------------

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: tuple, labels: torch.Tensor,
                     logits: tuple) -> None:
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
                
                # setattr(obj,attr,assigned_value)
                setattr(self, attr_str, list())
                # examples are in a tuple form, including 8 elements(Tensors)
                if attr_str == 'examples':
                    for ii in range(len(attr)):
                        self.examples.append(torch.zeros((self.buffer_size,
                            *attr[ii].shape[1:]), dtype=typ, device=self.device))
                    self.examples = tuple(self.examples)
                
                #labels here to obtain gradient score, tuple including 2 elements(Tensors)
                if attr_str == 'labels':
                    for jj in range(len(attr)):
                        self.labels.append(torch.zeros((self.buffer_size,
                            *attr[jj].shape[1:]), dtype=typ, device=self.device))
                    self.labels = tuple(self.labels)               
                        #Sample scores initialization
                    self.scores = torch.zeros((self.buffer_size,*attr[0].shape[1:]),
                                            dtype=torch.float32, device=self.device)
                #GSS method does not need logits
                if self.model_name == "gss":
                    continue
                else:#the proposed method
                    #logits of UQnet are the heatmaps, which are in a form of Tensors
                    if attr_str == 'logits':
                        self.logits = torch.zeros((self.buffer_size,
                                *attr.shape[1:]), dtype=typ, device=self.device)


    def add_data(self, examples, labels=None, logits=None, task_labels=None, task_id=None,  record_buffer=None, batch_id=None):

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits)
        
        # compute buffer score
        if self.num_seen_examples > 0:
            bigX, bigY, indices = self.get_data(min(self.minibatch_size, self.num_seen_examples), give_index=True,
                                                random=True)
            c = self.get_grad_score(examples, labels, bigX, bigY, indices)
        else:
            bigX, bigY, indices = None, None, None
            c = 0.1

        for i in range(examples[0].shape[0]):
            new_example_tuple = ()
            new_label_tuple = ()
            for ex in examples:
                new_example_tuple += (ex[i],)
            for lb in labels:
                new_label_tuple += (lb[i],)


            index, score = self.functional_reservoir(new_example_tuple, new_label_tuple, c, bigX, bigY, indices)
            
            self.num_seen_examples += 1
        
            
            # step = self.minibatch_size*batch_id + i + 1
            if index >= 0:
                # record the replayed data for further analysis
                if record_buffer:
                   self.memory_recording(index, self.num_seen_examples, task_id)
                   self.task_id_dict[index] = task_id

                for ii in range(len(self.examples)):
                    self.examples[ii][index] = examples[ii][i].detach().to(self.device)
                if labels is not None:
                    for jj in range(len(self.labels)):
                        self.labels[jj][index] = labels[jj][i].detach().to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)               
            
                
                self.scores[index] = score
                if index in self.cache:
                    del self.cache[index]
            else:
                if record_buffer:
                   self.memory_recording(index, self.num_seen_examples, task_id)
                   
        if record_buffer:
            return self.memory_data
        else:
            return None



    def get_data(self, size: int, transform: nn.Module = None, give_index=False, random=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        """

        if size > self.examples[0].shape[0]:
            size = self.examples[0].shape[0]

        if random:
            choice = np.random.choice(min(self.num_seen_examples, self.examples[0].shape[0]),
                                  size=min(size, self.num_seen_examples),
                                  replace=False)
        else:
            choice = np.arange(self.fathom, min(self.fathom + size, self.examples[0].shape[0], self.num_seen_examples))
            choice = self.fathom_mask[choice]
            self.fathom += len(choice)
            if self.fathom >= self.examples[0].shape[0] or self.fathom >= self.num_seen_examples:
                self.fathom = 0
        
        ret_task_id_list = [self.task_id_dict[int(key)] for key in choice]
        
        if transform is None: transform = lambda x: x
        
        # To create the tuple to return
        ret_list = [0 for _ in range(len(self.examples))]
        for id_ex in range(len(self.examples)):
            ret_list[id_ex] = (torch.stack([transform(ee.cpu()) for ee in self.examples[id_ex][choice]]).to(self.device),) #此时ret_list中所有元素为包含张量的子元组
       
        # Replace the sub-tuple as tensors
        ret_tuple_tmp = tuple(ret_list)
        example_ret_tuple_tmp = ()
        for st in ret_tuple_tmp:
            example_ret_tuple_tmp += st        
        


        label_ret_tuple_tmp = ()
        for attr_str in self.attributes[1:]:#The 1st is "labels"
            if attr_str=='labels':
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    for jj in range(len(self.labels)): 
                        label_ret_tuple_tmp +=(attr[jj][choice],) 

       
        index_ret_tuple_tmp = ()


        if give_index:
            # print("give index, True")
            index_ret_tuple_tmp += (choice,)
            ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp, index_ret_tuple_tmp)
        else:
            # print("give index, False")
            if self.model_name == "gss":
                ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp, ret_task_id_list)
            else:
                ret_tuple = (example_ret_tuple_tmp, label_ret_tuple_tmp, self.logits[choice], ret_task_id_list)
        return ret_tuple







    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
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
        :param transform: the transformation to be applied (data augmentation)
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

        # with open('./logging/replayed_memory/buffer_diverse_'+self.model_name+'_bf_'+str(self.buffer_size)+'.txt', 'a') as f:
        #     f.write(f"Step {len(self.memory_data)-1}: {tmp_list_memory_in_this_step}\n")
