import numpy as np
import torch
from torch import nn
from torch.optim import Adam
try:
    import quadprog
except BaseException:
    print('Warning: GEM and A-GEM cannot be used on Windows (quadprog required)')

from cl_model.continual_model import Continual_Model
# from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
# from utils.gem_buffer import Buffer
from utils.reservoir_buffer import Buffer


def get_parser(parser):
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


class Gem(nn.Module):
    NAME = 'gem'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Gem, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.current_task = 0
        self.buffer = Buffer(self.args.buffer_size, self.device, self.NAME)

        # Allocate temporary synaptic memory
        self.grad_dims = []
        for pp in self.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        self.current_task += 1
        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))

        # add data to the buffer
        samples_per_task = self.args.buffer_size // self.args.train_task_num

        loader = dataset.train_loader
        traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = next(iter(loader))
        tensors_list_tmp = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
        tensors_list_tmp = [t.to(self.device) for t in tensors_list_tmp]
        cur_x =  (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
        cur_y = [ls, y]
        self.buffer.add_data(
            examples= cur_x,
            labels=cur_y,
            task_labels=torch.ones(samples_per_task,
                                   dtype=torch.long).to(self.device) * (self.current_task - 1)
        )

    def observe(self, inputs, labels):

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.buffer_size, transform=self.transform)

            for tt in buf_task_labels.unique():
                # compute gradient on the memory buffer
                self.opt.zero_grad()

                cur_task_inputs = ()
                for ii in range(len(buf_inputs)):
                    cur_task_inputs += (buf_inputs[ii][buf_task_labels == tt],)
                cur_task_labels = ()
                for jj in range(len(buf_labels)):
                    cur_task_labels += (buf_labels[jj][buf_task_labels == tt],)
                
                cur_task_outputs = self.net.forward(cur_task_inputs)
                penalty = self.loss(cur_task_outputs, cur_task_labels)
                penalty.backward()
                store_grad(self.parameters, self.grads_cs[tt], self.grad_dims)

        # now compute the grad on the current data
        self.opt.zero_grad()
        outputs = self.net.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        # check if gradient violates buffer constraints
        if not self.buffer.is_empty():
            # copy gradient
            store_grad(self.parameters, self.grads_da, self.grad_dims)

            dot_prod = torch.mm(self.grads_da.unsqueeze(0),
                                torch.stack(self.grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self.grads_da.unsqueeze(1),
                              torch.stack(self.grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads_da,
                               self.grad_dims)

        self.opt.step()

        return loss.item()
