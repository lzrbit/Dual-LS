import torch
from torch.nn import functional as F


def cal_buffer_loss_reservoir_logits(self):
    buf_inputs_past, buf_labels_past, buf_logits_past, buf_task_id_past = self.buffer_reservoir.get_data(self.args.minibatch_size, transform=self.transform)
    buf_outputs = self.net(buf_inputs_past)
    buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
    loss = 100*self.args.alpha * F.mse_loss(buf_heatmap_logits, buf_logits_past)
    del  buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
    torch.cuda.empty_cache()
    return loss, buf_inputs_past,  buf_labels_past, buf_logits_past, buf_task_id_past

def cal_buffer_loss_reservoir_positions(self):
    buf_inputs_past, buf_labels_past, buf_logits_past, buf_task_id_past = self.buffer_reservoir.get_data(self.args.minibatch_size, transform=self.transform)
    buf_outputs = self.net(buf_inputs_past)
    loss =  self.args.beta*self.loss(buf_outputs, buf_labels_past)
    del buf_outputs
    torch.cuda.empty_cache()
    return loss, buf_inputs_past, buf_labels_past, buf_logits_past, buf_task_id_past