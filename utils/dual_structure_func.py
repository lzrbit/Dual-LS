import torch
from torch.nn import functional as F


def cal_dual_structure_loss(self, buf_inputs_past, buf_labels_past, buf_logits_past):
    # plastic_model
    buf_outputs_p = self.plastic_model(buf_inputs_past)
    buf_log_lanescore_p, buf_heatmap_logits_p, buf_heatmap_reg_p = buf_outputs_p
    batch_size = buf_log_lanescore_p.size(0)  # Assuming buf_outputs_p has shape [batch_size, ...]
    individual_losses_p = torch.zeros(batch_size)  # Tensor to store each loss
    # Loop through each sample in the batch and compute loss individually
    for i in range(batch_size):
        # Select the outputs and labels of the i-th sample
        # The unsqueeze(0) adds a batch dimension with size 1 to match the expected input shape of self.loss
        sample_output = (buf_log_lanescore_p[i].unsqueeze(0), buf_heatmap_logits_p[i].unsqueeze(0), buf_heatmap_reg_p[i].unsqueeze(0))
        sample_label = [buf_labels_past[0][i].unsqueeze(0),buf_labels_past[1][i].unsqueeze(0) ]
        # Compute the loss for the i-th sample
        sample_loss = self.loss(sample_output, sample_label)
        # Store the computed loss
        individual_losses_p[i] = sample_loss

    # rsvr buffer  clser loss   
    # stable_model
    buf_outputs_s = self.stable_model(buf_inputs_past) 
    buf_log_lanescore_s, buf_heatmap_logits_s, buf_heatmap_reg_s = buf_outputs_s
    batch_size = buf_log_lanescore_s.size(0) 
    individual_losses_s = torch.zeros(batch_size)  # Tensor to store each loss
    # Loop through each sample in the batch and compute loss individually
    for i in range(batch_size):
        # Select the outputs and labels of the i-th sample
        # The unsqueeze(0) adds a batch dimension with size 1 to match the expected input shape of self.loss
        sample_output = (buf_log_lanescore_s[i].unsqueeze(0), buf_heatmap_logits_s[i].unsqueeze(0), buf_heatmap_reg_s[i].unsqueeze(0))
        sample_label = [buf_labels_past[0][i].unsqueeze(0),buf_labels_past[1][i].unsqueeze(0) ]
        # Compute the loss for the i-th sample
        sample_loss = self.loss(sample_output, sample_label)
        # Store the computed loss
        individual_losses_s[i] = sample_loss


    indices_p = torch.where(individual_losses_p <= individual_losses_s)
    indices_s = torch.where(individual_losses_p > individual_losses_s)
    # indexed by batch select the combination of s and p model
    buf_heatmap_logits_com = torch.zeros_like(buf_heatmap_logits_p)
    buf_heatmap_logits_com[indices_p] = buf_heatmap_logits_p[indices_p]
    buf_heatmap_logits_com[indices_s] = buf_heatmap_logits_s[indices_s]
    loss_dual_structure = 100 * self.args.alpha * F.mse_loss(buf_heatmap_logits_com, buf_logits_past)

    batch_sample_loss_record = torch.stack([individual_losses_p, individual_losses_s], dim=1).detach().cpu().numpy()

    del  buf_inputs_past
    del  buf_log_lanescore_p, buf_heatmap_logits_p, buf_heatmap_reg_p, buf_outputs_p, 
    del  buf_log_lanescore_s, buf_heatmap_logits_s, buf_heatmap_reg_s, buf_outputs_s
    del individual_losses_p, individual_losses_s, sample_loss, sample_output, sample_label
    del buf_heatmap_logits_com
    torch.cuda.empty_cache()
    return loss_dual_structure, batch_sample_loss_record



def update_plastic_model_variables(self):
    alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
    for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def update_stable_model_variables(self):
    alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
    for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
