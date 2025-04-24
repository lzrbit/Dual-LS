


def current_task_loss(self, inputs, labels):
    outputs = self.net(inputs)
    log_lanescore, heatmap, heatmap_reg = outputs
    outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
    loss_current_task = self.loss(outputs_prediction, labels) # OverallLoss
    return loss_current_task, heatmap