
import torch
import torch.nn.functional as F
import torch.nn as nn
    
class DeepMutualLearning(nn.Module):
    def __init__(self, temperature):
        super(DeepMutualLearning,self).__init__()

    def forward(self,idx,outputs):
        my_outputs=outputs[idx]
        other_outputs = outputs[outputs!=outputs[idx]].detach().clone()
        kd_loss=F.kl_div(my_outputs,other_outputs).mean(dim=0)


        return kd_loss
