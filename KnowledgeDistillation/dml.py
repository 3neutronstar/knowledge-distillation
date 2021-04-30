
import torch
import torch.nn.functional as F
import torch.nn as nn
    
class DeepMutualLearning(nn.Module):
    def __init__(self):
        super(DeepMutualLearning,self).__init__()

    def forward(self,idx,outputs):
        my_outputs=outputs[idx]
        print(outputs[outputs!=outputs[idx]])
        other_outputs = outputs[outputs!=outputs[idx]].clone().detach()
        kd_loss=F.kl_div(my_outputs,other_outputs,reduction='batchmean')


        return kd_loss
