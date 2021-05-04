
import torch
import torch.nn.functional as F
import torch.nn as nn
    
class DeepMutualLearning(nn.Module):
    def __init__(self):
        super(DeepMutualLearning,self).__init__()

    def forward(self,idx,outputs):
        my_outputs=outputs[idx]
        kd_loss=0
        for i,other_outputs in enumerate(outputs):
            if i!=idx:
                kd_loss+=F.kl_div(F.log_softmax(my_outputs,dim=-1),F.softmax(other_outputs.data.clone(),dim=1),reduction='batchmean')
        if i!=1:
            kd_loss/=(i+1-1)#avg

        return kd_loss
