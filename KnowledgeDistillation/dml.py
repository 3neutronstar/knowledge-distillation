
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
                kd_loss+=F.kl_div(F.log_softmax(my_outputs),F.softmax(other_outputs.data.clone()))
        kd_loss/=(i-1)

        return kd_loss
