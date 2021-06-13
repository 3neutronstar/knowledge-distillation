
import torch
import torch.nn.functional as F
import torch.nn as nn
    
class SoftTarget(nn.Module):
    def __init__(self, temperature):
        super(SoftTarget,self).__init__()
        self.T=temperature
        self.type_loss='kl' #'mse'


    def forward(self,output,target):
        if self.type_loss=='kl':
            kd_loss=F.kl_div(F.log_softmax(output/self.T,dim=1),
                            F.softmax(target/self.T,dim=1),reduction='batchmean')*self.T*self.T
        elif self.type_loss=='mse':
            kd_loss=F.mse_loss(F.softmax(output/self.T,dim=1),F.softmax(target/self.T,dim=1))
        else:
            raise NotImplementedError
        return kd_loss
