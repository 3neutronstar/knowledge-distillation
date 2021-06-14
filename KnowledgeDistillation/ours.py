
import torch
import torch.nn.functional as F
import torch.nn as nn

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, temperature):
        super(PearsonCorrelationLoss,self).__init__()
        self.T=temperature

    def forward(self,output,target):
        kd_loss=self._covariance_loss(output,target)
        return kd_loss

    def _covariance_loss(self, logits, labels):
        loss=0.0
        for label in labels.unique():
            logits_label=logits[labels==label].clone()
            if logits_label.size()[0]==1: # batch_size
                continue
            labels_label=labels[labels==label].clone()
            batch_size, n_cats= logits_label.size()
            # removing the ground truth prob
            all_probs = F.softmax(logits_label/self.T, dim=1)

            label_inds=torch.ones_like(logits_label)
            label_inds[range(batch_size),labels_label.long()] = 0
            probs = all_probs * label_inds.detach()

            # re-normalize such that probs sum to 1
            #probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)
            probs = (F.softmax(logits_label/self.T, dim=1) + 1e-8)
            
            # cosine regularization
            #### I added under 2-line
            probs -= probs.mean(dim=1, keepdim=True)
            probs = probs / torch.sqrt(((probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
            ####
            #probs = probs / torch.sqrt(((all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
            cov_mat = torch.mm(probs,probs.T)
            cut_idx=1 - torch.eye(batch_size,dtype=torch.float,device='cuda')
            cov_mat=cov_mat*cut_idx
            loss += cov_mat.norm(p=1)
        loss/=labels.size()[0]
        return loss
