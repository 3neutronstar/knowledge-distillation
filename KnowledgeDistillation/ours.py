
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
            probs -= probs.mean(dim=1, keepdim=True)#TODO KL을 이 이후에 보면 될듯 함
            probs = probs / torch.sqrt(((probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
            ####
            #probs = probs / torch.sqrt(((all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
            cov_mat = torch.mm(probs,probs.T)
            cut_idx= ~torch.eye(batch_size,dtype=torch.bool,device='cuda')
            cov_mat=cov_mat[cut_idx]
            loss += (cov_mat.abs()**2).sum()
        loss/=labels.size()[0]
        return loss


class KLDivNoTruthLoss(nn.Module):
    def __init__(self, temperature):
        super(KLDivNoTruthLoss,self).__init__()
        self.T=temperature

    def forward(self,output,target):
        kd_loss=self._kldiv_loss(output,target)
        return kd_loss

    def _kldiv_loss(self, logits, labels):
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
            kl_loss=torch.zeros((batch_size,batch_size),device='cuda')
            for i,p in enumerate(probs):
                for j,p_2 in enumerate(probs):
                    if i==j:
                        continue
                    kl_loss[i,j]=F.kl_div(p,p_2)
            loss+=(kl_loss.abs()**2).sum()
        loss/=labels.size()[0]
        return loss
