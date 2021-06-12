
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

    def _covariance_loss(self,logits, labels):
        bsz, n_cats, n_heads = logits.size()
        if n_heads < 2:
            return 0
        all_probs = torch.softmax(logits/self.T, dim=1)
        label_inds = torch.ones(bsz, n_cats).cuda()
        label_inds[range(bsz), labels] = 0
        # removing the ground truth prob
        probs = all_probs * label_inds.unsqueeze(-1).detach()
        # re-normalize such that probs sum to 1
        #probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)
        probs = (torch.softmax(logits/self.T, dim=1) + 1e-8)
        # cosine regularization
        #### I added under 2-line
        probs -= probs.mean(dim=1, keepdim=True)
        probs = probs / torch.sqrt(((probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
        ####
        #probs = probs / torch.sqrt(((all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
        cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
        pairwise_inds = 1 - torch.eye(n_heads).cuda()
        den = bsz * (n_heads -1) * n_heads
        loss = ((cov_mat * pairwise_inds).abs().sum() / den)
        return loss
