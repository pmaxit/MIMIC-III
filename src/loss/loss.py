import torch.nn as nn
import torch.nn.functional as F
from fastai2.layers import *
import torch

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
    

    
class ICD_Loss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_func = BCEWithLogitsLossFlat()

    def forward(self,logits, labels):
        # logits is shape of 4 x 19 and labels is 19 with one hot encoding vector

        logits = logits.view(-1)
        labels = labels.view(-1)

        # ignore the first class . since that belongs to ""

        vlogits = logits[1:]                        # n x 19    -> n x 18
        vlabels = labels[1:]                        # n x 19    -> n x 18

        return self.loss_func(vlogits, vlabels)

    def decodes(self, x): return x > self.thres

    def activation(self, x): return torch.sigmoid(x)

