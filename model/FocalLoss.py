# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:48:01 2020

@author: del
"""

"""
focal loss for multi-class classification
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, 
                 class_num,
                 alpha = None, 
                 gamma = 2,
                 use_alpha = False, 
                 size_average=None, 
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        
        self.ignore_index = ignore_index
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
            
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pi = torch.tensor(3.141592653589793)
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min = 0.0001, max = 1.0)
        
        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter(1, target.view(-1,1).long(), 1.)
        
        if self.use_alpha:
            #batch_loss = - self.alpha.double() * torch.pow(1-prob, self.gamma).double() * prob.log().double() * target_.double()
            batch_loss = - self.alpha.double() * torch.pow(torch.cos((pi/2)*prob), self.gamma).double() * prob.log().double() * target_.double()
        
        else:
            #batch_loss = torch.pow(1-prob, self.gamma).double() * prob.log().double() * target_.double()
            batch_loss = torch.pow(torch.cos((pi/2)*prob), self.gamma).double() * prob.log().double() * target_.double()
        
        batch_loss = batch_loss.sum(dim=1)
        
        if self.reduction == 'mean':
            loss = batch_loss.mean()
            
        else:
            loss = batch_loss.sum()
        
        return loss