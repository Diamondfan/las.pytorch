#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    pass


class GlobalAttention(nn.Module):
    "reference: Luong et al. https://arxiv.org/pdf/1508.04025.pdf"
    def __init__(self, mode='dot'):
        '''
        Args:
            mode : types of product, support only "dot"
        '''
        #TODO: bilinear and concat mode
        super(GlobalAttention, self).__init__()
        self.mode = mode

    def forward(self, srch, tgth):  
        '''
        Args:
            srch(batch_size, time, hidden_size) :   source hiiden states
            tgth(batch_size, 1, hidden_size)    :   target hiiden state
        Returns:
            ctx(batch_size, 1, hidden_size)     :   context vector to decoder
            ali(batch_size, time)               :   the distribution of srch, alpha(i, u), apply location-based attention
        '''
        tgth = tgth.transpose(1,2)
        score = torch.matmul(srch, tgth).squeeze(2)
        
        ali = F.softmax(score, dim=1)
        ctx = ali.unsqueeze(1)
        ctx = torch.matmul(ctx, srch)
        return ctx, ali

