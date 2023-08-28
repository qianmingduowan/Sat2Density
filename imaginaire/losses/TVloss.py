import torch
import torch.nn as nn


class TV_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input):
        B,D1,D2,D3 = input.size()
        tv_d1 = torch.pow(input[:,1:,:,:]-input[:,:-1,:,:], 2).sum()
        tv_d2 = torch.pow(input[:,:,1:,:]-input[:,:,:-1,:], 2).sum()
        tv_d3 = torch.pow(input[:,:,:,1:]-input[:,:,:,:-1], 2).sum()
        return (tv_d1+tv_d2+tv_d3)/(B*D1*D2*D3)

