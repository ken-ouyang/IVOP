import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]



class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvRescaleNet, self).__init__()

        operations = []
        operations_final = []
        current_channel = channel_in
        
        # without SR
        if down_num == 0:
            channel_out = 1
            for j in range(block_num[0]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

            for j in range(2):
                b = InvBlockExp(subnet_constructor, 6, 3)
                operations_final.append(b)

        self.operations = nn.ModuleList(operations)
        self.operations_final = nn.ModuleList(operations_final)
        
        bBranch = [nn.Conv2d(channel_in, channel_in*4, kernel_size=5, stride=1, padding=2, bias=True), nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(channel_in*4, channel_in*8, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(channel_in*8, channel_in*16, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
          nn.ConvTranspose2d(channel_in*16, channel_in*8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
          nn.ConvTranspose2d(channel_in*8, channel_in*4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(channel_in*4, channel_in, kernel_size=5, stride=1, padding=2, bias=True)]
          
        self.uninvBranch = nn.Sequential(*bBranch)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
                    
            out_uninv = self.uninvBranch(x)    
            out_ =  torch.cat((out, out_uninv ), 1)   
            
            for op in self.operations_final:
                out_ = op.forward(out_, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_, rev)      
            return out_   
             
        else:
            for op in reversed(self.operations_final):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        
            out_ = out[:,:3,:,:]
            for op in reversed(self.operations):
                out_ = op.forward(out_, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_, rev)

        if cal_jacobian:
            return out_, jacobian
        else:
            return out_

