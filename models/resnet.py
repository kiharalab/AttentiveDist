from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    def __init__(self, channels,kernel_size, padding, dropout, stride=1, dilation=1):

        super(BasicBlock2D, self).__init__()

        padding = padding * dilation

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding,dilation)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding,dilation)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.elu2 = nn.ELU()

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.elu2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_blocks, dropout, dilation_list):

        super(ResNet, self).__init__()
        assert (num_blocks > 4)

        kernel_size = 3
        padding = 1
        dilations = dilation_list

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu = nn.ELU()

        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =1, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =2, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =4, dropout=dropout))

        d_idx = 0
        for i in range(num_blocks - 4):
            self.residual_blocks.append(BasicBlock2D(channels,kernel_size, padding, dilation =dilations[d_idx], dropout=dropout))
            d_idx = (d_idx + 1) % len(dilations)

        self.convlast = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, return_intermediate=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        for layer in self.residual_blocks:
            x = layer(x)

        # x has dimension 1x64xLxL
        if return_intermediate:
            return x

        outs = self.convlast(x)
        return outs

class AttentionNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_blocks, dropout, dilation_list, blocks_after_attn):

        super(AttentionNet, self).__init__()

        kernel_size = 3
        padding = 1
        dilations = dilation_list

        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2))
        self.residual_blocks.append(nn.InstanceNorm2d(channels, affine=True))
        self.residual_blocks.append(nn.ELU())
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =1, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =2, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =4, dropout=dropout))

        d_idx = 0
        for i in range(num_blocks - 4):
            self.residual_blocks.append(BasicBlock2D(channels,kernel_size, padding, dilation =dilations[d_idx], dropout=dropout))
            d_idx = (d_idx + 1) % len(dilations)

        self.attention_linear = nn.Linear(in_features=channels, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

        self.residual_blocks2 = nn.ModuleList([])
        d_idx = 0
        for i in range(blocks_after_attn):
            self.residual_blocks2.append(BasicBlock2D(channels,kernel_size, padding, dilation =dilations[d_idx], dropout=dropout))
            d_idx = (d_idx + 1) % len(dilations)

        self.convlast = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2, x3, x4, return_intermediate=False):

        # Feature for each eval 
        for layer in self.residual_blocks:
            x1 = layer(x1)
        for layer in self.residual_blocks:
            x2 = layer(x2)
        for layer in self.residual_blocks:
            x3 = layer(x3)
        for layer in self.residual_blocks:
            x4 = layer(x4)

        # Attention
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 3, 1)
        x4 = x4.permute(0, 2, 3, 1)
        # print(x1)
        # print(x2)
        w_x1 = self.attention_linear(x1)
        w_x2 = self.attention_linear(x2)
        w_x3 = self.attention_linear(x3)
        w_x4 = self.attention_linear(x4)
        combined = torch.cat((w_x1,w_x2,w_x3,w_x4), dim=-1)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)
        attention_maps = self.softmax(combined)

        attention_maps = attention_maps.permute(0, 3, 1, 2)
        attention_maps = attention_maps.squeeze()
        

        #Weighted sum
        x = x1*attention_maps[0] + x2*attention_maps[1] + x3*attention_maps[2] + x4*attention_maps[3]

        for layer in self.residual_blocks2:
            x = layer(x)

        # x has dimension 1x64xLxL
        if return_intermediate:
            return x, attention_maps

        outs = self.convlast(x)
        return outs, attention_maps
