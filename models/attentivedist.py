from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet, AttentionNet

class AttentiveDist(nn.Module):
    def __init__(self, in_channels, out_channels_dist, out_channels_angle, channels, num_blocks, dropout, dilation_list, pool, 
                out_channels_omega, out_channels_theta, out_channels_phi, attention=False, blocks_after_attn=0, stacking_attention=False):

        super(AttentiveDist, self).__init__()

        kernel_size = 3
        padding = 1
        self.out_channels_angle = out_channels_angle
        self.attention = attention

        if self.attention:
            self.shared = AttentionNet(in_channels=in_channels, out_channels=out_channels_dist, channels=channels, num_blocks=num_blocks, dropout=dropout, dilation_list=dilation_list, blocks_after_attn=blocks_after_attn)
        else:
            self.shared = ResNet(in_channels=in_channels, out_channels=out_channels_dist, channels=channels, num_blocks=num_blocks, dropout=dropout, dilation_list=dilation_list)

        self.conv_dist = nn.ModuleList([])
        self.conv_dist.append(nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
        self.conv_dist.append(nn.InstanceNorm2d(channels, affine=True))
        self.conv_dist.append(nn.ELU())
        self.conv_dist.append(nn.Conv2d(channels, out_channels_dist, kernel_size=1, stride=1, padding=0))

        self.conv_omega = nn.ModuleList([])
        self.conv_omega.append(nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
        self.conv_omega.append(nn.InstanceNorm2d(channels, affine=True))
        self.conv_omega.append(nn.ELU())
        self.conv_omega.append(nn.Conv2d(channels, out_channels_omega, kernel_size=1, stride=1, padding=0))

        self.conv_theta = nn.ModuleList([])
        self.conv_theta.append(nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
        self.conv_theta.append(nn.InstanceNorm2d(channels, affine=True))
        self.conv_theta.append(nn.ELU())
        self.conv_theta.append(nn.Conv2d(channels, out_channels_theta, kernel_size=1, stride=1, padding=0))

        self.conv_phi = nn.ModuleList([])
        self.conv_phi.append(nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
        self.conv_phi.append(nn.InstanceNorm2d(channels, affine=True))
        self.conv_phi.append(nn.ELU())
        self.conv_phi.append(nn.Conv2d(channels, out_channels_phi, kernel_size=1, stride=1, padding=0))

        if pool == 'Max':
            self.pool = nn.AdaptiveMaxPool2d((1,channels))
        elif pool == 'Avg':
            self.pool = nn.AdaptiveAvgPool2d((1,channels))
        self.conv_angle = nn.ModuleList([])
        self.conv_angle.append(nn.Conv1d(channels, 128, kernel_size=3, stride=1, padding=1))
        self.conv_angle.append(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1))
        self.conv_angle.append(nn.Conv1d(128, out_channels_angle, kernel_size=1, stride=1, padding=0))


    def forward(self, x1, x2=None, x3=None, x4=None, x5=None, x6=None, x7=None, x8=None):

        if self.attention:
            x, attention_maps = self.shared(x1, x2, x3, x4, return_intermediate=True)
        else:
            x = self.shared(x1, return_intermediate=True)

        #Dist prediction
        dist_outs = x
        for layer in self.conv_dist:
            dist_outs = layer(dist_outs)

        #Orientation prediction
        theta_outs = x
        for layer in self.conv_theta:
            theta_outs = layer(theta_outs)

        omega_outs = x
        for layer in self.conv_omega:
            omega_outs = layer(omega_outs)

        orientation_phi_outs = x
        for layer in self.conv_phi:
            orientation_phi_outs = layer(orientation_phi_outs)

        #Angle pred
        x = x.permute(0,2,3,1)
        x = self.pool(x)
        x = x.permute(0,3,1,2)
        x = x.squeeze(3)

        for layer in self.conv_angle:
            x = layer(x)
        angle_outs = x

        #For n output channels, n/2 channels are phi and n/2 channels are psi
        if self.attention:
            return dist_outs,omega_outs,theta_outs,orientation_phi_outs,angle_outs[:,:int(self.out_channels_angle/2),:],angle_outs[:,int(self.out_channels_angle/2):,:], attention_maps
        else:
            return dist_outs,omega_outs,theta_outs,orientation_phi_outs,angle_outs[:,:int(self.out_channels_angle/2),:],angle_outs[:,int(self.out_channels_angle/2):,:]