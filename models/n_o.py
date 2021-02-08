import torch.nn as nn

from models.resnet import ResNet

class NOModel(nn.Module):
    def __init__(self, in_channels, out_channels_n_o, channels, num_blocks, dropout, dilation_list):

        super(NOModel, self).__init__()

        self.shared = ResNet(in_channels=in_channels, out_channels=out_channels_n_o, channels=channels, num_blocks=num_blocks, dropout=dropout, dilation_list=dilation_list)

        self.conv_n_o = nn.Conv2d(channels, out_channels_n_o, kernel_size=1, stride=1, padding=0)
        self.conv_o_n = nn.Conv2d(channels, out_channels_n_o, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.shared(x, return_intermediate=True)

        n_o_outs = self.conv_n_o(x)
        o_n_outs = self.conv_o_n(x)

        return n_o_outs,o_n_outs