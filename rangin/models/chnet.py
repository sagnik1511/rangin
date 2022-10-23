import torch
import numpy as np
from .common import ConvBR, TransConvBR
import torch.nn as nn


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBR(in_channels, out_channels, 3, 1, 0),
            ConvBR(out_channels, out_channels, 2, 2, 1)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            TransConvBR(in_channels, out_channels, 4, 1, 2),
            TransConvBR(out_channels, out_channels, 4, 2, 0),
        )

    def forward(self, x):
        return self.block(x)


class CHNet(nn.Module):

    def __init__(self, levels=3, base_filter_dim=16):
        super(CHNet, self).__init__()
        self.bf = base_filter_dim
        self.encoder = self.create_encoder(levels)
        self.decoder = self.create_decoder(levels)

    def create_encoder(self, levels):

        base = []
        base.append(DownBlock(1, self.bf))
        for level in range(levels - 1):
            base.append(DownBlock(self.bf * 2 ** level, self.bf * 2 ** (level + 1)))

        return nn.Sequential(*base)

    def create_decoder(self, levels):
        base = []
        for lev in reversed(range(levels - 1)):
            base.append(UpBlock(self.bf * 3 * 2 ** (lev + 1), self.bf * 3 * 2 ** lev))
        base.append(UpBlock(self.bf * 3, 3))

        return nn.Sequential(*base)

    def _forward_call(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, x, x], dim=1)
        x = x.view(x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), -1)
        x = self.decoder(x)
        return x

    def forward(self, x):
        return self._forward_call(x)
