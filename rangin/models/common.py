import torch
import torch.nn as nn


def init_conv_weights(module):
    seq_obj = next(iter(module._modules.items()))
    conv_obj = seq_obj[1]
    nn.init.xavier_normal_(conv_obj.weight)


class ConvBR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, eps: float = 1e-3):
        super(ConvBR, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, eps=eps),
            nn.ReLU(),
        )
        init_conv_weights(self.cnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class TransConvBR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, eps: float = 1e-3, relu_slope=0.1):
        super(TransConvBR, self).__init__()
        self.ctn = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, eps=eps),
            nn.LeakyReLU(relu_slope),
        )
        init_conv_weights(self.ctn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctn(x)
