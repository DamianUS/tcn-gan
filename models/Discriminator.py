import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, pooling=False):
        super(ConvolutionalBlock, self).__init__()
        self.conv2d = nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding))
        self.relu = nn.ReLU()
        self.pooling = None
        self.net = nn.Sequential(self.conv2d, self.relu)
        # if pooling:
        #     self.pooling = nn.AvgPool2d(kernel_size=n_outputs)
        #     self.net = nn.Sequential(self.conv2d, self.relu, self.pooling)
        self.init_weights()

    def init_weights(self):
        self.conv2d.weight.data.normal_(0, 0.01)
        if self.pooling is not None:
            self.pooling.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, stride=1, dilation=1, padding=1, seq_len=None):
        super(Discriminator, self).__init__()
        self.input_len = seq_len
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            #dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [ConvolutionalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                     padding=padding)]
        #self.avg_pool = nn.AvgPool1d(kernel_size=num_channels[-1])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.conv_network = nn.Sequential(*self.layers)

    def forward(self, x):
        #  ignoring target_seq until we decide if teacher forcing is necessary
        x = x.permute(0, 2, 1).unsqueeze(-1)
        conv_out = self.conv_network(x).squeeze(-1)
        #out_pool = self.avg_pool(conv_out).squeeze(-1)
        out_pool = self.avg_pool(conv_out).view(conv_out.shape[0], -1)
        out_linear = self.linear(out_pool)
        return out_linear