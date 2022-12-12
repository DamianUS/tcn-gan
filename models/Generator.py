import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.causal1 = CausalConv1d(in_channels = n_inputs,
                 out_channels = n_outputs,
                 kernel_size = kernel_size,
                 stride = stride,
                 dilation = dilation,
                 groups=1,
                 bias=True)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.causal2 = CausalConv1d(in_channels=n_outputs,
                                    out_channels=n_outputs,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    groups=1,
                                    bias=True)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.leaky_relu_2 = nn.LeakyReLU()
        self.net = nn.Sequential(self.causal1, self.bn1, self.leaky_relu_1,
                                 self.causal2, self.bn2, self.leaky_relu_2,)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.causal1.weight.data.normal_(0, 0.01)
        self.causal2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, seq_len=None, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.input_len = seq_len
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.permute(0,2,1)).permute(0,2,1)