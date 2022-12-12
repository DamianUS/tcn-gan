import torch
import torch.nn
from models.Discriminator import Discriminator
from models.Generator import TemporalConvNet

class TCNGAN(torch.nn.Module):
    def __init__(self, num_features, seq_len, batch_size, generator_channels, discriminator_channels, generator_kernel_size=7, discriminator_kernel_size=3, dropout=0.2):
        super(TCNGAN, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        generator_channels.append(num_features)
        self.generator = TemporalConvNet(num_inputs=num_features, num_channels=generator_channels, kernel_size=generator_kernel_size, seq_len=seq_len, dropout=dropout)
        self.discriminator = Discriminator(num_inputs=num_features, num_channels=discriminator_channels, kernel_size=discriminator_kernel_size, stride=1, dilation=1, padding=1, seq_len=seq_len)

    def forward(self, X, obj='discriminator'):
        assert obj in ['generator','discriminator'], "obj must be either generator or discriminator"
        if obj == 'generator':
            device = next(self.parameters()).device
            noise = torch.randn((self.batch_size, self.seq_len, self.num_features)).float().to(device)
            return self.generator(noise)
        elif obj == 'discriminator':
            return self.discriminator(X)

