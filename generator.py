import torch
from torch import nn, optim

from helpers import get_sfft_shape

class Generator(nn.Module):
    def __init__(self, sample_rate=22500, snippet_length=10, input_size=1024, time_steps=51):
        super(Generator, self).__init__()
        self.time_steps, self.num_freqs = get_sfft_shape(sample_rate, snippet_length, time_steps)

        def ganlayer(n_input, n_output, dropout=True):
            pipeline = [nn.Linear(n_input, n_output)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.25))
            return pipeline

        self.model = nn.Sequential(
            *ganlayer(input_size, 128, dropout=False),
            *ganlayer(128, 256),
            *ganlayer(256, 512),
            *ganlayer(512, 1024),
            nn.Linear(1024, self.time_steps*self.num_freqs),
            nn.Tanh() # frequency amplitudes are normalized
        )

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(*(self.time_steps, self.num_freqs))
        return signal

