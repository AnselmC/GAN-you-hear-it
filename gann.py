import torch
from torch import nn

from helpers import get_stft_shape


class Generator(nn.Module):
    """Given some noise, or entropy, this generator creates an audio signal in the Short-Time Fourier space

    :param sample_rate: the sample_rate that the signal should be created at (default: 22050)
    :param snippet_length: the length of the created signal in seconds (default: 10)
    :param input_size: the size of the entropy noise (default: 1024)
    :param time_steps: the number of time steps in the Short-Time Fourier representation of the signal
    """

    def __init__(self, sample_rate=22050, snippet_length=10, input_size=1024, time_steps=65):
        super(Generator, self).__init__()
        self.time_steps, self.num_freqs = get_stft_shape(
            sample_rate, snippet_length, time_steps)

        def ganlayer(n_input, n_output, dropout=True):
            pipeline = [nn.Linear(n_input, n_output)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.25))
            return pipeline

        self.model = nn.Sequential(
            *ganlayer(input_size, 32, dropout=False),
            *ganlayer(32, 64),
            nn.Linear(64, 2*self.time_steps*self.num_freqs),
            nn.Tanh()  # frequency amplitudes are normalized
        )

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(*(2, self.time_steps, self.num_freqs))
        return signal


class Discriminator(nn.Module):
    """The discriminator uses a convolutional set-up to discriminate between fake and genuine audio signals in Short-Time Fourier space

    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 16 * 1722, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
