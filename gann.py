import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from helpers import get_stft_shape


class CustomModel(nn.Module):
    def __init__(self, device):
        super(CustomModel, self).__init__()
        self.device = device

    def load(self, model):
        if model:
            self.load_state_dict(torch.load(model))

    def train_step(self, data):
        raise NotImplementedError()


class Generator(CustomModel):
    """Given some noise, or entropy, this generator creates an audio signal in the Short-Time Fourier space

    :param sample_rate: the sample_rate that the signal should be created at (default: 22050)
    :param snippet_length: the length of the created signal in seconds (default: 10)
    :param entropy_size: the size of the entropy noise (default: 1024)
    :param time_steps: the number of time steps in the Short-Time Fourier representation of the signal
    """

    def __init__(self, device, model=None, sample_rate=22050, snippet_length=10, entropy_size=32, time_steps=65, lr=0.0001):
        super(Generator, self).__init__(device)
        self.time_steps, self.num_freqs = get_stft_shape(
            sample_rate, snippet_length, time_steps)
        self.sample_rate = sample_rate
        self.snippet_length = snippet_length
        self.time_steps = time_steps
        self.entropy_size = entropy_size

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(50, 2*self.time_steps*self.num_freqs),
            nn.Tanh()  # frequency amplitudes are normalized
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(*(-1, 2, self.time_steps, self.num_freqs))
        return signal

    def generate_data(self, num_samples, device, train=False):
        if not train:
            data = self(Variable(torch.randn(num_samples, self.entropy_size, 1, 1)).to(device)).detach()
        else:
            data =  self(Variable(torch.randn(num_samples, self.entropy_size, 1, 1)).to(device))
        return data



class Discriminator(CustomModel):
    """The discriminator uses a convolutional set-up to discriminate between fake and genuine audio signals in Short-Time Fourier space

    """

    def __init__(self, device, model=None, lr=0.0001, momentum=0.9):
        super(Discriminator, self).__init__(device)

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4,
                      kernel_size=(1, 11), stride=(1, 4), padding=(0, 0)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=(5, 5), stride=(1, 2), padding=(0,2)),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=8,
                      kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.AvgPool2d(kernel_size=2, stride=2),
            )

        self.linear_layers = nn.Sequential(
            nn.Linear(8 * 30 * 430, 1), # Depth x height x width
            nn.Sigmoid()
        )

        self.optim = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.loss = nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
