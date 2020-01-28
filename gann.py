import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from helpers import get_stft_shape


def compute_conv_output_dim(input_h, input_w, kernel_size, padding, stride):
    output_h = int(((input_h - kernel_size[0] + 2 * padding[0])/stride[0]) + 1)
    output_w = int(((input_w - kernel_size[1] + 2 * padding[1])/stride[1]) + 1)
    return output_h, output_w


class CustomModel(nn.Module):
    def __init__(self, device):
        super(CustomModel, self).__init__()
        self.device = device

    def load(self, model):
        if model:
            self.load_state_dict(torch.load(model))

    def train_step(self, data):
        raise NotImplementedError()


class LinearGenerator(CustomModel):
    def __init__(self, device, output_h, output_w, model=None, entropy_size=10, lr=0.0001):
        super(LinearGenerator, self).__init__(device)
        self.output_h = output_h
        self.output_w = output_w
        self.entropy_size = entropy_size
        self.entropy_size = entropy_size

        def ganlayer(input_dim, output_dim, dropout=True):
            pipeline = [nn.Linear(input_dim, output_dim)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.5))
            return pipeline

        self.model = nn.Sequential(
            *ganlayer(entropy_size, 32, dropout=False),
            *ganlayer(32, 64),
            *ganlayer(64, 128),
            *ganlayer(128, 256),
            nn.Linear(256, 2 * self.output_h * self.output_w),
            nn.Tanh()
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, x):
        x = self.model(x)
        x = x.view(*(-1, 2, self.output_h, self.output_w))
        return x

    def generate_data(self, num_samples, device, train=False):
        if not train:
            data = self(Variable(torch.randn(
                num_samples, self.entropy_size)).to(device)).detach()
        else:
            data = self(Variable(torch.randn(
                num_samples, self.entropy_size)).to(device))
        return data


class ConvolutionalGenerator(CustomModel):
    """Given some noise, or entropy, this generator creates an audio signal in the Short-Time Fourier space

    :param sample_rate: the sample_rate that the signal should be created at (default: 22050)
    :param bpm: the tempo of the created sample in beats per minute (default:120)
    :param entropy_size: the size of the entropy noise (default: 1024)
    :param num_beats: the number of time steps in the Short-Time Fourier representation of the signal
    """

    def __init__(self, device, output_h, output_w, model=None, entropy_size=10, lr=0.0001):
        super(ConvolutionalGenerator, self).__init__(device)
        self.output_h = output_h
        self.output_w = output_w
        self.entropy_size = entropy_size

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.entropy_size,
                               out_channels=8, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=8, out_channels=6,
                               kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=6, out_channels=4,
                               kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=4, out_channels=2,
                               kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(50, 2*self.output_w*self.output_h),
            nn.Tanh()  # frequency amplitudes are normalized
        )
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(*(-1, 2, self.output_h, self.output_w))
        return signal

    def generate_data(self, num_samples, device, train=False):
        if not train:
            data = self(Variable(torch.randn(
                num_samples, self.entropy_size, 1, 1)).to(device)).detach()
        else:
            data = self(Variable(torch.randn(
                num_samples, self.entropy_size, 1, 1)).to(device))
        return data


class Discriminator(CustomModel):
    """The discriminator uses a convolutional set-up to discriminate between fake and genuine audio signals in Short-Time Fourier space

    """

    CONFIG = {"num_layers": 5,
              "in_channels": [2, 4, 6, 8, 10],
              "out_channels": [4, 6, 8, 10, 12],
              "kernels": [(4, 13), (4, 10), (4, 4), (4, 4), (4, 4)],
              "strides": [(1, 4), (1, 2), (1, 2), (1, 2), (1, 2)],
              "paddings": [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0)],
              "relu_params": [0.2, 0.2, 0.2, 0.2, 0.2]
              }

    @staticmethod
    def conv_layer(in_channels, out_channels, kernel_size, stride, padding, relu_param):
        layer = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)]
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.LeakyReLU(relu_param, inplace=True))
        return layer

    @staticmethod
    def build_cnn_layers(config, input_h, input_w):
        num_layers = config["num_layers"]
        model = []
        output_h = input_h
        output_w = input_w
        output_c = config["out_channels"][-1]
        for i in range(num_layers):
            in_channels = config["in_channels"][i]
            out_channels = config["out_channels"][i]
            kernel_size = config["kernels"][i]
            stride = config["strides"][i]
            padding = config["paddings"][i]
            relu_param = config["relu_params"][i]
            layer = Discriminator.conv_layer(in_channels, out_channels,
                               kernel_size, stride, padding, relu_param)
            model += layer
            output_h, output_w = compute_conv_output_dim(
                output_h, output_w, kernel_size, padding, stride)
        output_size = output_h * output_w * output_c
        return nn.Sequential(*model), output_size

    def __init__(self, device, input_h, input_w, model = None, lr = 0.0001, momentum = 0.9, config = CONFIG):
        super(Discriminator, self).__init__(device)

        self.cnn_layers, output_dim=Discriminator.build_cnn_layers(
            config, input_h, input_w)
        self.linear_layers=nn.Sequential(
            nn.Linear(int(output_dim), 1),
            nn.Sigmoid()
        )

        self.optim=optim.SGD(self.parameters(), lr = lr, momentum = momentum)
        self.loss=nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, x):
        x=self.cnn_layers(x)
        x=x.view(x.size(0), -1)
        x=self.linear_layers(x)
        return x
