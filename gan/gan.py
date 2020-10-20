import torch
from torch import nn, optim
from torch.autograd import Variable


def compute_conv_output_dim(input_h, input_w, kernel_size, padding, stride):
    kernel_h, kernel_w = kernel_size if isinstance(
        kernel_size, tuple) else (kernel_size, kernel_size)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride,
                                                                   stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding,
                                                                       padding)
    output_h = int(((input_h - kernel_h + 2 * padding_h) / stride_h) + 1)
    output_w = int(((input_w - kernel_w + 2 * padding_w) / stride_w) + 1)
    return int(output_h), int(output_w)


def compute_deconv_output_dim(input_h, input_w, kernel_size, dilation, stride,
                              padding, out_padding):
    kernel_h, kernel_w = kernel_size if isinstance(
        kernel_size, tuple) else (kernel_size, kernel_size)
    dilation_h, dilation_w = dilation if isinstance(
        dilation, tuple) else (dilation, dilation)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride,
                                                                   stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding,
                                                                       padding)
    out_padding_h, out_padding_w = out_padding if isinstance(
        out_padding, tuple) else (out_padding, out_padding)
    output_h = int((input_h - 1) - 2 * padding_h + dilation_h *
                   (kernel_h - 1) + out_padding_h + 1)
    output_w = int((input_w - 1) - 2 * padding_h + dilation_w *
                   (kernel_w - 1) + out_padding_w + 1)
    return int(output_h), int(output_w)


def compute_pool_output_dim(input_h, input_w, kernel_size):
    return int(input_h / kernel_size[0]), int(input_w / kernel_size[1])


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
    def __init__(self,
                 device,
                 output_h,
                 output_w,
                 model=None,
                 entropy_size=128,
                 lr=0.0001):
        super(LinearGenerator, self).__init__(device)
        self.output_h = output_h
        self.output_w = output_w
        self.entropy_size = entropy_size

        def ganlayer(input_dim, output_dim, dropout=True):
            pipeline = [nn.Linear(input_dim, output_dim)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.5))
            return pipeline

        self.model = nn.Sequential(
            *ganlayer(entropy_size, 32, dropout=False), *ganlayer(32, 64),
            *ganlayer(64, 128), *ganlayer(128, 256),
            nn.Linear(256, 2 * self.output_h * self.output_w), nn.Tanh())
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
            data = self(
                Variable(torch.randn(num_samples,
                                     self.entropy_size)).to(device)).detach()
        else:
            data = self(
                Variable(torch.randn(num_samples,
                                     self.entropy_size)).to(device))
        return data


class ConvolutionalGenerator(CustomModel):
    """Given some noise, or entropy, this generator creates an audio signal in the Short-Time Fourier space

    :param sample_rate: the sample_rate that the signal should be created at (default: 22050)
    :param bpm: the tempo of the created sample in beats per minute (default:120)
    :param entropy_size: the size of the entropy noise (default: 1024)
    :param num_beats: the number of time steps in the Short-Time Fourier representation of the signal
    """
    CONFIG = {
        "num_layers": 9,
        "in_channels": [-1, 64, 48, 32, 24, 16, 12, 8, 4],
        "out_channels": [64, 48, 32, 24, 16, 12, 8, 4, 2],
        "kernels": [3, 3, 3, 3, 3, 3, 3, 3, 3],
        "strides": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "paddings": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "out_paddings": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "dropout": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "relu_params": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    }

    @staticmethod
    def conv_layer(in_channels, out_channels, kernel_size, stride, padding,
                   dropout, relu_param):
        layer = [
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(relu_param, inplace=True)
        ]
        return layer

    @staticmethod
    def build_model(config, entropy_size, final_output_h, final_output_w):
        num_layers = config["num_layers"]
        config["in_channels"][0] = entropy_size
        model = []
        output_h = 1
        output_w = 1
        output_c = config["out_channels"][-1]
        print("GENERATOR DIMENSIONS")
        for i in range(num_layers):
            in_channels = config["in_channels"][i]
            out_channels = config["out_channels"][i]
            print(f"{output_h}x{output_w}x{in_channels}")
            kernel_size = config["kernels"][i]
            stride = config["strides"][i]
            padding = config["paddings"][i]
            dropout = config["dropout"][i]
            dilation = config["dilations"][i]
            out_padding = config["out_paddings"][i]
            relu_param = config["relu_params"][i]
            layer = ConvolutionalGenerator.conv_layer(in_channels,
                                                      out_channels,
                                                      kernel_size, stride,
                                                      padding, dropout,
                                                      relu_param)
            model += layer
            output_h, output_w = compute_deconv_output_dim(
                output_h, output_w, kernel_size, dilation, stride, padding, out_padding)
        print(f"{output_h}x{output_w}x{out_channels}")
        output_size = output_h * output_w * output_c
        model = nn.Sequential(
            *model, nn.Flatten(),
            nn.Linear(output_size, 2 * final_output_h * final_output_w),
            nn.Tanh())
        print(f"{final_output_h}x{final_output_w}x2")
        return model

    def __init__(self,
                 device,
                 output_h,
                 output_w,
                 model=None,
                 entropy_size=128,
                 lr=0.0001,
                 config=CONFIG):
        super(ConvolutionalGenerator, self).__init__(device)
        self.output_h = output_h
        self.output_w = output_w
        self.entropy_size = entropy_size

        self.model = ConvolutionalGenerator.build_model(
            config, entropy_size, output_h, output_w)

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
            data = self(
                Variable(torch.randn(num_samples, self.entropy_size, 1,
                                     1)).to(device)).detach()
        else:
            data = self(
                Variable(torch.randn(num_samples, self.entropy_size, 1,
                                     1)).to(device))
        return data


class Discriminator(CustomModel):
    """The discriminator uses a convolutional set-up to discriminate between fake and genuine audio signals in Short-Time Fourier space

    """

    CONFIG = {
        "num_layers": 5,
        "in_channels": [2, 8, 16, 32, 64],
        "out_channels": [8, 16, 32, 64, 128],
        "kernels": [(4, 13), (4, 10), (3, 3), (3, 3), (3, 3)],
        "strides": [(1, 4), (1, 2), (1, 1), (1, 1), (1, 1)],
        "paddings": [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0)],
        "pooling": [(1, 2), (1, 2), (1, 2), (2, 2), (2, 2)],
        "relu_params": [0.2, 0.2, 0.2, 0.2, 0.2]
    }

    @staticmethod
    def conv_layer(in_channels, out_channels, kernel_size, stride, padding,
                   pooling, relu_param):
        layer = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
        ]
        layer.append(nn.BatchNorm2d(out_channels))
        if pooling != (0, 0):
            layer.append(nn.AvgPool2d(pooling))
        layer.append(nn.LeakyReLU(relu_param, inplace=True))
        return layer

    @staticmethod
    def build_cnn_layers(config, input_h, input_w):
        num_layers = config["num_layers"]
        model = []
        output_h = input_h
        output_w = input_w
        output_c = config["out_channels"][-1]
        print("DISCRIMINATOR DIMENSIONS")
        for i in range(num_layers):
            in_channels = config["in_channels"][i]
            out_channels = config["out_channels"][i]
            print(f"{output_h}x{output_w}x{in_channels}")
            kernel_size = config["kernels"][i]
            stride = config["strides"][i]
            padding = config["paddings"][i]
            pooling = config["pooling"][i]
            relu_param = config["relu_params"][i]
            layer = Discriminator.conv_layer(in_channels, out_channels,
                                             kernel_size, stride, padding,
                                             pooling, relu_param)
            model += layer
            output_h, output_w = compute_conv_output_dim(
                output_h, output_w, kernel_size, padding, stride)
            if pooling != (0, 0):
                output_h, output_w = compute_pool_output_dim(
                    output_h, output_w, pooling)
        print(f"{output_h}x{output_w}x{out_channels}")
        output_size = output_h * output_w * output_c
        return nn.Sequential(*model), output_size

    def __init__(self,
                 device,
                 input_h,
                 input_w,
                 model=None,
                 lr=0.0001,
                 momentum=0.9,
                 config=CONFIG):
        super(Discriminator, self).__init__(device)

        self.cnn_layers, output_dim = Discriminator.build_cnn_layers(
            config, input_h, input_w)
        self.linear_layers = nn.Sequential(nn.Linear(int(output_dim), 1),
                                           nn.Sigmoid())

        self.optim = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.loss = nn.BCELoss()
        self.load(model)
        self.to(device)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
