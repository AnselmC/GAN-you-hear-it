import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor as tensor


def visualize_sample(sample, fignum=0):
    real = sample[0][0].detach().numpy()
    imag = sample[0][1].detach().numpy()
    # Min-max normalization s.t. it can be displayed as imag
    real = (real-real.min())/(real.max()-real.min())
    imag = (imag-imag.min())/(imag.max()-imag.min())

    x, y = real.shape
    x_factor = (y // np.sqrt(x*y)).astype(int)
    new_x = x * x_factor
    new_y = (x*y) // new_x
    real = real[:, :new_y*x_factor]
    imag = imag[:, :new_y*x_factor]
    real = real.reshape(new_x, new_y)
    imag = imag.reshape(new_x, new_y)
    plt.ion()
    plt.matshow(np.hstack([real, imag]), cmap="coolwarm", fignum=fignum)
    plt.show()
    plt.pause(0.0000001)


def get_stft_shape(sample_rate, snippet_length, time_steps):
    """ Gets the shape for the Short-Time Fourier Transform matrix corresponding to a sample_rate, snippet_length, and time_steps
    :param sample_rate: the sample rate of the time signal in Hz
    :param snippet_length: the length of the signal in seconds
    :param time_steps: the number of time steps that the signal should be split into

    :returns: the shape of the matrix with dim time steps times number of frequencies
    :rtype: tuple(int, int)
    """
    sample_length = snippet_length * sample_rate
    n_fft = (time_steps - 1) * 2
    win_length = int(n_fft/4)
    return (time_steps, int(sample_length/win_length + 1))


def save_as_time_signal(stft_signal, output_file, sr=22050):
    """Saves a signal in Short-Time Fourier space to file

    :param stft_signal: The 2-dimensional signal (time steps, number of frequencies) as `numpy.array` or `torch.Tensor`
    :param output_file: The path where to save the file to as a string
    :param sr: The sample rate of the signal (default 22050)

    """
    if type(stft_signal) is tensor:
        logger.debug("Converting signal from `torch.Tensor` to numpy array")
        stft_signal = stft_signal.detach().numpy()
    time_signal = librosa.istft(stft_signal)
    librosa.output.write(output_file, time_signal, sr)


class Progress:
    DISCRIMINATOR = "discriminator"
    GENERATOR = "generator"

    def __init__(self, num_epochs, num_batches, run_from_term):
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._run_from_term = run_from_term
        self._model = 0
        self._epoch = 0
        self._batch = 0
        self._loss = 0.
        self._fake_loss = 0.
        self._start_time_training = time.time()
        self._time_per_epoch = 0
        self._eta = "Estimating..."
        self._gen_progress_string = "\rGEN|Ep.: {:4d}/{:4d}|Batch: {:3d}/{:3d} | {:.2f}% | ETA: {} | Loss: {:.2f}|"
        self._dis_progress_string = "\rDIS|Ep.: {:4d}/{:4d}|Batch: {:3d}/{:3d} | {:.2f}% | ETA: {} | Losses(r/f): {:.2f}/{:.2f}|"

    def init_print(self):
        _, width = os.popen("stty size", "r").read().split()
        print("=" * int(width))
        print("Starting training...")
        print("=" * int(width))

    def update_epoch(self):
        self.switch_to_discriminator()
        self._loss = 0.
        self._fake_loss = 0.
        if self._epoch == 1:
            self._time_per_epoch = time.time() - self._start_time_training
        self._epoch += 1
        secs_left = self._time_per_epoch * (self._num_epochs - self._epoch)
        self._eta = datetime.timedelta(seconds=round(secs_left))
        self._print_progress_string()

    def update_batch(self, loss, fake_loss=0.):
        self._batch += 1
        self._loss = loss
        self._fake_loss = fake_loss
        self._print_progress_string()

    def switch_to_generator(self):
        self._batch = 0
        self._model = Progress.GENERATOR

    def switch_to_discriminator(self):
        self._batch = 0
        self._model = Progress.DISCRIMINATOR

    def _print_progress_string(self):
        _, width = os.popen("stty size", "r").read().split()
        percentage_done = 100 * (self._epoch * 2 * self._num_batches +
                                 int(self._model == Progress.GENERATOR) * self._num_batches + self._batch)/(self._num_epochs*2*self._num_batches)
        if self._model is Progress.DISCRIMINATOR:
            current_progress_string = self._dis_progress_string.format(self._epoch,
                                                                       self._num_epochs,
                                                                       self._batch,
                                                                       self._num_batches,
                                                                       percentage_done,
                                                                       self._eta,
                                                                       self._loss,
                                                                       self._fake_loss)
        else:
            current_progress_string = self._gen_progress_string.format(self._epoch,
                                                                       self._num_epochs,
                                                                       self._batch,
                                                                       self._num_batches,
                                                                       percentage_done,
                                                                       self._eta,
                                                                       self._loss)

        available_width = max(
            0, int(width) - len(current_progress_string.expandtabs()) - 2)

        progress_width = int(available_width * percentage_done/100)
        progress_bar = "[" + "#" * progress_width + \
            " " * (available_width - progress_width) + "]"
        current_progress_string += progress_bar
        print(current_progress_string, end="\r")
