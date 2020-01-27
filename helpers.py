import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor as tensor


def visualize_sample(samples):
    num_samples = len(samples)
    width = int(np.sqrt(num_samples))
    height = int(np.floor(num_samples/width))
    full_img = None
    row_img = None
    width_counter = 0
    for i, sample in enumerate(samples):
        if i >= height * width:
            break
        real = sample[0].detach().numpy()
        imag = sample[1].detach().numpy()
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
        img = np.hstack([real, imag])
        if row_img is None:
            row_img = img
        else:
            if width_counter <= width:
                row_img = np.hstack([row_img, img])
                width_counter += 1
            else:
                width_counter = 0
                if full_img is None:
                    full_img = row_img
                else:
                    full_img = np.vstack([full_img, row_img])
                row_img = None
    if full_img is None:
        full_img = row_img
    plt.ion()
    plt.imshow(full_img)
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
