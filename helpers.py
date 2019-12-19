# Stdlib
import concurrent.futures
import logging
from glob import glob
# Thirdparty
import librosa
import torch
import numpy as np


def get_stft_shape(sample_rate, snippet_length, time_steps):
    """ Gets the shape for the Short-Time Fourier Transform matrix corresponding to a sample_rate, snippet_length, and time_steps
    :param sample_rate: the sample rate of the time signal in Hz
    :param snippet_length: the length of the signal in seconds
    :param time_steps: the number of time steps that the signal should be split into

    :returns: the shape of the matrix with dim time steps times number of frequencies
    :rtype: tuple(int, int)
    """
    return (time_steps, np.ceil(sample_rate*2*snippet_length/(time_steps-1)).astype(int))


def save_as_time_signal(stft_signal, output_file, sr=22050):
    """Saves a signal in Short-Time Fourier space to file

    :param stft_signal: The 2-dimensional signal (time steps, number of frequencies) as `numpy.array` or `torch.Tensor`
    :param output_file: The path where to save the file to as a string
    :param sr: The sample rate of the signal (default 22050)

    """
    if type(stft_signal) is torch.Tensor:
        logging.debug("Converting signal from `torch.Tensor` to numpy array")
        stft_signal = stft_signal.detach().numpy()
    time_signal = librosa.istft(stft_signal)
    librosa.output.write(output_file, time_signal, sr)


def preprocess_audio(audio_folder, output_file, snippet_length=10, time_steps=65):
    """Preprocesses a folder of `.wav` files and generates a `.npy` file with snippets of the files in Short-Time Fourier space

    :param audio_folder: Path to the folder containing `.wav` files
    :param output_file: Pathname of file to save results to
    :param snippet_length: Each file will be split into n snippets of length `snippet_length` in seconds (default: 10)
    :param time_steps: The number of time steps to split each snippet into in Short-Time Fourier space (default: 65)

    ..note:: Files are processed in separate processes
    """
    files = glob(audio_folder + "*.wav")
    if len(files) == 0:
        logging.error("Found no .wav files, raising error")
        raise IOError("No .wav file found in {}".format(audio_folder))
    logging.debug("Found {} .wav files in {}".format(len(files), audio_folder))
    n_fft = (time_steps - 1) * 2
    # preprocess files concurrently
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for f in files:
            logging.debug("Added {} to process queue".format(f))
            futures.append(executor.submit(
                preprocess_single_file, *[snippet_length, n_fft]))
    concurrent.futures.wait(futures)
    data = []
    for f in futures:
        data.append(f)

    logging.debug("Finished preprocessing of {} files.\nSaving result to {}".format(
        len(files), output_file))
    np.save(output_file, data)


def preprocess_single_file(f, snippet_length, n_fft):
    """Takes a single `.wav` filename, splits it into snippets and runs the Short-Time Fourier Transform on each snippet.

    :param f: The pathname of the `.wav` file.
    :param snippet_length: The length of each snippet in seconds.
    :param n_fft: The length of the windowed signal (see <https://librosa.github.io/librosa/generated/librosa.core.stft.html>)
    :returns: A list containing the STFT matrices for each snippet
    :rtype: list(numpy.array)

    ..note:: Each snippet is transformed in a separate thread.
    """
    signal, sr = librosa.load(f)
    # cutoff first and last ten seconds that may contain less music
    signal = signal[10*sr:-10*sr]
    # split signal into chunks of snippet_length
    splits = [signal[i:i+10*sr] for i in np.arange(0, len(signal), 10*sr)]
    logging.debug("Generated {} splits from {}".format(len(splits), f))
    # create stft for each split concurrently
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, split in enumerate(splits):
            logging.debug(
                "Adding split no. {} of {} to thread queue".format(i, f))
            futures.append(executor.submit(
                librosa.stft, **{"y": split, "n_fft": n_fft}))

    concurrent.futures.wait(futures)
    logging.debug("Finished processing of {}".format(f))

    return [f.result() for f in futures]
