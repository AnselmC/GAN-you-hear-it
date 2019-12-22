# Stdlib
import concurrent.futures
import time
import sys
import logging
import argparse
from glob import glob
# Thirdparty
import librosa
import numpy as np

__description__ = """
   This script takes a folder containing audio files, slices each file into snippets, and generates a `.npz` archive.
   This archive contains one `.npy` file per training sample.
   Each training sample is the Short-Time Fourier space representation of a snippet of a song.
   Processing is done concurrently.
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("Preprocessor")


def preprocess_audio(audio_folder, output_file, snippet_length=10, time_steps=65):
    """Preprocesses a folder of audio files (`.wav`, `.mp3`, `.mp4`, `.m4a`) and generates a `.npy` file with snippets of the files in Short-Time Fourier space

    :param audio_folder: Path to the folder containing `.wav` files
    :param output_file: Pathname of file to save results to
    :param snippet_length: Each file will be split into n snippets of length `snippet_length` in seconds (default: 10)
    :param time_steps: The number of time steps to split each snippet into in Short-Time Fourier space (default: 65)

    ..note:: Files are processed in separate processes
    """
    files = glob(audio_folder + "*.wav")
    files += glob(audio_folder + "*.mp3")
    files += glob(audio_folder + "*.mp4")
    files += glob(audio_folder + "*.m4a")
    if len(files) == 0:
        logger.error("Found no .wav files, raising error")
        raise IOError("No .wav file found in {}".format(audio_folder))
    logger.debug("Found {} .wav files in {}".format(len(files), audio_folder))
    n_fft = (time_steps - 1) * 2
    # preprocess files concurrently
    jobs = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for f in files:
            logger.debug("Added {} to process queue".format(f))
            job = executor.submit(
                preprocess_single_file, *[f, snippet_length, n_fft])
            jobs[job] = f
    data = []
    for job in concurrent.futures.as_completed(jobs):
        data += job.result()
        del jobs[job]

    logger.debug("Finished preprocessing of {} files.".format(
        len(files)))
    logger.info("Saving result to {}".format(
        output_file))
    logger.debug("Generated {} training samples".format(len(data)))
    np.savez_compressed(output_file, *data)


def preprocess_single_file(f, snippet_length, n_fft):
    """Takes a single audio file (`.mp3`, `.mp4`, `.m4a`, `.wav`) filename, splits it into snippets and runs the Short-Time Fourier Transform on each snippet.

    :param f: The pathname of the audio file.
    :param snippet_length: The length of each snippet in seconds.
    :param n_fft: The length of the windowed signal (see <https://librosa.github.io/librosa/generated/librosa.core.stft.html>)
    :returns: A list containing the STFT matrices for each snippet
    :rtype: list(numpy.array)

    ..note:: Each snippet is transformed in a separate thread.
    """
    signal, sr = librosa.load(f)
    logger.debug("Loaded {}".format(f))
    logger.debug("Sample rate is {}".format(sr))
    # cutoff first and last ten seconds that may contain less music
    signal = signal[10*sr:-10*sr]
    # split signal into chunks of snippet_length
    splits = [signal[i:i+10*sr] for i in np.arange(0, len(signal), 10*sr)]
    del signal
    logger.debug("Generated {} splits from {}".format(len(splits), f))
    # create stft for each split concurrently
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i, split in enumerate(splits):
            logger.debug(
                "Adding split no. {} of {} to thread queue".format(i, f))
            job = executor.submit(
                librosa.stft, **{"y": split, "n_fft": n_fft})
            jobs[job] = split

    data = []
    for job in concurrent.futures.as_completed(jobs):
        data.append(job.result())
        del jobs[job]
    logger.debug("Finished processing of {}".format(f))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__description__)
    parser.add_argument("-i, --input_folder", dest="input", type=str, required=True,
                        help="The folder containing the audio files.")
    parser.add_argument("-o, --output_file", dest="output", type=str, default="train",
                        help="The npz archivename for the generated output (default is train)")
    parser.add_argument("-sl, --snippet_length", dest="snippet_length", type=int, default=10,
                        help="The length of each slice in seconds (default is 10)")
    parser.add_argument("-ts, --time_steps", dest="time_steps", type=int, default=65,
                        help="The number of steps to use in the STFT (value minus 1 times 2 should be power of 2 for optimal performance")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to be verbose when logging.")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


    print(__description__)
    print("\n\nStarting preprocessing...")
    start = time.time()
    preprocess_audio(args.input, args.output,
                     args.snippet_length, args.time_steps)
    print("Finished processing files")
    print("Processing took {:.2f} s".format(time.time() - start))
