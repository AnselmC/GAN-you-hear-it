# Stdlib
import concurrent.futures
import os
import time
import sys
import logging
import argparse
from glob import glob
from shutil import rmtree
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
global_logger = logging.getLogger("Preprocessor")


def preprocess_audio(audio_folder, output_folder, target_bpm=120, num_beats=8, num_steps=8):
    """Preprocesses a folder of audio files (`.wav`, `.mp3`, `.mp4`, `.m4a`) and generates a `.npy` file with snippets of the files in Short-Time Fourier space

    :param audio_folder: Path to the folder containing `.wav` files
    :param output_folder: Pathname of file to save results to
    :param target_bpm: the BPM that each audio file should be stretched to (default: 120)
    :param num_beats: the number of beats that each sample should have (default: 8)
    :param num_steps: the number of steps to divide each sample into for STFT (default: 8)

    ..note:: Files are processed in separate threads
    """
    files = glob(audio_folder + "*.wav")
    files += glob(audio_folder + "*.mp3")
    files += glob(audio_folder + "*.mp4")
    files += glob(audio_folder + "*.m4a")
    if len(files) == 0:
        global_logger.error("Found no audio files, raising error")
        raise IOError("No audio file found in {}".format(audio_folder))
    global_logger.debug("Found {} audio files in {}".format(len(files), audio_folder))
    # preprocess files concurrently
    jobs = {}
    file_prefix = "arr_"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for f in files:
            global_logger.debug("Added {} to process queue".format(f))
            job = executor.submit(
                preprocess_single_file, *[f, target_bpm, num_beats, num_steps])
            jobs[job] = f
        i = len(glob(output_folder + "*"))
        for job in concurrent.futures.as_completed(jobs):
            data = job.result()
            file_names = [file_prefix + str(j) for j in range(i, i+len(data))]
            global_logger.debug("Saving {} to {} ... {}".format(jobs[job], file_names[0], file_names[-1])) 
            d = dict(zip(file_names, data))
            for k, v in d.items():
                np.save(os.path.join(output_folder, k), v)
            i += len(data)
            del jobs[job]

    global_logger.debug("Finished preprocessing of {} files.".format(
        len(files)))
    global_logger.info("Saving result to {}".format(
        output_folder))
    global_logger.debug("Generated {} training samples".format(i))


def preprocess_single_file(f, target_bpm, num_beats, num_steps):
    """Takes a single audio file (`.mp3`, `.mp4`, `.m4a`, `.wav`) filename, stretches it to match a target bpm, splits it into snippets of num_beats, and runs the Short-Time Fourier Transform on each snippet.

    :param f: The pathname of the audio file.
    :param target_bpm: the tempo that the track should be stretched to
    :param num_beats: the number of beats every snippet should contain
    :param num_step: the number of steps in the STFT
    :returns: A list containing the STFT matrices for each snippet
    :rtype: list(numpy.array)

    """
    n_fft = (num_steps-1) * 2
    signal, sr = librosa.load(f)
    fname = f.split("/")[-1]
    logger = logging.getLogger(fname[:5] + "..." + fname[-10:-4]) 
    logger.debug("Loaded {}".format(f))
    logger.debug("Sample rate is {}".format(sr))
    tempo, beat_frames = librosa.beat.beat_track(signal, sr=sr)
    logger.debug("Tempo was {}".format(tempo))
    stretch_factor = target_bpm/tempo
    signal = librosa.effects.time_stretch(signal, stretch_factor)
    # split signal into chunks of num_beats
    tempo, beat_frames = librosa.beat.beat_track(signal, sr=sr)
    logger.debug("Tempo now: {}".format(tempo))
    sample_frames = librosa.frames_to_samples(beat_frames)
    splits = [signal[sample_frames[i]:sample_frames[i+num_beats]-30] for i in range(0, len(sample_frames)-(1+num_beats), num_beats)]
    del signal
    logger.debug("Generated {} splits from {}".format(len(splits), f))
    # create stft for each split
    data = []
    target_len_secs = num_beats / (target_bpm/60)
    logger.debug("Target length in seconds: {}".format(target_len_secs))
    for split in splits:
        curr_len = len(split)
        logger.debug("Current length in seconds: {}".format(curr_len/sr))
        target_sr = target_len_secs * sr**2 / curr_len
        logger.debug("Target sample rate: {}".format(target_sr))
        split = librosa.resample(split, sr, target_sr, fix=True)
        if len(split) != target_len_secs * sr:
            logger.debug("Split has wrong length {}. Should be {}".format(len(split), target_len_secs * sr))
            split = split[:int(target_len_secs * sr)]
        transformed = librosa.stft(split, n_fft)
        data.append(transformed)
    logger.debug("Finished processing of {} with {} snippets".format(f, len(data)))

    return data


if __name__ == "__main__":
    print(__description__)
    parser = argparse.ArgumentParser(__description__)
    parser.add_argument("-i, --input_folder", dest="input", type=str, required=True,
                        help="The folder containing the audio files.")
    parser.add_argument("-o, --output_folder", dest="output", type=str, default="training_data",
                        help="The folder to save the training data (default is training_data)")
    parser.add_argument("-s, --num_steps", dest="steps", type=int, default=8,
                        help="The number of steps per snippet (default: 8)")
    parser.add_argument("-b, --num_beats", dest="beats", type=int, default=8,
                        help="The number of beats per snippet (default: 8)")
    parser.add_argument("-t, --tempo", dest="bpm", type=int, default=120,
                        help="The tempo given in beats per minute (bpm) that each signal should be stretched to (default: 120)")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to be verbose when logging.")

    args = parser.parse_args()
    if args.verbose:
        global_logger.setLevel(logging.DEBUG)
    else:
        global_logger.setLevel(logging.INFO)

    if os.path.exists(args.output):
        print("{} already exists. (O)verwrite/(A)ppend/(C)ancel".format(args.output))
        user_input = ""
        while user_input.lower() not in ["o", "a", "c"]:
            user_input = input()
            if user_input.lower() == "o":
                rmtree(args.output)
                os.makedirs(args.output)
            elif user_input.lower() == "a":
                print("Appending to existing folder...")
            elif user_input.lower() == "c":
                print("Exiting")
                exit()
            else:
                print("Did not understand {}. Please use O/A/C".format(user_input))
    else:
        os.makedirs(args.output)



    print("\n\nStarting preprocessing...")
    start = time.time()
    preprocess_audio(args.input, args.output,
                     args.bpm, args.beats, args.steps)
    print("Finished processing files")
    print("Processing took {:.2f} s".format(time.time() - start))
