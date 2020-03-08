# Stdlib
import concurrent.futures
import os
import time
import sys
import logging
import argparse
import hashlib
from glob import glob
from shutil import rmtree
# Thirdparty
import librosa
import numpy as np

__description__ = """
   This script takes a folder containing audio files, stretches each song to a
   given tempo, and divides it into snippets.
   Processing is done concurrently.
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
GLOBAL_LOGGER = logging.getLogger("Preprocessor")


def preprocess_audio(audio_folder, output_folder, target_bpm=120, num_beats=8):
    """Takes an audio files directory and splits it into fixed-tempo snippets

    :param audio_folder: Path to the directory containing audio files
    :param output_folder: Directory to save snippets to
    :param target_bpm: the BPM that each audio file will be stretched to
    (default: 120)
    :param num_beats: the number of beats that each sample should have
    (default: 8)

    ..note:: Files are processed in separate threads
    """
    files = glob(audio_folder + "*.wav")
    files += glob(audio_folder + "*.mp3")
    files += glob(audio_folder + "*.mp4")
    files += glob(audio_folder + "*.m4a")
    if not files:
        GLOBAL_LOGGER.error("Found no audio files, raising error")
        raise IOError("No audio file found in {}".format(audio_folder))
    GLOBAL_LOGGER.debug("Found %d audio files in %s", len(files), audio_folder)
    # preprocess files concurrently
    jobs = {}
    num_snippets = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for fname in files:
            GLOBAL_LOGGER.debug("Added %s to process queue", fname)
            job = executor.submit(preprocess_single_file,
                                  *[fname, target_bpm, num_beats])
            jobs[job] = fname
        for job in concurrent.futures.as_completed(jobs):
            data = job.result()
            num_snippets += len(data)
            file_hash = hashlib.blake2b(
                jobs[job].split("/")[-1].encode("utf-8"), digest_size=6).hexdigest()
            file_names = [
                file_hash + "_" + f'{j:03}' for j in range(0, len(data))
            ]
            GLOBAL_LOGGER.debug("Saving %s to %s ... %s", jobs[job],
                                file_names[0], file_names[-1])
            for snippet_fname, snippet_data in zip(file_names, data):
                np.save(os.path.join(output_folder, snippet_fname),
                        snippet_data)
            del jobs[job]

    GLOBAL_LOGGER.debug("Finished preprocessing of %d files.", len(files))
    GLOBAL_LOGGER.info("Saving result to %s", output_folder)
    GLOBAL_LOGGER.debug("Generated %d training samples", num_snippets)


def preprocess_single_file(fname, target_bpm, num_beats):
    """Takes an audio file  filename, stretches it, and divides it into snippets

    :param fname: The pathname of the audio file (`.mp{3, 4}`, `.m4a`, `.wav`)
    :param target_bpm: the tempo that the track should be stretched to
    :param num_beats: the number of beats every snippet should contain
    :returns: A list containing the snippets
    :rtype: list(numpy.array)

    """
    signal, sample_rate = librosa.load(fname)
    fname = fname.split("/")[-1]
    logger = logging.getLogger(fname[:5] + "..." + fname[-10:-4])
    logger.debug("Loaded %s", fname)
    logger.debug("Sample rate is %d", sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(signal, sr=sample_rate)
    logger.debug("Tempo was: %.2f", tempo)
    stretch_factor = target_bpm / tempo
    signal = librosa.effects.time_stretch(signal, stretch_factor)
    # split signal into chunks of num_beats
    tempo, beat_frames = librosa.beat.beat_track(signal, sr=sample_rate)
    logger.debug("Tempo now: %.2f", tempo)
    sample_frames = librosa.frames_to_samples(beat_frames)
    splits = [
        signal[sample_frames[i]:sample_frames[i + num_beats] - 30]
        for i in range(0,
                       len(sample_frames) - (1 + num_beats), num_beats)
    ]
    del signal
    logger.debug("Generated %d splits from %s", len(splits), fname)
    snippets = []
    target_len_secs = num_beats / (target_bpm / 60)
    logger.debug("Target length in seconds: %d", target_len_secs)
    for split in splits:
        curr_len = len(split)
        logger.debug("Current length in seconds: %.2f", curr_len / sample_rate)
        target_sr = target_len_secs * sample_rate**2 / curr_len
        logger.debug("Target sample rate: %d", target_sr)
        split = librosa.resample(split, sample_rate, target_sr, fix=True)
        if len(split) != target_len_secs * sample_rate:
            logger.debug("Split has wrong length: %d. Should be %d",
                         len(split), target_len_secs * sample_rate)
            split = split[:int(target_len_secs * sample_rate)]
        snippets.append(split)
    logger.debug("Finished processing of %s with %d snippets", fname,
                 len(snippets))

    return snippets


def parse_args():
    """Parses command line arguments

    :returns: the parsed arguments
    """
    parser = argparse.ArgumentParser(__description__)
    parser.add_argument("-i, --input_folder",
                        dest="input",
                        type=str,
                        required=True,
                        help="The folder containing the audio files.")
    parser.add_argument(
        "-o, --output_folder",
        dest="output",
        type=str,
        default="training_data",
        help="The folder to save the training data (default is training_data)")
    parser.add_argument("-b, --num_beats",
                        dest="beats",
                        type=int,
                        default=8,
                        help="The number of beats per snippet (default: 8)")
    parser.add_argument(
        "-t, --tempo",
        dest="bpm",
        type=int,
        default=120,
        help="The tempo given in beats per minute (bpm) that each signal"
        "should be stretched to (default: 120)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Whether to be verbose when logging.")

    return parser.parse_args()


if __name__ == "__main__":
    print(__description__)
    ARGS = parse_args()
    if ARGS.verbose:
        GLOBAL_LOGGER.setLevel(logging.DEBUG)
    else:
        GLOBAL_LOGGER.setLevel(logging.INFO)

    if os.path.exists(ARGS.output):
        print("{} already exists. (O)verwrite/(A)ppend/(C)ancel?".format(
            ARGS.output))
        user_input = ""
        while user_input.lower() not in ["o", "a", "c"]:
            user_input = input()
            if user_input.lower() == "o":
                rmtree(ARGS.output)
                os.makedirs(ARGS.output)
            elif user_input.lower() == "a":
                print("Appending to existing folder...")
            elif user_input.lower() == "c":
                print("Exiting")
                exit()
            else:
                print("Did not understand {}. Please use O/A/C".format(
                    user_input))
    else:
        os.makedirs(ARGS.output)

    print("\n\nStarting preprocessing...")
    START = time.time()
    preprocess_audio(ARGS.input, ARGS.output, ARGS.bpm, ARGS.beats)
    print("Finished processing files")
    print("Processing took {:.2f} s".format(time.time() - START))
