import argparse
import time

import simpleaudio as sa
import numpy as np

__description__ = """
Plays a training snippet saved as an `npy` file.
"""


def parse_args():
    parser = argparse.ArgumentParser(__description__)
    parser.add_argument("files",
                        type=str,
                        nargs="+",
                        help="The files to play.")
    parser.add_argument("--sr",
                        type=int,
                        help="Sample rate of file (default: 22050 Hz)",
                        default=22050)
    parser.add_argument(
        "--t",
        type=int,
        help="Number of ms to sleep inbetween snippets (default: 50)",
        default=50
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(__description__)
    ARGS = parse_args()
    for fname in ARGS.files:
        if not fname.endswith(".npy"):
            break
        snippet = np.load(fname)
        snippet *= 32767 / np.max(np.abs(snippet))
        snippet = snippet.astype(np.int16)
        play_obj = sa.play_buffer(snippet, 1, 2, ARGS.sr)
        play_obj.wait_done()
        time.sleep(ARGS.t*1e-3)
