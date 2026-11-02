"""Measure bounded decoder memory and timing on a real input video."""

from __future__ import annotations

import argparse
import os
import time

from core.frame_iterator import FrameIterator
from utils.video_decoder import inspect_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--window-size", type=int, default=81)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    metadata = inspect_video(args.video)
    peak_rss = 0
    started = time.perf_counter()
    for start in range(0, metadata["frames"], max(1, args.window_size - 1)):
        with FrameIterator(args.video, start, args.window_size,
                           resize=(args.width, args.height)) as decoder:
            count = sum(1 for _ in decoder.frames())
        try:
            import psutil
            peak_rss = max(peak_rss, psutil.Process(os.getpid()).memory_info().rss)
        except ImportError:
            pass
        if not count:
            break
    print({"metadata": metadata, "seconds": time.perf_counter() - started,
           "peak_rss_bytes": peak_rss})


if __name__ == "__main__":
    main()
