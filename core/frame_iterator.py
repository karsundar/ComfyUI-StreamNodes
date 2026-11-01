"""Incremental PyAV video frame decoding."""

import av
import numpy as np
import torch


class FrameIterator:
    def __init__(self, path, start=0, limit=0):
        self.path = path
        self.start = max(0, int(start))
        self.limit = max(0, int(limit))
        self.container = None
        self.stream = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0

    def __enter__(self):
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.fps = float(self.stream.average_rate or 0)
        self.total_frames = self.stream.frames or 0
        return self

    def __exit__(self, *args):
        if self.container:
            self.container.close()

    def frames(self):
        emitted = 0
        for index, frame in enumerate(self.container.decode(self.stream)):
            if index < self.start:
                continue

            if self.limit and emitted >= self.limit:
                break

            rgb = frame.to_ndarray(format="rgb24").copy()
            yield torch.from_numpy(rgb).float().div_(255.0)
            emitted += 1
