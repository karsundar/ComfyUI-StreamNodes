"""Bounded, seekable PyAV video frame decoding."""

import av
import numpy as np
import torch
import torch.nn.functional as F


class FrameIterator:
    def __init__(self, path, start=0, limit=0, resize=None):
        self.path = path
        self.start = max(0, int(start))
        self.limit = max(0, int(limit))
        self.container = None
        self.stream = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.resize = resize

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
        # PyAV's frame index is the reliable boundary for VFR and codecs whose
        # keyframe seek lands before the requested frame.  The iterator still
        # has bounded output: at most ``limit`` frames are yielded.
        for index, frame in enumerate(self.container.decode(self.stream)):
            if index < self.start:
                continue

            if self.limit and emitted >= self.limit:
                break

            rgb = frame.to_ndarray(format="rgb24").copy()
            image = torch.from_numpy(rgb).float().div_(255.0)
            if self.resize is not None:
                width, height = self.resize
                image = F.interpolate(
                    image.permute(2, 0, 1).unsqueeze(0),
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0).clamp_(0, 1)
            yield image
            emitted += 1
