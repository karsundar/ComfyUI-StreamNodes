"""Bounded CPU frame buffer."""

from collections import deque


class StreamBuffer:
    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive.")
        self.capacity = int(capacity)
        self.frames = deque(maxlen=self.capacity)

    def append(self, frame):
        self.frames.append(frame)

    def clear(self):
        self.frames.clear()

    def __len__(self):
        return len(self.frames)

    def snapshot(self):
        return list(self.frames)
