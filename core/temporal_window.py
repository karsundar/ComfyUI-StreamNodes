"""Wan temporal frame-count validation and window planning.

Planning only: this module does not independently sample model chunks.
"""

from dataclasses import dataclass


def valid_wan_length(length):
    """Wan Animate 2 uses one initial frame plus groups of four."""
    length = int(length)
    if length < 1:
        raise ValueError("Length must be positive.")
    return ((length - 1) // 4) * 4 + 1


def latent_length(pixel_frames):
    return ((int(pixel_frames) - 1) // 4) + 1


@dataclass(frozen=True)
class TemporalWindow:
    start: int
    length: int
    frame_offset: int


def plan_windows(total_frames, window_length=81, overlap=1):
    """Produce continuation-aware window metadata.

    Does not guarantee that a model supports a particular continuation
    strategy. The caller must pass previous generated motion and the
    native video_frame_offset to Wan Animate 2.
    """
    total_frames = int(total_frames)
    window_length = valid_wan_length(window_length)
    overlap = max(1, int(overlap))

    if overlap >= window_length:
        raise ValueError("Overlap must be smaller than window length.")

    windows = []
    start = 0

    while start < total_frames:
        remaining = total_frames - start
        length = valid_wan_length(min(window_length, remaining))

        if length <= 0:
            break

        windows.append(TemporalWindow(
            start=start,
            length=length,
            frame_offset=start
        ))

        if start + length >= total_frames:
            break

        start += length - overlap

    return windows
