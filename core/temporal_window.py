"""Wan temporal frame-count validation and window planning.

Planning only: this module does not independently sample model chunks.
"""

from dataclasses import dataclass


def valid_wan_length(length):
    """Wan Animate 2 uses one initial frame plus groups of four."""
    length = int(length)
    if length < 1:
        raise ValueError("Length must be positive.")
    # Round up so a short final window still advances the planner.  The
    # controller's decoder may return fewer source frames for that window;
    # native Wan conditioning pads the pose clip to the requested length.
    return ((length - 1 + 3) // 4) * 4 + 1


def latent_length(pixel_frames):
    return ((int(pixel_frames) - 1) // 4) + 1


@dataclass(frozen=True)
class TemporalWindow:
    start: int
    length: int
    frame_offset: int
    output_length: int = 0


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
        requested = min(window_length, remaining + (overlap if start else 0))
        length = valid_wan_length(requested)

        if length <= 0:
            break

        windows.append(TemporalWindow(
            start=start,
            length=length,
            frame_offset=start,
            output_length=min(
                remaining - (overlap if start else 0),
                length - (overlap if start else 0),
            ),
        ))

        if start + length >= total_frames:
            break

        start += length - overlap

    return windows


def iter_windows(total_frames, window_length=81, overlap=1):
    """Yield bounded windows while preserving exact frame coverage.

    The first frame of a continuation window is the native Wan motion frame;
    it is decoded as input but is removed from the encoded output by the
    controller after native conditioning reports ``trim_image``.
    """
    for window in plan_windows(total_frames, window_length, overlap):
        yield window
