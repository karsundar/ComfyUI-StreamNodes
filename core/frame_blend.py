"""Small boundary blend used to soften streaming window transitions."""

import torch


def blend_boundary(previous_tail, current_frames, frame_count):
    """Crossfade the start of a window with the previous output tail."""
    frame_count = min(
        max(0, int(frame_count)),
        int(previous_tail.shape[0]),
        int(current_frames.shape[0]),
    )
    if frame_count == 0:
        return current_frames

    previous_tail = previous_tail[-frame_count:].to(
        device=current_frames.device,
        dtype=current_frames.dtype,
    )
    alpha = torch.linspace(
        0.0, 1.0, frame_count,
        device=current_frames.device,
        dtype=current_frames.dtype,
    ).view(frame_count, 1, 1, 1)
    blended = current_frames.clone()
    blended[:frame_count] = (
        previous_tail * (1.0 - alpha) + current_frames[:frame_count] * alpha
    )
    return blended
