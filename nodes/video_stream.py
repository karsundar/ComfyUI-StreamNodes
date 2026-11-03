"""Video loader node."""

import os  # Fixed: import at top of file
import torch
from comfy_api.latest import io
from ..core.frame_iterator import FrameIterator
from ..utils.video_decoder import inspect_video
from functools import lru_cache

# ROCm/GPU optimization for AMD GPUs
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
else:
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Dynamic thread count
    os.environ['OMP_NUM_WAIT_POLICY'] = 'PASSIVE'  # Reduce CPU contention

class VideoStreamLoader(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoStreamLoader",
            category="video/stream",
            inputs=[
                io.String.Input("video_path", default=""),
                io.Int.Input("start_frame", default=0, min=0),
                io.Int.Input("max_frames", default=81, min=1,
                             tooltip="Maximum frames to decode; keep this bounded."),
                io.Int.Input("resize_width", default=0, min=0, step=16,
                             tooltip="0 keeps source width; resize occurs during decode."),
                io.Int.Input("resize_height", default=0, min=0, step=16,
                             tooltip="0 keeps source height; resize occurs during decode."),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="source_width"),
                io.Int.Output(display_name="source_height"),
                io.Int.Output(display_name="decoded_frames"),
            ],
        )

    @classmethod
    def execute(cls, video_path, start_frame=0, max_frames=0,
                resize_width=0, resize_height=0):
        if not video_path:
            raise ValueError("A video path is required.")

        max_frames = int(max_frames)
        if max_frames < 1:
            raise ValueError("max_frames must be positive so the loader remains bounded.")

        if start_frame >= max_frames:
            raise ValueError("start_frame must be less than max_frames")

        metadata = inspect_video(video_path)
        resize = (resize_width, resize_height) if resize_width and resize_height else None

        # Device-aware allocation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        images = None
        decoded = 0
        with FrameIterator(video_path, start_frame, max_frames, resize=resize) as decoder:
            for frame in decoder.frames():
                if images is None:
                    # Allocate the bounded output once instead of retaining a
                    # Python list and then creating a second copy via stack().
                    images = torch.empty(
                        (max_frames, *frame.shape),
                        dtype=frame.dtype,
                        device=device,
                        pin_memory=True  # For GPU transfer optimization
                    )
                images[decoded].copy_(frame)
                decoded += 1

        if images is None or decoded == 0:
            raise ValueError("No video frames were decoded.")

        # ComfyUI IMAGE convention: [T,H,W,C], float32, range 0..1.
        images = images[:decoded]

        # Add profiling for performance analysis
        with torch.profiler.profile(profile_memory=True) as prof:
            pass  # Add this around the critical section

        return io.NodeOutput(
            images,
            metadata["fps"],
            metadata["width"],
            metadata["height"],
            decoded,
        )
