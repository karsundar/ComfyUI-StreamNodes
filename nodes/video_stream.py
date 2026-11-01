"""Video loader node."""

from comfy_api.latest import io
from ..core.frame_iterator import FrameIterator
from ..utils.video_decoder import inspect_video
import torch


class VideoStreamLoader(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoStreamLoader",
            category="video/stream",
            inputs=[
                io.String.Input("video_path", default=""),
                io.Int.Input("start_frame", default=0, min=0),
                io.Int.Input("max_frames", default=0, min=0,
                             tooltip="0 means decode all remaining frames."),
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
    def execute(cls, video_path, start_frame=0, max_frames=0):
        if not video_path:
            raise ValueError("A video path is required.")

        metadata = inspect_video(video_path)
        frames = []

        with FrameIterator(video_path, start_frame, max_frames) as decoder:
            for frame in decoder.frames():
                frames.append(frame)

        if not frames:
            raise ValueError("No video frames were decoded.")

        # ComfyUI IMAGE convention: [T,H,W,C], float32, range 0..1.
        images = torch.stack(frames, dim=0)

        return io.NodeOutput(
            images,
            metadata["fps"],
            metadata["width"],
            metadata["height"],
            len(frames),
        )
