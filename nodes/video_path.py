"""Path-only video loader for streaming nodes."""

import os

from comfy_api.latest import io
import folder_paths


class StreamVideoPath(io.ComfyNode):
    """Select a driver video without decoding frames into an IMAGE batch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamVideoPath",
            display_name="Load Video (Streaming Path)",
            category="video/stream",
            inputs=[
                io.String.Input(
                    "video_path",
                    default="",
                    tooltip="Absolute path to the driver video. Only the path is returned; frames are decoded by the streaming controller.",
                ),
            ],
            outputs=[io.String.Output(display_name="video_path")],
        )

    @classmethod
    def execute(cls, video_path):
        video_path = os.path.expanduser(video_path.strip())
        if not video_path:
            raise ValueError("A video path is required.")
        if not os.path.isabs(video_path):
            input_dir = os.path.abspath(folder_paths.get_input_directory())
            video_path = os.path.abspath(os.path.join(input_dir, video_path))
            if os.path.commonpath((input_dir, video_path)) != input_dir:
                raise ValueError("The selected video must be inside ComfyUI's input directory.")
        else:
            video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")
        return io.NodeOutput(video_path)
