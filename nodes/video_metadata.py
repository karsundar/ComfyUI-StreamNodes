from comfy_api.latest import io
from ..utils.video_decoder import inspect_video


class VideoMetadata(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoMetadata",
            category="video/stream",
            inputs=[io.String.Input("video_path", default="")],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="fps_numerator"),
                io.Int.Output(display_name="fps_denominator"),
                io.Int.Output(display_name="frames"),
                io.Float.Output(display_name="duration_seconds"),
                io.Boolean.Output(display_name="variable_frame_rate"),
            ],
        )

    @classmethod
    def execute(cls, video_path):
        data = inspect_video(video_path)
        return io.NodeOutput(
            data["width"], data["height"], data["fps"],
            data["fps_numerator"], data["fps_denominator"],
            data["frames"], data["duration"], data["variable_frame_rate"]
        )
