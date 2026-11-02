"""Audio remains on disk; this node does not decode it into a tensor."""

from comfy_api.latest import io


class VideoStreamAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoStreamAudio",
            category="video/stream",
            inputs=[io.String.Input("video_path", default="")],
            outputs=[io.String.Output(display_name="audio_source")],
        )

    @classmethod
    def execute(cls, video_path):
        if not video_path:
            raise ValueError("A video path is required.")
        return io.NodeOutput(video_path)
