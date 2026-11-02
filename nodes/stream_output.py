from comfy_api.latest import io
from ..core.memory_manager import memory_report
from ..core.stream_writer import StreamVideoWriter


class StreamOutput(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamOutput",
            category="video/stream",
            inputs=[io.Image.Input("images")],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="memory_report"),
            ],
        )

    @classmethod
    def execute(cls, images):
        report = memory_report()
        text = ", ".join(f"{k}={v}" for k, v in report.items())
        return io.NodeOutput(images, text)


class StreamVideoWriterNode(io.ComfyNode):
    """Encode an IMAGE batch incrementally; the controller avoids this edge."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamVideoWriter",
            category="video/stream",
            inputs=[io.Image.Input("images"),
                    io.String.Input("output_path", default="video/stream.mp4"),
                    io.Float.Input("fps", default=30.0, min=1.0),
                    io.String.Input("codec", default="h264"),
                    io.String.Input("audio_source", default="")],
            outputs=[io.String.Output(display_name="output_path"),
                     io.Int.Output(display_name="frames_written")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, output_path, fps=30.0, codec="h264", audio_source=""):
        with StreamVideoWriter(output_path, fps, codec, audio_source or None) as writer:
            for frame in images:
                writer.append(frame)
            count = writer.frame_count
        return io.NodeOutput(output_path, count)
