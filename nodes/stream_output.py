from comfy_api.latest import io
from ..core.memory_manager import memory_report


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
