from comfy_api.latest import io
from ..core.resolution_manager import select_resolution, resize_images


class ResolutionSelector(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamResolutionSelector",
            category="video/stream",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "mode",
                    options=["Auto Resize", "Preserve Resolution"],
                    default="Auto Resize"
                ),
                io.Int.Input("target_width", default=832, min=16, step=16),
                io.Int.Input("target_height", default=480, min=16, step=16),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, images, mode="Auto Resize",
                target_width=832, target_height=480):
        h, w = images.shape[1:3]

        out_w, out_h = select_resolution(
            w, h, target_width, target_height, mode
        )

        result = resize_images(images, out_w, out_h)

        return io.NodeOutput(result, out_w, out_h)
