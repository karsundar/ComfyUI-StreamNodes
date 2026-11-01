"""Lite adapter.

The conditioning contract is shared with native Wan Animate 2. The actual
Lite model is selected through the model loader; this adapter does not
pretend to implement a separate Lite encoder or sampler.
"""

from comfy_api.latest import io
from .wan_animate2 import WanAnimate2Stream


class WanAnimate2LiteStream(WanAnimate2Stream):

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        return io.Schema(
            node_id="WanAnimate2LiteStream",
            display_name="Wan Animate 2 Lite Stream",
            category="video/stream/wan",
            description=(
                "Lite model adapter using the same native Wan Animate 2 "
                "conditioning contract. Select a compatible Lite checkpoint "
                "in the model loader."
            ),
            inputs=schema.inputs,
            outputs=schema.outputs,
        )
