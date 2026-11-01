"""ComfyUI StreamNodes extension."""

from comfy_api.latest import ComfyExtension

from .nodes import (
    VideoStreamLoader,
    VideoMetadata,
    ResolutionSelector,
    StreamOutput,
)
from .adapters.wan_animate2 import WanAnimate2Stream
from .adapters.wan_animate2_lite import WanAnimate2LiteStream


class StreamNodesExtension(ComfyExtension):

    async def get_node_list(self):
        return [
            VideoStreamLoader,
            VideoMetadata,
            ResolutionSelector,
            StreamOutput,
            WanAnimate2Stream,
            WanAnimate2LiteStream,
        ]


async def comfy_entrypoint():
    return StreamNodesExtension()
