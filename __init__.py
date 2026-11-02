"""ComfyUI StreamNodes extension."""

WEB_DIRECTORY = "./web"

from comfy_api.latest import ComfyExtension

from .nodes import (
    VideoStreamLoader,
    StreamVideoPath,
    VideoMetadata,
    VideoStreamAudio,
    ResolutionSelector,
    StreamOutput,
    StreamVideoWriterNode,
    StreamVideoSave,
)
from .adapters.wan_animate2 import WanAnimate2Stream
from .adapters.wan_animate2_lite import WanAnimate2LiteStream
from .nodes.stream_controller import WanAnimate2StreamController


class StreamNodesExtension(ComfyExtension):

    async def get_node_list(self):
        return [
            VideoStreamLoader,
            StreamVideoPath,
            VideoMetadata,
            VideoStreamAudio,
            ResolutionSelector,
            StreamOutput,
            StreamVideoWriterNode,
            StreamVideoSave,
            WanAnimate2Stream,
            WanAnimate2LiteStream,
            WanAnimate2StreamController,
        ]


async def comfy_entrypoint():
    return StreamNodesExtension()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
