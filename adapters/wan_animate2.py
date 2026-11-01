"""Thin adapter around ComfyUI's native WanAnimate2ToVideo node.

Conditioning, VAE encoding, masks, pose conditioning, latent creation,
and frame-offset bookkeeping remain owned by ComfyUI.
"""

from comfy_api.latest import io
from comfy_extras.nodes_wan import WanAnimate2ToVideo


class WanAnimate2Stream(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanAnimate2Stream",
            display_name="Wan Animate 2 Stream",
            category="video/stream/wan",
            description=(
                "Streaming integration using ComfyUI's native Wan Animate 2 "
                "conditioning implementation. This node does not replace "
                "the native sampler."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, step=16),
                io.Int.Input("height", default=480, min=16, step=16),
                io.Int.Input("length", default=81, min=1, step=4),
                io.Int.Input("batch_size", default=1, min=1),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Conditioning.Input("positive_pose", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_pose", optional=True),
                io.Image.Input("continue_motion", optional=True),
                io.Int.Input("video_frame_offset", default=0, min=0),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0),
                io.Float.Input("pose_start_percent", default=0.0, min=0.0, max=1.0),
                io.Float.Input("pose_end_percent", default=1.0, min=0.0, max=1.0),
                io.Float.Input("reference_image_strength", default=1.0, min=0.0, max=10.0),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="video_frame_offset"),
            ],
        )

    @classmethod
    def execute(
        cls, positive, negative, vae, width, height, length, batch_size,
        reference_image=None, pose_video=None, clip_vision_output=None,
        positive_pose=None, clip_vision_output_pose=None,
        continue_motion=None, video_frame_offset=0,
        pose_strength=1.0, pose_start_percent=0.0,
        pose_end_percent=1.0, reference_image_strength=1.0
    ):
        return WanAnimate2ToVideo.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            reference_image=reference_image,
            pose_video=pose_video,
            clip_vision_output=clip_vision_output,
            positive_pose=positive_pose,
            clip_vision_output_pose=clip_vision_output_pose,
            continue_motion=continue_motion,
            video_frame_offset=video_frame_offset,
            pose_strength=pose_strength,
            pose_start_percent=pose_start_percent,
            pose_end_percent=pose_end_percent,
            reference_image_strength=reference_image_strength,
        )
