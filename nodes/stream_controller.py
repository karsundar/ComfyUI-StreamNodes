"""Single-node Wan Animate 2 orchestration for bounded-memory generation."""

from __future__ import annotations

import gc
import logging
import math
import os
import re

import torch
import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import io, ui
from comfy_extras.nodes_wan import WanAnimate2ToVideo
from nodes import VAEDecodeTiled, common_ksampler

from ..core.frame_iterator import FrameIterator
from ..core.frame_blend import blend_boundary
from ..core.stream_writer import StreamVideoWriter
from ..core.temporal_window import iter_windows
from ..utils.video_decoder import inspect_video


def _resolve_compute_device(requested, model):
    """Resolve the device used by tensors created inside the stream loop.

    Model placement remains owned by ComfyUI's model loader and sampler.  In
    ``auto`` mode, follow the model patcher's load device when available;
    otherwise use ComfyUI's configured torch device.
    """
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("compute_device must be one of: auto, gpu, cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "auto":
        model_device = getattr(model, "load_device", None)
        if model_device is not None:
            device = torch.device(model_device)
        else:
            device = model_management.get_torch_device()
    else:
        device = model_management.get_torch_device()

    if requested == "gpu" and model_management.is_device_cpu(device):
        raise RuntimeError(
            "compute_device='gpu' was requested, but ComfyUI reports no "
            "usable accelerator. Start ComfyUI with a GPU-enabled PyTorch build."
        )
    return device

LOGGER = logging.getLogger("ComfyUI-StreamNodes")


class WanAnimate2StreamController(io.ComfyNode):
    """Run native Wan conditioning/sampling one driver window at a time."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanAnimate2StreamController",
            display_name="Wan Animate 2 Stream Controller",
            category="video/stream/wan",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"), io.Vae.Input("vae"),
                io.String.Input("video_path", default=""),
                io.String.Input("output_suffix", default="_wan_animate_stream",
                                tooltip="Suffix appended to the source filename."),
                io.Int.Input("width", default=832, min=16, step=16),
                io.Int.Input("height", default=480, min=16, step=16),
                io.Int.Input("window_size", default=81, min=1, step=4),
                io.Int.Input("window_overlap", default=1, min=1),
                io.Float.Input(
                    "duration_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Process only this many seconds from the start; 0 processes the entire video.",
                ),
                io.Int.Input("boundary_blend_frames", default=0, min=0, step=1,
                             tooltip="Crossfade this many frames at each streaming window boundary; 0 disables blending."),
                io.Int.Input("steps", default=2, min=1),
                io.Float.Input("cfg", default=1.0, min=0.0),
                io.String.Input("sampler_name", default="lcm"),
                io.String.Input("scheduler", default="simple"),
                io.Int.Input("seed", default=0, min=0),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0),
                io.Image.Input("reference_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Conditioning.Input("positive_pose", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_pose", optional=True),
                io.Float.Input("pose_strength", default=1.0, min=0.0),
                io.Float.Input("pose_start_percent", default=0.0, min=0.0, max=1.0),
                io.Float.Input("pose_end_percent", default=1.0, min=0.0, max=1.0),
                io.Float.Input("reference_image_strength", default=1.0, min=0.0),
                io.String.Input("audio_mode", default="source"),
                io.Int.Input("tile_size", default=640, min=64, step=32),
                io.Int.Input("tile_overlap", default=64, min=0, step=32),
                io.Int.Input("temporal_size", default=64, min=8, step=4),
                io.Int.Input("temporal_overlap", default=16, min=4, step=4),
                io.String.Input("video_codec", default="h264"),
                io.Combo.Input(
                    "compute_device",
                    options=["auto", "gpu", "cpu"],
                    default="auto",
                    tooltip=(
                        "Device for StreamNodes-created tensors. Auto follows "
                        "the ComfyUI model device; GPU requires an available accelerator."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[io.String.Output(display_name="output_path"),
                     io.Int.Output(display_name="frames_written")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, positive, negative, vae, video_path, output_suffix,
                width, height, window_size, window_overlap, duration_seconds,
                boundary_blend_frames, steps, cfg,
                sampler_name, scheduler, seed, denoise, reference_image=None,
                clip_vision_output=None, positive_pose=None,
                clip_vision_output_pose=None, pose_strength=1.0,
                pose_start_percent=0.0, pose_end_percent=1.0,
                reference_image_strength=1.0, audio_mode="source",
                tile_size=640, tile_overlap=64, temporal_size=64,
                temporal_overlap=16, video_codec="h264", compute_device="auto"):
        if not video_path:
            raise ValueError("A driver video path is required.")
        if window_overlap != 1:
            raise ValueError("Wan Animate 2 native continuation supports exactly one overlap frame.")
        compute_device = _resolve_compute_device(compute_device, model)
        LOGGER.info(
            "Stream controller: compute_device=%s model_load_device=%s",
            compute_device,
            getattr(model, "load_device", "managed by ComfyUI"),
        )
        metadata = inspect_video(video_path)
        if metadata["frames"] <= 0:
            raise ValueError("The driver video has no reliable frame count.")
        duration_seconds = float(duration_seconds or 0.0)
        if duration_seconds < 0.0:
            raise ValueError("duration_seconds must be zero or positive.")
        source_frames = metadata["frames"]
        frames_to_process = source_frames
        if duration_seconds > 0.0:
            frames_to_process = min(
                source_frames,
                max(1, int(math.ceil(duration_seconds * metadata["fps"]))),
            )
        LOGGER.info(
            "Stream controller: input=%s source_frames=%d processing_frames=%d duration=%.3fs fps=%.3f source=%dx%d window=%d output=%dx%d steps=%d",
            video_path, source_frames, frames_to_process, duration_seconds,
            metadata["fps"], metadata["width"], metadata["height"], window_size,
            width, height, steps,
        )
        source_stem = os.path.splitext(os.path.basename(video_path))[0]
        suffix = str(output_suffix or "_wan_animate_stream").strip()
        if not suffix:
            suffix = "_wan_animate_stream"
        if not suffix.startswith("_"):
            suffix = "_" + suffix
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
        output_subfolder = "video"
        output_name = f"{source_stem}{suffix}.mp4"
        output = os.path.abspath(os.path.join(
            folder_paths.get_output_directory(), output_subfolder, output_name,
        ))
        audio_path = video_path if audio_mode.lower() in ("source", "copy", "transcode") else None
        previous_motion = None
        previous_tail = None
        written = 0

        with StreamVideoWriter(output, metadata["fps"] or 30.0, video_codec, audio_path) as writer:
            for index, window in enumerate(iter_windows(frames_to_process, window_size, window_overlap)):
                LOGGER.info(
                    "Stream controller: window %d start=%d length=%d output=%d decoding",
                    index + 1, window.start, window.length, window.output_length,
                )
                with FrameIterator(video_path, window.start, window.length,
                                   resize=(width, height)) as decoder:
                    pose_frames = list(decoder.frames())
                if not pose_frames:
                    raise ValueError(f"No frames decoded for window {index}.")
                LOGGER.info(
                    "Stream controller: window %d decoded %d frames; preparing conditioning",
                    index + 1, len(pose_frames),
                )
                pose_video = torch.stack(pose_frames, dim=0).to(compute_device)
                del pose_frames
                # The pose tensor is window-local.  Native continuation uses
                # offset 1 to consume its first (overlap) frame; the global
                # source position is represented by the decoder's start.
                native_offset = 1 if previous_motion is not None else 0
                conditioning = WanAnimate2ToVideo.execute(
                    positive=positive, negative=negative, vae=vae,
                    width=width, height=height, length=window.length,
                    batch_size=1, reference_image=reference_image,
                    pose_video=pose_video, clip_vision_output=clip_vision_output,
                    positive_pose=positive_pose,
                    clip_vision_output_pose=clip_vision_output_pose,
                    continue_motion=previous_motion,
                    video_frame_offset=native_offset,
                    pose_strength=pose_strength,
                    pose_start_percent=pose_start_percent,
                    pose_end_percent=pose_end_percent,
                    reference_image_strength=reference_image_strength,
                )
                LOGGER.info(
                    "Stream controller: window %d conditioning ready; sampling",
                    index + 1,
                )
                sampled = common_ksampler(
                    model, seed + index, steps, cfg, sampler_name, scheduler,
                    conditioning[0], conditioning[1], conditioning[2], denoise=denoise,
                )[0]
                LOGGER.info(
                    "Stream controller: window %d sampling complete; decoding VAE output",
                    index + 1,
                )
                trim_latent = int(conditioning[3])
                latent = sampled.copy()
                latent["samples"] = sampled["samples"][:, :, trim_latent:]
                decoded = VAEDecodeTiled().decode(
                    vae, latent, tile_size, tile_overlap,
                    temporal_size, temporal_overlap,
                )[0]
                LOGGER.info(
                    "Stream controller: window %d VAE decode complete; writing frames",
                    index + 1,
                )
                trim_image = 1 if previous_motion is not None else 0
                frames_to_write = decoded[trim_image:trim_image + window.output_length]
                if previous_tail is not None and boundary_blend_frames:
                    frames_to_write = blend_boundary(
                        previous_tail, frames_to_write, boundary_blend_frames,
                    )
                    LOGGER.info(
                        "Stream controller: window %d blended %d boundary frames",
                        index + 1,
                        min(boundary_blend_frames, len(previous_tail), len(frames_to_write)),
                    )
                for frame in frames_to_write:
                    writer.append(frame)
                written += len(frames_to_write)
                LOGGER.info(
                    "Stream controller: window %d complete; total frames written=%d",
                    index + 1, written,
                )
                previous_motion = frames_to_write[-1:].detach().to(compute_device)
                if boundary_blend_frames:
                    previous_tail = frames_to_write[-boundary_blend_frames:].detach().to(compute_device)
                del pose_video, conditioning, sampled, latent, decoded, frames_to_write
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        preview = ui.PreviewVideo([
            ui.SavedResult(output_name, output_subfolder, io.FolderType.output),
        ])
        return io.NodeOutput(output, written, ui=preview)
