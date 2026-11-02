"""Finalize an already-streamed video and expose it as a ComfyUI preview."""

from __future__ import annotations

import os
import shutil

import folder_paths
from comfy_api.latest import io, ui


class StreamVideoSave(io.ComfyNode):
    """Save/preview a video path produced by the streaming controller."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamVideoSave",
            display_name="Save Stream Video (Preview)",
            category="video/stream",
            inputs=[
                io.String.Input("video_path", default=""),
                io.String.Input(
                    "output_path",
                    default="",
                    tooltip="Optional destination. Leave blank to keep the controller's saved path.",
                ),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video_path, output_path=""):
        if not video_path:
            raise ValueError("A streamed video path is required.")
        source = os.path.abspath(os.path.expanduser(video_path))
        if not os.path.isfile(source):
            raise ValueError(f"Streamed video does not exist: {source}")

        destination = source
        if output_path and output_path.strip():
            destination = os.path.abspath(os.path.expanduser(output_path.strip()))
            if destination != source:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)

        output_dir = os.path.abspath(folder_paths.get_output_directory())
        if os.path.commonpath((output_dir, destination)) != output_dir:
            raise ValueError("Preview output must be inside ComfyUI's output directory.")
        relative = os.path.relpath(destination, output_dir)
        subfolder, filename = os.path.split(relative)
        preview = ui.PreviewVideo([
            ui.SavedResult(filename, subfolder, io.FolderType.output),
        ])
        return io.NodeOutput(destination, ui=preview)
