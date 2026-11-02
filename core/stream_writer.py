"""Incremental FFmpeg writer used by the controller and output node."""

from __future__ import annotations

import os
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("ComfyUI-StreamNodes")


class StreamVideoWriter:
    """Encode frames as they arrive; no output frame list is retained."""

    def __init__(self, output_path, fps, codec="h264", audio_path=None,
                 crf=18):
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.codec = codec.lower()
        self.audio_path = audio_path or ""
        self.crf = int(crf)
        self._process = None
        self._temporary_video = None
        self._shape = None
        self.frame_count = 0

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=".video.mp4", dir=self.output_path.parent)
        os.close(fd)
        self._temporary_video = Path(name)
        return self

    def append(self, frame):
        if self._temporary_video is None:
            raise RuntimeError("StreamVideoWriter must be entered before append().")
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().to("cpu")
            if frame.ndim == 4:
                if frame.shape[0] != 1:
                    raise ValueError("append() accepts one frame at a time")
                frame = frame[0]
            array = frame.float().clamp(0, 1).mul(255).byte().numpy()
        else:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Frames must have shape [height, width, 3].")
        height, width = array.shape[:2]
        if self._shape is None:
            self._shape = (width, height)
            encoder = "libx264" if self.codec in ("h264", "auto") else self.codec
            self._process = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", encoder,
                 "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(self._temporary_video)], stdin=subprocess.PIPE)
        elif self._shape != (width, height):
            raise ValueError("All frames must have identical dimensions.")
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg stopped while encoding frames") from exc
        self.frame_count += 1

    def close(self, success=True):
        process = self._process
        temporary = self._temporary_video
        self._process = None
        self._temporary_video = None
        try:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                return_code = process.wait()
                if return_code:
                    raise RuntimeError(f"FFmpeg video encoding failed ({return_code})")
            if not success:
                return
            if temporary is None or self.frame_count == 0:
                raise ValueError("Cannot finalize a video without frames.")
            if self.audio_path:
                self._mux_audio(temporary)
            else:
                os.replace(temporary, self.output_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _mux_audio(self, temporary):
        # Use a distinct, unique path.  Reusing a suffix-derived path can
        # collide with an interrupted FFmpeg process or a stale mux file.
        fd, muxed_name = tempfile.mkstemp(
            suffix=".muxed.mp4", dir=str(temporary.parent),
        )
        os.close(fd)
        muxed = Path(muxed_name)
        muxed.unlink()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(temporary), "-i", self.audio_path,
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                   "-c:a", "copy", "-shortest", "-movflags", "+faststart",
                   str(muxed)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            audio_codec = command.index("-c:a") + 1
            command[audio_codec] = "aac"
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            # The video stream is already complete. Preserve it rather than
            # turning a non-essential audio failure into a failed workflow.
            LOGGER.warning(
                "Audio mux failed; preserving video without audio: %s",
                result.stderr.strip() or "unknown FFmpeg error",
            )
            os.replace(temporary, self.output_path)
            return
        os.replace(muxed, self.output_path)
        if muxed.exists():
            muxed.unlink()

    def __exit__(self, exc_type, exc, traceback):
        self.close(success=exc_type is None)
