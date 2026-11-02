# ComfyUI-StreamNodes

Independent ComfyUI extension for video input management and native
Wan Animate 2 conditioning integration.

## Nodes

- VideoStreamLoader (bounded window decoder)
- VideoMetadata
- StreamResolutionSelector
- StreamOutput
- StreamVideoSave (save and preview an already-streamed video path)
- VideoStreamAudio (path-only audio handoff)
- StreamVideoWriter (incremental FFmpeg encoder)
- WanAnimate2Stream
- WanAnimate2LiteStream
- WanAnimate2StreamController (end-to-end controller)

`Load Video (Streaming Path)` provides a video picker and browser preview. It
uploads the selected file to ComfyUI's input directory and passes only the
server-side path to the streaming controller; it does not materialize an
IMAGE batch. The controller saves output as
`output/video/<source-name><suffix>.mp4` and publishes it as a native ComfyUI
video preview on the controller node.

## Important memory behavior

VideoStreamLoader returns only the requested window. Set `max_frames` to a
bounded value and optionally set resize dimensions; resizing happens while
decoding, before the window tensor is assembled.

The Wan adapters delegate conditioning and latent construction to
ComfyUI's native WanAnimate2ToVideo implementation. The controller invokes
that same native node for every window and writes frames directly to FFmpeg.

The Lite adapter shares the native conditioning contract. Select a
compatible Lite checkpoint in the model loader.

## Temporal behavior

Wan Animate 2 uses temporal compression of four and frame lengths
following 4n+1. The temporal planner is informational; it does not
independently sample chunks. For continuation, use the native
    continue_motion and video_frame_offset inputs. Native continuation
    currently supports one overlap frame; larger overlap values are rejected.

## Resolution

Auto Resize is the default. Preserve Resolution retains source dimensions.
The native Wan node still validates dimensions and model constraints.

## Installation

PyAV and FFmpeg with `libx264` must be available in the existing ComfyUI
environment. The extension does not install or alter PyTorch, ROCm, ComfyUI,
or model files. Use the controller for long videos because ordinary IMAGE
edges materialize complete tensors.

The direct Python dependencies are listed in [`requirements.ini`](./requirements.ini).
Install the entries in its `[python]` section with the same interpreter that
launches ComfyUI, for example:

```bash
python -m pip install 'av>=12.0' 'numpy>=1.24'
```

PyTorch and ComfyUI must come from the existing ComfyUI environment. Install
FFmpeg through the operating system or your environment manager and verify
that `ffmpeg -encoders` lists `libx264`.

Run the core tests from this directory:

```bash
python -m unittest discover -s tests -v
```

`benchmark_streaming.py` measures decode timing and optional process RSS on a
real input. It does not claim that one Wan sampling window cannot OOM.
