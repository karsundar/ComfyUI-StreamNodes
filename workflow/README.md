# Wan Animate 2 — StreamNodes adapter workflow

## Files
- `WanAnimate2Stream.json`: ComfyUI API-format workflow using the bounded controller.
- `models.txt`: exact model filenames used by the source workflow.
- `settings.txt`: preserved parameters and practical test settings.

## What changed
1. The output node is `WanAnimate2StreamController` (`246`), which owns the window loop.
2. It reads and resizes each driver window with PyAV, invokes native Wan conditioning, samples, tiled-decodes, and writes frames to FFmpeg.
3. It preserves the original model files and the original 2-step/lcm/simple/CFG-1 settings.
4. Source audio is muxed from the driver path without materializing an audio tensor.

## Installation
1. Back up your existing working ComfyUI installation and workflow. Do **not** reinstall PyTorch or ROCm.
2. Place the provided ComfyUI-StreamNodes repository folder in `ComfyUI/custom_nodes/` and restart ComfyUI.
3. In `WanAnimate2Stream.json`, edit node `246` `video_path` to the absolute driver-video path.
4. Confirm FFmpeg with `libx264` and PyAV are already available in the existing environment.
6. Import the API-format JSON using a ComfyUI workflow/API importer compatible with your installation. API JSON is not the same as a full UI graph JSON.
7. Start with a short clip and validate output before longer runs.

## Memory and quality limitations
The controller bounds memory with the active window, native sampling workspace,
tiled VAE output, and one continuation frame. A single window can still exceed
VRAM. Native continuation currently supports one overlap frame, so larger
overlap strategies are not silently approximated. Separate windows can also
have small quality differences from a single native full-clip run.

## Known caveats
- `VideoStreamLoader` is bounded only when `max_frames` is nonzero; the controller is preferred for long clips.
- Audio is synchronized by FFmpeg muxing and is limited to the encoded video duration.
- The native Wan adapter preserves conditioning behavior; it does not guarantee identical results across driver-frame counts or different runtimes.
- Your original source prompt wording is preserved unchanged, including any character-description inconsistencies.
- No changes to existing ROCm, Python packages, model files, or ComfyUI core are made by this package.
