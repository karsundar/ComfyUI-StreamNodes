# ComfyUI-StreamNodes

Independent ComfyUI extension for video input management and native
Wan Animate 2 conditioning integration.

## Nodes

- VideoStreamLoader
- VideoMetadata
- StreamResolutionSelector
- StreamOutput
- WanAnimate2Stream
- WanAnimate2LiteStream

## Important memory behavior

VideoStreamLoader decodes incrementally with PyAV but returns a ComfyUI
IMAGE tensor containing all selected frames. This is necessary for the
current native Wan Animate 2 IMAGE conditioning interface.

It therefore reduces decoder-side overhead but does not make arbitrary
length generation OOM-proof.

The Wan adapters delegate conditioning and latent construction to
ComfyUI's native WanAnimate2ToVideo implementation. Sampling remains
the responsibility of the regular ComfyUI sampler.

The Lite adapter shares the native conditioning contract. Select a
compatible Lite checkpoint in the model loader.

## Temporal behavior

Wan Animate 2 uses temporal compression of four and frame lengths
following 4n+1. The temporal planner is informational; it does not
independently sample chunks. For continuation, use the native
continue_motion and video_frame_offset inputs and validate the
model's continuation behavior.

## Resolution

Auto Resize is the default. Preserve Resolution retains source dimensions.
The native Wan node still validates dimensions and model constraints.

## Installation

PyAV must be available in the ComfyUI Python environment.
No ComfyUI core, PyTorch, ROCm, or existing custom-node files are modified.
