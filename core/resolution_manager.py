"""Resolution selection without changing source aspect ratio."""

import torch
import torch.nn.functional as F


def select_resolution(width, height, target_width, target_height,
                      mode="Auto Resize", multiple=16):
    width, height = int(width), int(height)

    if mode == "Preserve Resolution":
        return width, height

    scale = min(
        target_width / width,
        target_height / height
    )

    out_w = max(multiple, int(width * scale) // multiple * multiple)
    out_h = max(multiple, int(height * scale) // multiple * multiple)

    return out_w, out_h


def resize_images(images, width, height):
    if images.shape[1:3] == (height, width):
        return images

    x = images.movedim(-1, 1)
    x = F.interpolate(
        x, size=(height, width),
        mode="bilinear", align_corners=False
    )
    return x.movedim(1, -1).clamp(0, 1)
