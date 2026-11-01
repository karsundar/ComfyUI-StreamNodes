"""PyAV metadata and decoding helpers."""

import av


def inspect_video(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        return {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate or 0),
            "frames": int(stream.frames or 0),
            "duration": float(stream.duration * stream.time_base)
            if stream.duration is not None else 0.0,
        }
