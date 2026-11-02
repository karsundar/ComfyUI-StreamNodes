"""PyAV metadata helpers that never decode the complete video."""

import av


def inspect_video(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else 0.0)
        frames = int(stream.frames or 0)
        if not frames and duration and rate:
            frames = int(round(duration * float(rate)))
        return {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(rate or 0),
            "fps_numerator": int(rate.numerator) if rate else 0,
            "fps_denominator": int(rate.denominator) if rate else 1,
            "variable_frame_rate": bool(
                stream.average_rate and getattr(stream, "base_rate", None)
                and stream.average_rate != stream.base_rate
            ),
            "frames": frames,
            "duration": duration,
        }
