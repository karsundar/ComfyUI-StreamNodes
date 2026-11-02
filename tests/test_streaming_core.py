import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import torch

from core.frame_blend import blend_boundary
from core.frame_iterator import FrameIterator
from core.stream_writer import StreamVideoWriter
from core.temporal_window import plan_windows, valid_wan_length
from utils.video_decoder import inspect_video


class StreamingCoreTests(unittest.TestCase):
    def make_video(self, path, count=13, width=32, height=24):
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=13)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            for index in range(count):
                image = np.full((height, width, 3), index, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_window_boundaries_cover_every_frame(self):
        windows = plan_windows(100, window_length=81, overlap=1)
        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[-1].start + windows[-1].output_length + 1, 100)
        self.assertEqual(sum(window.output_length for window in windows), 100)

    def test_window_planner_terminates_on_short_final_window(self):
        windows = plan_windows(875, window_length=81, overlap=1)
        self.assertGreater(len(windows), 1)
        self.assertEqual(sum(window.output_length for window in windows), 875)
        self.assertGreaterEqual(
            windows[-1].start + windows[-1].length,
            875,
        )
        self.assertEqual(windows[-1].length, valid_wan_length(76))

    def test_boundary_blend_preserves_frame_count_and_endpoints(self):
        previous = torch.zeros((3, 2, 2, 3))
        current = torch.ones((4, 2, 2, 3))
        blended = blend_boundary(previous, current, 3)
        self.assertEqual(tuple(blended.shape), tuple(current.shape))
        self.assertTrue(torch.equal(blended[0], previous[-1]))
        self.assertTrue(torch.equal(blended[2], current[2]))
        self.assertTrue(torch.equal(blended[3], current[3]))

    def test_decoder_resizes_only_the_selected_window(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self.make_video(source)
            with FrameIterator(source, start=5, limit=3, resize=(16, 12)) as iterator:
                frames = list(iterator.frames())
            self.assertEqual(len(frames), 3)
            self.assertEqual(tuple(frames[0].shape), (12, 16, 3))

    def test_writer_does_not_need_an_image_list(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            with StreamVideoWriter(output, 13) as writer:
                for index in range(4):
                    writer.append(np.full((24, 32, 3), index / 4, dtype=np.float32))
            self.assertEqual(inspect_video(output)["frames"], 4)

    def test_writer_preserves_video_when_audio_mux_has_no_audio_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            self.make_video(source)
            with StreamVideoWriter(output, 13, audio_path=source) as writer:
                writer.append(np.zeros((24, 32, 3), dtype=np.float32))
            self.assertEqual(inspect_video(output)["frames"], 1)


if __name__ == "__main__":
    unittest.main()
