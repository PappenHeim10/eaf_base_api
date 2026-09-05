import contextlib
import io
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from base_api.base import BaseCore


def permission_error(winerror, message):
    error = PermissionError(winerror, message)
    error.winerror = winerror
    return error


class FakeInput:
    def __init__(self, fmt_name="mp4"):
        self.format = types.SimpleNamespace(name=fmt_name)
        self.streams = []
        self.closed = False

    def close(self):
        self.closed = True

    def demux(self, streams):
        return iter(())


class FakeOutput:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def add_stream_from_template(self, template):
        return types.SimpleNamespace()

    def add_stream(self, *args, **kwargs):
        return types.SimpleNamespace()

    def mux(self, packet):
        return None


class FakeStreams:
    def __init__(self, video):
        self.video = [video]
        self._streams = [video]

    def __iter__(self):
        return iter(self._streams)


@contextlib.contextmanager
def fake_av(input_container):
    av_module = types.ModuleType("av")
    av_module.open = lambda *args, **kwargs: input_container
    audio_module = types.ModuleType("av.audio")
    frame_module = types.ModuleType("av.audio.frame")
    resampler_module = types.ModuleType("av.audio.resampler")
    resampler_module.AudioResampler = object
    audio_module.frame = frame_module
    audio_module.resampler = resampler_module
    av_module.audio = audio_module
    original = {name: sys.modules.get(name) for name in (
        "av", "av.audio", "av.audio.frame", "av.audio.resampler"
    )}
    sys.modules.update({
        "av": av_module,
        "av.audio": audio_module,
        "av.audio.frame": frame_module,
        "av.audio.resampler": resampler_module,
    })
    try:
        yield
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class RemuxLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.core = object.__new__(BaseCore)
        self.core.logger = logging.getLogger("remux-tests")

    def test_pass_through_rename_moves_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            output_path = Path(temp_dir) / "output.mp4"
            input_path.write_bytes(b"already-mp4")
            container = FakeInput("mp4")
            with fake_av(container):
                self.core._convert_ts_to_mp4(str(input_path), str(output_path))
            self.assertFalse(input_path.exists())
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), b"already-mp4")

    def test_pass_through_closes_input_before_rename(self):
        container = FakeInput("mp4")
        rename_observations = []

        def observe_rename(source, target):
            rename_observations.append(container.closed)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"input")
            with fake_av(container), patch("base_api.base.os.replace", side_effect=observe_rename):
                self.core._convert_ts_to_mp4(str(input_path), str(Path(temp_dir) / "output.mp4"))
        self.assertEqual(rename_observations, [True])

    def test_remux_closes_input_and_output(self):
        input_container = FakeInput("mpegts")
        output_container = FakeOutput()
        video_stream = types.SimpleNamespace(
            type="video", index=0,
            codec_context=types.SimpleNamespace(name="h264", bit_rate=0),
        )
        input_container.streams = FakeStreams(video_stream)

        def open_container(*args, **kwargs):
            return output_container if kwargs.get("mode") == "w" else input_container

        av_module = types.ModuleType("av")
        av_module.open = open_container
        audio_module = types.ModuleType("av.audio")
        frame_module = types.ModuleType("av.audio.frame")
        resampler_module = types.ModuleType("av.audio.resampler")
        resampler_module.AudioResampler = object
        audio_module.frame = frame_module
        audio_module.resampler = resampler_module
        av_module.audio = audio_module
        with patch.dict(sys.modules, {
            "av": av_module,
            "av.audio": audio_module,
            "av.audio.frame": frame_module,
            "av.audio.resampler": resampler_module,
        }), patch("base_api.base.os.path.getsize", return_value=1), patch("base_api.base.os.replace"):
            self.core._convert_ts_to_mp4("input.ts", "output.mp4")
        self.assertTrue(input_container.closed)
        self.assertTrue(output_container.closed)

    def test_single_winerror_32_is_retried(self):
        container = FakeInput("mp4")
        calls = []

        def flaky_rename(source, target):
            calls.append(1)
            if len(calls) == 1:
                raise permission_error(32, "locked")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"input")
            with fake_av(container), patch("base_api.base.os.replace", side_effect=flaky_rename):
                self.core._convert_ts_to_mp4(str(input_path), str(Path(temp_dir) / "output.mp4"))
        self.assertEqual(len(calls), 2)

    def test_persistent_winerror_32_propagates_after_bounded_retry(self):
        container = FakeInput("mp4")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"input")
            with fake_av(container), patch(
                "base_api.base.os.replace", side_effect=permission_error(32, "locked")
            ) as rename:
                with self.assertRaises(PermissionError):
                    self.core._convert_ts_to_mp4(str(input_path), str(Path(temp_dir) / "output.mp4"))
        self.assertLessEqual(rename.call_count, 5)

    def test_other_permission_error_is_not_retried(self):
        container = FakeInput("mp4")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"input")
            with fake_av(container), patch(
                "base_api.base.os.replace", side_effect=permission_error(13, "denied")
            ) as rename:
                with self.assertRaises(PermissionError):
                    self.core._convert_ts_to_mp4(str(input_path), str(Path(temp_dir) / "output.mp4"))
        self.assertEqual(rename.call_count, 1)

    def test_remux_exception_closes_containers(self):
        input_container = FakeInput("mpegts")
        output_container = FakeOutput()
        video_stream = types.SimpleNamespace(
            type="video", index=0,
            codec_context=types.SimpleNamespace(name="h264", bit_rate=0),
        )
        input_container.streams = FakeStreams(video_stream)
        av_module = types.ModuleType("av")
        av_module.open = lambda *args, **kwargs: (
            output_container if kwargs.get("mode") == "w" else input_container
        )
        audio_module = types.ModuleType("av.audio")
        frame_module = types.ModuleType("av.audio.frame")
        resampler_module = types.ModuleType("av.audio.resampler")
        resampler_module.AudioResampler = object
        audio_module.frame = frame_module
        audio_module.resampler = resampler_module
        av_module.audio = audio_module
        with patch.dict(sys.modules, {
            "av": av_module,
            "av.audio": audio_module,
            "av.audio.frame": frame_module,
            "av.audio.resampler": resampler_module,
        }), patch.object(input_container, "demux", side_effect=RuntimeError("demux failed")):
            with self.assertRaises(RuntimeError):
                self.core._convert_ts_to_mp4("input.ts", "output.mp4")
        self.assertTrue(input_container.closed)
        self.assertTrue(output_container.closed)

    def test_output_existing_behavior_is_explicit(self):
        container = FakeInput("mp4")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"input")
            output_path = Path(temp_dir) / "output.mp4"
            output_path.write_bytes(b"old")
            with fake_av(container):
                try:
                    self.core._convert_ts_to_mp4(str(input_path), str(output_path))
                except OSError:
                    pass
            self.assertEqual(output_path.read_bytes(), b"input")

    def test_cleanup_failure_keeps_input(self):
        container = FakeInput("mp4")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.tmp"
            input_path.write_bytes(b"keep")
            with fake_av(container), patch("base_api.base.os.replace", side_effect=permission_error(13, "denied")):
                with self.assertRaises(PermissionError):
                    self.core._convert_ts_to_mp4(str(input_path), str(Path(temp_dir) / "output.mp4"))
            self.assertTrue(input_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
