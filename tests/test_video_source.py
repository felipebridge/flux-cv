from __future__ import annotations

import pytest

from traffic_intelligence.pipeline.video_source import VideoSource, VideoSourceError


def test_missing_file_raises_video_source_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.mp4"
    with pytest.raises(VideoSourceError, match="not found"):
        VideoSource(missing_path)


def test_corrupt_or_unsupported_file_raises_video_source_error(tmp_path):
    fake_video = tmp_path / "not_a_video.mp4"
    fake_video.write_bytes(b"this is not a real video file")

    with pytest.raises(VideoSourceError):
        VideoSource(fake_video)
