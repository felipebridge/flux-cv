from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("persistence.video_encoder")


def finalize_video(raw_path: Path, final_path: Path) -> Path:
    """Transcodes the raw annotated video to H.264 via the system `ffmpeg` binary.

    `pipeline.runner` writes frames with OpenCV's `mp4v` codec because it's the only one this
    project can rely on always being available -- OpenCV's own H.264 writer depends on an
    OpenH264 DLL that isn't reliably present (see docs/architecture.md). `mp4v` isn't
    hardware-decodable on most systems and produces a very high bitrate at 4K, which shows up
    as stutter during playback despite the file itself being valid. Transcoding through
    `ffmpeg` afterwards fixes that with a normal, widely-supported encode.

    Falls back to leaving the raw file in place (renamed to `final_path`) if `ffmpeg` isn't on
    PATH or the transcode fails, so a missing/broken ffmpeg install never loses the run's
    output -- it's just bulkier and choppier.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning(
            "ffmpeg not found on PATH; leaving the annotated video as the raw mp4v encode "
            "(%s). Install ffmpeg for a smaller, smoother-playing H.264 output.",
            raw_path,
        )
        return _use_raw_as_final(raw_path, final_path)

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "ffmpeg transcode failed (exit %d); leaving the annotated video as the raw mp4v "
            "encode. stderr tail: %s",
            result.returncode,
            result.stderr[-500:],
        )
        return _use_raw_as_final(raw_path, final_path)

    raw_path.unlink(missing_ok=True)
    return final_path


def _use_raw_as_final(raw_path: Path, final_path: Path) -> Path:
    if raw_path != final_path:
        raw_path.replace(final_path)
    return final_path
