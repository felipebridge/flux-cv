from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class VideoSourceError(Exception):
    """Raised for missing, unreadable, or malformed input video files."""


class VideoSource:
    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise VideoSourceError(f"Video file not found: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise VideoSourceError(f"Could not open video file (unsupported or corrupt): {self._path}")

        detected_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps_override if fps_override else detected_fps
        if not self.fps or self.fps <= 0:
            self._capture.release()
            raise VideoSourceError(
                f"Video '{self._path}' reports an invalid FPS ({detected_fps}). "
                "Set video.fps_override in the config to work around this."
            )

        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_width <= 0 or self.frame_height <= 0:
            self._capture.release()
            raise VideoSourceError(
                f"Video '{self._path}' reports an invalid resolution "
                f"({self.frame_width}x{self.frame_height})"
            )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        frame_index = 0
        while True:
            has_frame, frame = self._capture.read()
            if not has_frame:
                break
            yield frame_index, frame_index / self.fps, frame
            frame_index += 1

        if frame_index == 0:
            raise VideoSourceError(f"Video '{self._path}' contains no readable frames")

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
