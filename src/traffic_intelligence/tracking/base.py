from __future__ import annotations

from typing import Protocol

import numpy as np

from traffic_intelligence.schemas.detection import TrackedDetection


class Tracker(Protocol):
    """Assigns persistent identities to detections across frames."""

    def track(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> list[TrackedDetection]: ...

    def reset(self) -> None: ...
