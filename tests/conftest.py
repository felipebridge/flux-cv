from __future__ import annotations

import pytest

from traffic_intelligence.schemas.detection import TrackedDetection


def make_detection(
    track_id: int,
    frame_index: int,
    class_name: str = "car",
    class_id: int = 2,
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    fps: float = 30.0,
) -> TrackedDetection:
    return TrackedDetection(
        frame_index=frame_index,
        timestamp=frame_index / fps,
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )


@pytest.fixture
def detection_factory():
    return make_detection
