from __future__ import annotations

import cv2
import numpy as np

from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.metrics import CongestionState

_CONGESTION_COLORS = {
    CongestionState.LOW: (80, 200, 120),
    CongestionState.MODERATE: (0, 200, 255),
    CongestionState.HIGH: (0, 0, 255),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_SIZE = (280, 100)

_PERSON_COLOR = (0, 255, 255)
_VEHICLE_COLORS = {
    "car": (255, 180, 0),
    "bus": (180, 0, 255),
    "truck": (0, 140, 255),
    "motorcycle": (0, 0, 255),
    "bicycle": (0, 220, 0),
}
_FALLBACK_VEHICLE_COLOR = (200, 200, 200)


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    if class_name == "person":
        return _PERSON_COLOR
    return _VEHICLE_COLORS.get(class_name, _FALLBACK_VEHICLE_COLOR)


def _put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    cv2.putText(frame, text, origin, _FONT, scale, color, 1, cv2.LINE_AA)


class FrameAnnotator:
    """Renders a clean overlay: bounding boxes with a compact label, a short
    trail per track, and a single summary panel with the three headline
    numbers this project reports — vehicle count, person count, traffic
    level. No per-object text walls."""

    def __init__(self, trail_length: int = 20) -> None:
        self._trail_length = trail_length

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[TrackedDetection],
        trails: dict[int, list[tuple[float, float]]],
        vehicle_count: int,
        person_count: int,
        traffic_level: CongestionState,
    ) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(annotated, detection, trails.get(detection.track_id, []))
        self._draw_summary_panel(annotated, vehicle_count, person_count, traffic_level)
        return annotated

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: TrackedDetection,
        trail: list[tuple[float, float]],
    ) -> None:
        color = _color_for_class(detection.class_name)
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _put_label(frame, f"#{detection.track_id} {detection.class_name}", (x1, max(0, y1 - 6)), color, scale=0.45)

        recent_trail = trail[-self._trail_length :]
        if len(recent_trail) >= 2:
            points = np.array(recent_trail, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=1)

    def _draw_summary_panel(
        self,
        frame: np.ndarray,
        vehicle_count: int,
        person_count: int,
        traffic_level: CongestionState,
    ) -> None:
        panel_color = _CONGESTION_COLORS[traffic_level]
        width, height = _PANEL_SIZE
        cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), (20, 20, 20), thickness=-1)
        cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), panel_color, thickness=2)
        _put_label(frame, f"Vehicles: {vehicle_count}", (20, 35), (255, 255, 255), scale=0.6)
        _put_label(frame, f"People: {person_count}", (20, 60), (255, 255, 255), scale=0.6)
        _put_label(frame, f"Traffic: {traffic_level.value}", (20, 90), panel_color, scale=0.65)
