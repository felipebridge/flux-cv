from __future__ import annotations

import cv2
import numpy as np

from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.metrics import CongestionState

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_CONGESTION_COLORS = {
    CongestionState.LOW: (110, 200, 90),
    CongestionState.MODERATE: (0, 179, 255),
    CongestionState.HIGH: (60, 60, 220),
}

_PERSON_COLOR = (222, 196, 60)
_VEHICLE_COLORS = {
    "car": (231, 158, 40),
    "bus": (168, 76, 173),
    "truck": (0, 149, 255),
    "motorcycle": (66, 66, 214),
    "bicycle": (110, 190, 90),
}
_FALLBACK_VEHICLE_COLOR = (190, 190, 190)

_PANEL_BG = (32, 30, 28)
_PANEL_TEXT = (240, 240, 240)
_PANEL_MUTED_TEXT = (165, 165, 165)

# Sizing is derived from frame width relative to this reference so labels and the panel stay
# legible from small clips through 4K, instead of a fixed pixel size tuned for one resolution.
# _MIN_SCALE is well above 1.0 because "legible at 1x" on a 1280px-wide frame still reads as
# small text on a modern display -- most everything is bumped up from there.
_REFERENCE_WIDTH = 1280.0
_MIN_SCALE = 1.3
_MAX_SCALE = 4.0


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    if class_name == "person":
        return _PERSON_COLOR
    return _VEHICLE_COLORS.get(class_name, _FALLBACK_VEHICLE_COLOR)


def _text_color_for_background(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (20, 20, 20) if sum(color) > 380 else (245, 245, 245)


def _put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _put_label_outlined(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    """Like _put_label, but with a dark halo behind the glyphs -- the panel sits over live
    video, so a light color alone can wash out against a bright patch of frame behind it."""
    cv2.putText(frame, text, origin, _FONT, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


class FrameAnnotator:
    """Renders bounding boxes, per-track trails, per-vehicle speed labels, and a summary
    panel."""

    def __init__(self, trail_length: int = 20) -> None:
        self._trail_length = trail_length

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[TrackedDetection],
        trails: dict[int, list[tuple[float, float]]],
        speeds: dict[int, float | None],
        vehicle_count: int,
        person_count: int,
        traffic_level: CongestionState,
    ) -> np.ndarray:
        scale = min(_MAX_SCALE, max(_MIN_SCALE, frame.shape[1] / _REFERENCE_WIDTH))
        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(
                annotated,
                detection,
                trails.get(detection.track_id, []),
                speeds.get(detection.track_id),
                scale,
            )
        self._draw_summary_panel(annotated, vehicle_count, person_count, traffic_level, scale)
        return annotated

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: TrackedDetection,
        trail: list[tuple[float, float]],
        speed_kmh: float | None,
        scale: float,
    ) -> None:
        color = _color_for_class(detection.class_name)
        box_thickness = max(2, round(2.4 * scale))
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        label = f"#{detection.track_id} {detection.class_name}"
        if detection.class_name != "person" and speed_kmh is not None:
            # Hershey fonts (OpenCV's only built-in option) only cover ASCII, so this stays
            # plain "-" rather than a nicer unicode separator that would render as garbage.
            label += f" - {speed_kmh:.0f} km/h"

        font_scale = 0.62 * scale
        text_thickness = 2 if scale >= 1.6 else 1
        pad = max(4, round(3.5 * scale))
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
        text_color = _text_color_for_background(color)

        label_top = y1 - text_h - baseline - 2 * pad
        if label_top < 0:
            # Not enough room above the box (it starts near the top of the frame) -- draw the
            # label banner just inside the box instead of letting it clip off-screen.
            bg_top, bg_bottom = y1, y1 + text_h + baseline + 2 * pad
        else:
            bg_top, bg_bottom = label_top, y1
        cv2.rectangle(frame, (x1, bg_top), (x1 + text_w + 2 * pad, bg_bottom), color, thickness=-1)
        _put_label(frame, label, (x1 + pad, bg_bottom - pad - baseline), text_color, font_scale, text_thickness)

        recent_trail = trail[-self._trail_length :]
        if len(recent_trail) >= 2:
            points = np.array(recent_trail, dtype=np.int32)
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=color,
                thickness=max(2, round(2.2 * scale)),
                lineType=cv2.LINE_AA,
            )

    def _draw_summary_panel(
        self,
        frame: np.ndarray,
        vehicle_count: int,
        person_count: int,
        traffic_level: CongestionState,
        scale: float,
    ) -> None:
        level_color = _CONGESTION_COLORS[traffic_level]
        margin = round(20 * scale)
        width, height = round(380 * scale), round(200 * scale)
        x0, y0 = margin, margin
        x1, y1 = x0 + width, y0 + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _PANEL_BG, thickness=-1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), thickness=max(1, round(scale)))

        accent_width = max(6, round(8 * scale))
        cv2.rectangle(frame, (x0, y0), (x0 + accent_width, y1), level_color, thickness=-1)

        text_x = x0 + accent_width + round(16 * scale)
        line_scale = 1.0 * scale
        line_gap = round(48 * scale)
        text_thickness = 2 if scale >= 1.6 else 1
        first_line_y = y0 + round(48 * scale)

        _put_label_outlined(
            frame, f"Vehicles  {vehicle_count}", (text_x, first_line_y), _PANEL_TEXT, line_scale, text_thickness
        )
        _put_label_outlined(
            frame,
            f"People    {person_count}",
            (text_x, first_line_y + line_gap),
            _PANEL_TEXT,
            line_scale,
            text_thickness,
        )
        _put_label_outlined(
            frame,
            f"Traffic   {traffic_level.value}",
            (text_x, first_line_y + 2 * line_gap),
            level_color,
            line_scale * 1.1,
            text_thickness,
        )
        _put_label_outlined(
            frame,
            "Speed shown per vehicle is a CV estimate",
            (text_x, y1 - round(14 * scale)),
            _PANEL_MUTED_TEXT,
            0.48 * scale,
            1,
        )
