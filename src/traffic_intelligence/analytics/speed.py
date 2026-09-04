from __future__ import annotations

import statistics

_DEFAULT_REFERENCE_WIDTHS_M: dict[str, float] = {
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.5,
    "motorcycle": 0.8,
    "bicycle": 0.6,
}


class SpeedEstimator:
    """Converts pixel motion into an estimated real-world speed via automatic scale
    calibration: no camera calibration is provided for an arbitrary input video, so this
    assumes a typical real-world width per vehicle class (`reference_widths_m`) and infers a
    pixels-per-meter scale from the median observed bounding-box width across confirmed
    vehicles. Perspective and camera angle aren't modeled, so reported speeds are estimates,
    not measurements from a calibrated sensor (see docs/architecture.md)."""

    def __init__(
        self,
        reference_widths_m: dict[str, float] | None = None,
        min_calibration_samples: int = 30,
    ) -> None:
        self._reference_widths_m = reference_widths_m or dict(_DEFAULT_REFERENCE_WIDTHS_M)
        self._min_calibration_samples = min_calibration_samples
        self._pixels_per_meter_samples: list[float] = []

    def observe(self, class_name: str, bbox_width_px: float) -> None:
        reference_width_m = self._reference_widths_m.get(class_name)
        if not reference_width_m or bbox_width_px <= 1.0:
            return
        self._pixels_per_meter_samples.append(bbox_width_px / reference_width_m)

    @property
    def is_calibrated(self) -> bool:
        return len(self._pixels_per_meter_samples) >= self._min_calibration_samples

    @property
    def meters_per_pixel(self) -> float | None:
        if not self.is_calibrated:
            return None
        pixels_per_meter = statistics.median(self._pixels_per_meter_samples)
        if pixels_per_meter <= 0:
            return None
        return 1.0 / pixels_per_meter

    def speed_kmh(self, distance_px: float, duration_s: float) -> float | None:
        meters_per_pixel = self.meters_per_pixel
        if meters_per_pixel is None or duration_s <= 0:
            return None
        meters = distance_px * meters_per_pixel
        return (meters / duration_s) * 3.6
