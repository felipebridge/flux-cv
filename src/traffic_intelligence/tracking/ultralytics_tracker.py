from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from traffic_intelligence.config.settings import DetectionConfig, TrackingConfig
from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.utils.device import resolve_device
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("tracking.ultralytics")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACKER_YAML = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}
_BOTSORT_REID_YAML = _REPO_ROOT / "configs" / "trackers" / "botsort_reid.yaml"


class ModelLoadError(Exception):
    """Raised when the configured YOLO weights cannot be loaded or don't know a configured class."""


class UltralyticsTracker:
    """Multi-object tracker built on Ultralytics' native ByteTrack/BoT-SORT
    integration rather than a hand-rolled reimplementation.

    Both trackers combine a Kalman-filter motion model with Hungarian
    assignment over detection/track affinity; reimplementing that machinery
    from scratch would trade a well-tested, widely benchmarked tracker for a
    weaker one without adding real value. Swapping trackers is a config
    change (`tracking.tracker: bytetrack|botsort`), and a different tracking
    backend can be plugged in later by implementing the same `Tracker`
    protocol.
    """

    def __init__(self, detection_config: DetectionConfig, tracking_config: TrackingConfig) -> None:
        self._detection_config = detection_config
        self._device = resolve_device(detection_config.device)
        try:
            self._model = YOLO(detection_config.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load YOLO weights from '{detection_config.model_path}': {exc}"
            ) from exc

        self._class_ids = self._resolve_class_ids(detection_config.classes)
        if tracking_config.tracker.value == "botsort" and tracking_config.appearance_reid_enabled:
            self._tracker_yaml = str(_BOTSORT_REID_YAML)
        else:
            self._tracker_yaml = _TRACKER_YAML[tracking_config.tracker.value]
        logger.info(
            "Loaded tracker model=%s device=%s tracker=%s",
            Path(detection_config.model_path).name,
            self._device,
            tracking_config.tracker.value,
        )

    def _resolve_class_ids(self, class_names: list[str]) -> list[int]:
        name_to_id = {name: idx for idx, name in self._model.names.items()}
        missing = [name for name in class_names if name not in name_to_id]
        if missing:
            raise ModelLoadError(
                f"Classes {missing} are not known to model "
                f"'{self._detection_config.model_path}'. Available classes: {sorted(name_to_id)}"
            )
        return [name_to_id[name] for name in class_names]

    def track(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> list[TrackedDetection]:
        results = self._model.track(
            frame,
            conf=self._detection_config.confidence_threshold,
            iou=self._detection_config.iou_threshold,
            classes=self._class_ids,
            device=self._device,
            tracker=self._tracker_yaml,
            persist=True,
            verbose=False,
        )
        return self._to_tracked_detections(results[0], frame_index, timestamp)

    def _to_tracked_detections(
        self, result, frame_index: int, timestamp: float
    ) -> list[TrackedDetection]:
        tracked: list[TrackedDetection] = []
        if result.boxes is None or result.boxes.id is None:
            return tracked

        for box in result.boxes:
            if box.id is None:
                continue
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            tracked.append(
                TrackedDetection(
                    track_id=int(box.id.item()),
                    frame_index=frame_index,
                    timestamp=timestamp,
                    class_id=class_id,
                    class_name=self._model.names[class_id],
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return tracked

    def reset(self) -> None:
        self._model.predictor = None
