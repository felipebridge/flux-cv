from __future__ import annotations

from collections import Counter, deque

from traffic_intelligence.config.settings import CongestionConfig
from traffic_intelligence.schemas.metrics import CongestionState


class CongestionClassifier:
    """Classifies congestion from vehicle density, smoothed by a rolling vote over `persistence_frames`."""

    def __init__(self, config: CongestionConfig) -> None:
        self._config = config
        self._history: deque[CongestionState] = deque(maxlen=config.persistence_frames)

    def classify_instant(self, vehicle_density: int) -> CongestionState:
        thresholds = self._config.density_thresholds
        if vehicle_density >= thresholds.high:
            return CongestionState.HIGH
        if vehicle_density >= thresholds.moderate:
            return CongestionState.MODERATE
        return CongestionState.LOW

    def update(self, vehicle_density: int) -> CongestionState:
        instant = self.classify_instant(vehicle_density)
        self._history.append(instant)
        most_common, _ = Counter(self._history).most_common(1)[0]
        return most_common
