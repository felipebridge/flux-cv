from __future__ import annotations

from pydantic import BaseModel


class TrackSummary(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    mean_confidence: float
    frame_count: int
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    first_centroid: tuple[float, float] = (0.0, 0.0)
    last_centroid: tuple[float, float] = (0.0, 0.0)
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    speed_estimated: bool = False

    @property
    def duration_s(self) -> float:
        return self.last_timestamp - self.first_timestamp
