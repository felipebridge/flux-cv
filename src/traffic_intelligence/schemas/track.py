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

    @property
    def duration_s(self) -> float:
        return self.last_timestamp - self.first_timestamp
