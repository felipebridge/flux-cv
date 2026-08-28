from __future__ import annotations

from pydantic import BaseModel, computed_field


class TrackedDetection(BaseModel):
    frame_index: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @computed_field
    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
