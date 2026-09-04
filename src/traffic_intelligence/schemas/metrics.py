from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CongestionState(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class TrafficMetrics(BaseModel):
    total_vehicles: int
    total_pedestrians: int
    vehicles_per_class: dict[str, int] = Field(default_factory=dict)
    traffic_level: CongestionState
    video_duration_s: float
    frames_processed: int
    average_vehicle_speed_kmh: float | None = None
    speed_estimated: bool = False
