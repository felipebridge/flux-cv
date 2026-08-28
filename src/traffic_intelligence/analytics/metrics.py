from __future__ import annotations

from collections import Counter

from traffic_intelligence.schemas.metrics import CongestionState, TrafficMetrics
from traffic_intelligence.schemas.track import TrackSummary

_PEDESTRIAN_CLASS_NAME = "person"


def compute_traffic_metrics(
    track_summaries: list[TrackSummary],
    traffic_level: CongestionState,
    video_duration_s: float,
    frames_processed: int,
) -> TrafficMetrics:
    vehicle_summaries = [t for t in track_summaries if t.class_name != _PEDESTRIAN_CLASS_NAME]
    total_vehicles = len(vehicle_summaries)
    total_pedestrians = len(track_summaries) - total_vehicles
    vehicles_per_class = dict(Counter(t.class_name for t in vehicle_summaries))

    return TrafficMetrics(
        total_vehicles=total_vehicles,
        total_pedestrians=total_pedestrians,
        vehicles_per_class=vehicles_per_class,
        traffic_level=traffic_level,
        video_duration_s=video_duration_s,
        frames_processed=frames_processed,
    )
