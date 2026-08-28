from __future__ import annotations

from traffic_intelligence.analytics.metrics import compute_traffic_metrics
from traffic_intelligence.schemas.metrics import CongestionState
from traffic_intelligence.schemas.track import TrackSummary


def _summary(track_id: int, class_name: str, class_id: int = 2) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        mean_confidence=0.9,
        frame_count=30,
        first_frame=0,
        last_frame=29,
        first_timestamp=0.0,
        last_timestamp=1.0,
    )


def test_compute_traffic_metrics_separates_vehicles_and_pedestrians():
    summaries = [
        _summary(1, "car"),
        _summary(2, "bus"),
        _summary(3, "person", class_id=0),
        _summary(4, "person", class_id=0),
    ]

    metrics = compute_traffic_metrics(
        summaries, CongestionState.LOW, video_duration_s=1.0, frames_processed=30
    )

    assert metrics.total_vehicles == 2
    assert metrics.total_pedestrians == 2
    assert metrics.vehicles_per_class == {"car": 1, "bus": 1}
    assert "person" not in metrics.vehicles_per_class


def test_compute_traffic_metrics_handles_no_tracks():
    metrics = compute_traffic_metrics(
        [], CongestionState.LOW, video_duration_s=30.0, frames_processed=900
    )

    assert metrics.total_vehicles == 0
    assert metrics.total_pedestrians == 0
    assert metrics.vehicles_per_class == {}


def test_compute_traffic_metrics_reports_traffic_level():
    summaries = [_summary(1, "car")]
    metrics = compute_traffic_metrics(
        summaries, CongestionState.HIGH, video_duration_s=10.0, frames_processed=300
    )

    assert metrics.traffic_level == CongestionState.HIGH
