from __future__ import annotations

from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.metrics import CongestionState, TrafficMetrics
from traffic_intelligence.schemas.track import TrackSummary


def test_tracked_detection_computed_centroid():
    detection = TrackedDetection(
        frame_index=0,
        timestamp=0.0,
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=(0, 0, 10, 20),
    )
    assert detection.centroid == (5.0, 10.0)


def test_tracked_detection_roundtrip_json():
    detection = TrackedDetection(
        track_id=42,
        frame_index=1,
        timestamp=0.033,
        class_id=2,
        class_name="car",
        confidence=0.87,
        bbox=(1.0, 2.0, 3.0, 4.0),
    )
    restored = TrackedDetection.model_validate_json(detection.model_dump_json())
    assert restored == detection


def test_track_summary_duration():
    summary = TrackSummary(
        track_id=1,
        class_id=2,
        class_name="car",
        mean_confidence=0.9,
        frame_count=10,
        first_frame=0,
        last_frame=9,
        first_timestamp=0.0,
        last_timestamp=0.3,
    )
    assert summary.duration_s == 0.3


def test_traffic_metrics_roundtrip():
    metrics = TrafficMetrics(
        total_vehicles=5,
        total_pedestrians=2,
        vehicles_per_class={"car": 5},
        traffic_level=CongestionState.MODERATE,
        video_duration_s=20.0,
        frames_processed=600,
    )
    restored = TrafficMetrics.model_validate(metrics.model_dump())
    assert restored == metrics
