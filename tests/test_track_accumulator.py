from __future__ import annotations

from traffic_intelligence.tracking.track_accumulator import TrackAccumulator


def test_rejects_tracks_shorter_than_minimum_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    for frame_index in range(3):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    assert accumulator.finalize() == []


def test_keeps_tracks_meeting_minimum_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3)
    for frame_index in range(5):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    summaries = accumulator.finalize()
    assert len(summaries) == 1
    assert summaries[0].track_id == 1
    assert summaries[0].frame_count == 5


def test_dominant_class_wins_over_flicker(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, class_name="car", class_id=2))
    accumulator.add(detection_factory(track_id=1, frame_index=1, class_name="car", class_id=2))
    accumulator.add(detection_factory(track_id=1, frame_index=2, class_name="truck", class_id=7))

    summaries = accumulator.finalize()
    assert summaries[0].class_name == "car"


def test_mean_confidence_is_averaged(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, confidence=0.6))
    accumulator.add(detection_factory(track_id=1, frame_index=1, confidence=0.8))

    summaries = accumulator.finalize()
    assert summaries[0].mean_confidence == 0.7


def test_trail_returns_recent_centroids(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, bbox=(0, 0, 10, 10)))
    accumulator.add(detection_factory(track_id=1, frame_index=1, bbox=(10, 10, 20, 20)))

    trail = accumulator.trail(1)
    assert trail == [(5.0, 5.0), (15.0, 15.0)]


def test_trail_empty_for_unknown_track():
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    assert accumulator.trail(999) == []


def test_single_frame_track_is_always_rejected(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.01)
    accumulator.add(detection_factory(track_id=1, frame_index=0, class_name="car"))
    accumulator.add(detection_factory(track_id=1, frame_index=1, class_name="car"))
    accumulator.add(detection_factory(track_id=2, frame_index=0, class_name="person", class_id=0))

    summaries = {s.track_id: s for s in accumulator.finalize()}
    assert set(summaries) == {1}


def test_rejects_flickering_track_despite_meeting_duration(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in (0, 5, 9):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    assert accumulator.finalize() == []


def test_keeps_track_meeting_visibility_ratio_despite_occlusion_gaps(detection_factory):
    accumulator = TrackAccumulator(min_track_seconds=0.3, min_visibility_ratio=0.6)
    for frame_index in (0, 1, 2, 3, 5, 6, 7, 9):
        accumulator.add(detection_factory(track_id=1, frame_index=frame_index, fps=10.0))

    summaries = accumulator.finalize()
    assert len(summaries) == 1
    assert summaries[0].frame_count == 8
