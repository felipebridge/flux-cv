from __future__ import annotations

from traffic_intelligence.schemas.track import TrackSummary
from traffic_intelligence.tracking.track_stitcher import stitch_fragmented_tracks

_DIAGONAL = 1000.0


def _summary(
    track_id: int,
    class_name: str,
    first_frame: int,
    last_frame: int,
    first_timestamp: float,
    last_timestamp: float,
    first_centroid: tuple[float, float] = (0.0, 0.0),
    last_centroid: tuple[float, float] = (0.0, 0.0),
    frame_count: int = 10,
    mean_confidence: float = 0.6,
) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        class_id=2,
        class_name=class_name,
        mean_confidence=mean_confidence,
        frame_count=frame_count,
        first_frame=first_frame,
        last_frame=last_frame,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        first_centroid=first_centroid,
        last_centroid=last_centroid,
    )


def test_merges_track_reacquired_close_by_shortly_after():
    earlier = _summary(1, "car", 0, 29, 0.0, 1.0, last_centroid=(100.0, 100.0), frame_count=30)
    later = _summary(2, "car", 45, 74, 1.5, 2.5, first_centroid=(110.0, 105.0), frame_count=30)

    result = stitch_fragmented_tracks(
        [earlier, later], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 1
    assert result[0].frame_count == 60
    assert result[0].last_frame == 74
    assert result[0].last_timestamp == 2.5


def test_does_not_merge_when_gap_too_long():
    earlier = _summary(1, "car", 0, 29, 0.0, 1.0, last_centroid=(100.0, 100.0))
    later = _summary(2, "car", 200, 229, 5.0, 6.0, first_centroid=(101.0, 100.0))

    result = stitch_fragmented_tracks(
        [earlier, later], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 2


def test_does_not_merge_when_far_apart_spatially():
    earlier = _summary(1, "car", 0, 29, 0.0, 1.0, last_centroid=(0.0, 0.0))
    later = _summary(2, "car", 45, 74, 1.2, 2.2, first_centroid=(900.0, 900.0))

    result = stitch_fragmented_tracks(
        [earlier, later], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 2


def test_does_not_merge_overlapping_simultaneous_tracks():
    earlier = _summary(1, "car", 0, 50, 0.0, 2.0, last_centroid=(100.0, 100.0))
    later = _summary(2, "car", 20, 74, 0.8, 2.5, first_centroid=(101.0, 100.0))

    result = stitch_fragmented_tracks(
        [earlier, later], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 2


def test_does_not_merge_across_different_classes():
    earlier = _summary(1, "car", 0, 29, 0.0, 1.0, last_centroid=(100.0, 100.0))
    later = _summary(2, "person", 45, 74, 1.2, 2.2, first_centroid=(101.0, 100.0))

    result = stitch_fragmented_tracks(
        [earlier, later], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 2


def test_merges_a_chain_of_more_than_two_fragments():
    first = _summary(1, "car", 0, 29, 0.0, 1.0, last_centroid=(100.0, 100.0), frame_count=30)
    second = _summary(
        2, "car", 45, 74, 1.2, 2.2, first_centroid=(101.0, 101.0), last_centroid=(150.0, 150.0), frame_count=30
    )
    third = _summary(3, "car", 90, 119, 2.4, 3.4, first_centroid=(151.0, 151.0), frame_count=30)

    result = stitch_fragmented_tracks(
        [first, second, third], frame_diagonal=_DIAGONAL, max_gap_seconds=1.0, max_centroid_distance_ratio=0.06
    )

    assert len(result) == 1
    assert result[0].frame_count == 90
    assert result[0].first_frame == 0
    assert result[0].last_frame == 119
