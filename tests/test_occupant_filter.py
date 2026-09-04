from __future__ import annotations

from traffic_intelligence.analytics.occupant_filter import exclude_vehicle_occupants
from traffic_intelligence.schemas.detection import TrackedDetection


def _detection(
    track_id: int, class_name: str, bbox: tuple[float, float, float, float], class_id: int = 2
) -> TrackedDetection:
    return TrackedDetection(
        frame_index=0,
        timestamp=0.0,
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
    )


def test_drops_person_mostly_contained_in_a_vehicle_box():
    car = _detection(1, "car", (0, 0, 200, 100))
    driver = _detection(2, "person", (60, 20, 100, 60), class_id=0)  # fully inside the car box

    result = exclude_vehicle_occupants([car, driver])

    assert [d.track_id for d in result] == [1]


def test_keeps_person_mostly_outside_any_vehicle_box():
    car = _detection(1, "car", (0, 0, 200, 100))
    pedestrian = _detection(2, "person", (250, 0, 280, 100), class_id=0)  # no overlap with the car

    result = exclude_vehicle_occupants([car, pedestrian])

    assert {d.track_id for d in result} == {1, 2}


def test_keeps_person_only_lightly_overlapping_a_vehicle_box():
    car = _detection(1, "car", (0, 0, 200, 100))
    # Only ~20% of the person box overlaps the car -- standing beside it, not inside it.
    person = _detection(2, "person", (180, 0, 220, 100), class_id=0)

    result = exclude_vehicle_occupants([car, person])

    assert {d.track_id for d in result} == {1, 2}


def test_no_vehicles_in_frame_keeps_all_people():
    pedestrian_a = _detection(1, "person", (0, 0, 20, 40), class_id=0)
    pedestrian_b = _detection(2, "person", (100, 0, 120, 40), class_id=0)

    result = exclude_vehicle_occupants([pedestrian_a, pedestrian_b])

    assert {d.track_id for d in result} == {1, 2}


def test_empty_detections_returns_empty():
    assert exclude_vehicle_occupants([]) == []
