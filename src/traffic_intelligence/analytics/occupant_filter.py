from __future__ import annotations

from traffic_intelligence.schemas.detection import TrackedDetection

_CONTAINMENT_THRESHOLD = 0.6


def _containment_ratio(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0

    overlap_x1, overlap_y1 = max(ix1, ox1), max(iy1, oy1)
    overlap_x2, overlap_y2 = min(ix2, ox2), min(iy2, oy2)
    overlap_area = max(0.0, overlap_x2 - overlap_x1) * max(0.0, overlap_y2 - overlap_y1)
    return overlap_area / inner_area


def exclude_vehicle_occupants(
    detections: list[TrackedDetection], person_class: str = "person"
) -> list[TrackedDetection]:
    """Drops a person detection whose box mostly falls inside a vehicle's box in the same
    frame -- a driver or passenger seen through the windshield, not a pedestrian on the
    street. A general-purpose detector has no notion of "inside a vehicle"; it just finds a
    person-shaped region wherever one appears, windshield or sidewalk alike. Left unfiltered,
    that occupant gets tracked as its own short-lived "person," inheriting the vehicle's speed
    on screen -- which is how a parked-looking clip ends up with a "pedestrian" moving at
    30+ km/h. `_CONTAINMENT_THRESHOLD` is deliberately not close to 1.0: a driver's visible
    silhouette is rarely the *entire* windshield opening, so requiring near-total containment
    would miss most real cases; a person genuinely standing beside a car reaches this level of
    overlap far less often, but a false suppression here and there is the accepted trade-off.
    """
    vehicle_bboxes = [d.bbox for d in detections if d.class_name != person_class]
    if not vehicle_bboxes:
        return detections

    def _is_occupant(person: TrackedDetection) -> bool:
        return any(
            _containment_ratio(person.bbox, vehicle_bbox) >= _CONTAINMENT_THRESHOLD for vehicle_bbox in vehicle_bboxes
        )

    return [d for d in detections if d.class_name != person_class or not _is_occupant(d)]
