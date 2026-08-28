from __future__ import annotations

from collections import Counter, defaultdict, deque

from traffic_intelligence.schemas.detection import TrackedDetection
from traffic_intelligence.schemas.track import TrackSummary

_TRAIL_LENGTH = 20


class TrackAccumulator:
    """Accumulates per-track duration, visibility, dominant class, and a recent-centroid trail."""

    def __init__(self, min_track_seconds: float, min_visibility_ratio: float = 0.6) -> None:
        self._min_track_seconds = min_track_seconds
        self._min_visibility_ratio = min_visibility_ratio
        self._frame_counts: dict[int, int] = defaultdict(int)
        self._class_votes: dict[int, Counter[int]] = defaultdict(Counter)
        self._class_names: dict[int, dict[int, str]] = defaultdict(dict)
        self._confidences: dict[int, list[float]] = defaultdict(list)
        self._first_seen: dict[int, tuple[int, float]] = {}
        self._last_seen: dict[int, tuple[int, float]] = {}
        self._first_centroid: dict[int, tuple[float, float]] = {}
        self._last_centroid: dict[int, tuple[float, float]] = {}
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=_TRAIL_LENGTH)
        )

    def add(self, detection: TrackedDetection) -> None:
        track_id = detection.track_id
        self._frame_counts[track_id] += 1
        self._class_votes[track_id][detection.class_id] += 1
        self._class_names[track_id][detection.class_id] = detection.class_name
        self._confidences[track_id].append(detection.confidence)
        self._trails[track_id].append(detection.centroid)

        if track_id not in self._first_seen:
            self._first_seen[track_id] = (detection.frame_index, detection.timestamp)
            self._first_centroid[track_id] = detection.centroid
        self._last_seen[track_id] = (detection.frame_index, detection.timestamp)
        self._last_centroid[track_id] = detection.centroid

    def trail(self, track_id: int) -> list[tuple[float, float]]:
        return list(self._trails.get(track_id, ()))

    def is_confirmed(self, track_id: int) -> bool:
        if track_id not in self._first_seen:
            return False
        first_frame, first_timestamp = self._first_seen[track_id]
        last_frame, last_timestamp = self._last_seen[track_id]
        if last_timestamp - first_timestamp < self._min_track_seconds:
            return False
        span_frames = last_frame - first_frame + 1
        return self._frame_counts[track_id] / span_frames >= self._min_visibility_ratio

    def finalize(self) -> list[TrackSummary]:
        summaries: list[TrackSummary] = []
        for track_id, frame_count in self._frame_counts.items():
            if not self.is_confirmed(track_id):
                continue

            first_frame, first_timestamp = self._first_seen[track_id]
            last_frame, last_timestamp = self._last_seen[track_id]
            dominant_class_id, _ = self._class_votes[track_id].most_common(1)[0]
            confidences = self._confidences[track_id]

            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    class_id=dominant_class_id,
                    class_name=self._class_names[track_id][dominant_class_id],
                    mean_confidence=sum(confidences) / len(confidences),
                    frame_count=frame_count,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                    first_centroid=self._first_centroid[track_id],
                    last_centroid=self._last_centroid[track_id],
                )
            )
        return summaries
