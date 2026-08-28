from __future__ import annotations

import math

from traffic_intelligence.schemas.track import TrackSummary


def stitch_fragmented_tracks(
    summaries: list[TrackSummary],
    frame_diagonal: float,
    max_gap_seconds: float,
    max_centroid_distance_ratio: float,
) -> list[TrackSummary]:
    """Merges tracks that likely represent one physical object whose tracker
    ID switched under occlusion, rather than two distinct objects.

    A track is only merged into an earlier one of the same class when both
    hold: it starts shortly after the earlier one was last seen (within
    `max_gap_seconds` — a real re-acquisition, not two simultaneous
    objects), and it reappears close to where the earlier one disappeared
    (within `max_centroid_distance_ratio` of the frame diagonal — a
    genuinely new object entering the scene would not, in general, start
    exactly where another just vanished). Both conditions together are the
    signature of a tracker ID switch; either alone is common and would merge
    unrelated objects.
    """
    max_distance = frame_diagonal * max_centroid_distance_ratio

    by_class: dict[str, list[TrackSummary]] = {}
    for summary in summaries:
        by_class.setdefault(summary.class_name, []).append(summary)

    stitched: list[TrackSummary] = []
    for class_summaries in by_class.values():
        chains = list(sorted(class_summaries, key=lambda s: s.first_frame))
        merged_chains: list[TrackSummary] = []
        for candidate in chains:
            best_index = _best_chain_to_extend(merged_chains, candidate, max_gap_seconds, max_distance)
            if best_index is None:
                merged_chains.append(candidate)
            else:
                merged_chains[best_index] = _merge(merged_chains[best_index], candidate)
        stitched.extend(merged_chains)
    return stitched


def _best_chain_to_extend(
    chains: list[TrackSummary],
    candidate: TrackSummary,
    max_gap_seconds: float,
    max_distance: float,
) -> int | None:
    best_index: int | None = None
    best_distance = math.inf
    for index, chain in enumerate(chains):
        gap = candidate.first_timestamp - chain.last_timestamp
        if gap <= 0 or gap > max_gap_seconds:
            continue
        distance = math.dist(chain.last_centroid, candidate.first_centroid)
        if distance <= max_distance and distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _merge(earlier: TrackSummary, later: TrackSummary) -> TrackSummary:
    total_frames = earlier.frame_count + later.frame_count
    mean_confidence = (
        earlier.mean_confidence * earlier.frame_count + later.mean_confidence * later.frame_count
    ) / total_frames
    return earlier.model_copy(
        update={
            "frame_count": total_frames,
            "last_frame": later.last_frame,
            "last_timestamp": later.last_timestamp,
            "last_centroid": later.last_centroid,
            "mean_confidence": mean_confidence,
        }
    )
