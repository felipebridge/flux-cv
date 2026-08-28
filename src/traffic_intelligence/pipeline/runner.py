from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from traffic_intelligence.analytics.congestion import CongestionClassifier
from traffic_intelligence.analytics.metrics import compute_traffic_metrics
from traffic_intelligence.config.settings import PipelineConfig
from traffic_intelligence.pipeline.video_source import VideoSource
from traffic_intelligence.schemas.metrics import CongestionState, TrafficMetrics
from traffic_intelligence.schemas.track import TrackSummary
from traffic_intelligence.tracking.track_accumulator import TrackAccumulator
from traffic_intelligence.tracking.track_stitcher import stitch_fragmented_tracks
from traffic_intelligence.tracking.ultralytics_tracker import UltralyticsTracker
from traffic_intelligence.utils.logging import get_logger
from traffic_intelligence.visualization.annotator import FrameAnnotator

logger = get_logger("pipeline.runner")


@dataclass
class RunResult:
    track_summaries: list[TrackSummary]
    metrics: TrafficMetrics
    annotated_video_path: Path | None


class PipelineRunner:
    """Orchestrates the video-to-counts pipeline: tracking, per-frame vehicle
    and person counting, congestion classification, and annotated-video
    rendering."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tracker = UltralyticsTracker(config.detection, config.tracking)
        self._accumulator = TrackAccumulator(
            config.tracking.min_track_seconds, config.tracking.min_visibility_ratio
        )
        self._congestion_classifier = CongestionClassifier(config.congestion)
        self._annotator = FrameAnnotator()
        self._traffic_level_history: list[CongestionState] = []

    def run(self, input_path: str | Path, output_dir: str | Path) -> RunResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        video_writer: cv2.VideoWriter | None = None
        annotated_video_path: Path | None = None
        last_frame_index = -1
        last_timestamp = 0.0
        frame_diagonal = 0.0

        with VideoSource(input_path, self._config.video.fps_override) as source:
            frame_diagonal = math.hypot(source.frame_width, source.frame_height)
            if self._config.output.save_annotated_video:
                videos_dir = output_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                annotated_video_path = videos_dir / f"{input_path.stem}_annotated.mp4"
                video_writer = cv2.VideoWriter(
                    str(annotated_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source.fps,
                    (source.frame_width, source.frame_height),
                )

            for frame_index, timestamp, frame in source.frames():
                tracked = self._tracker.track(frame, frame_index, timestamp)
                for detection in tracked:
                    self._accumulator.add(detection)

                person_class = self._config.detection.person_class
                confirmed = [d for d in tracked if self._accumulator.is_confirmed(d.track_id)]
                vehicle_count = sum(1 for d in confirmed if d.class_name != person_class)
                person_count = sum(1 for d in confirmed if d.class_name == person_class)

                traffic_level = self._congestion_classifier.update(vehicle_count)
                self._traffic_level_history.append(traffic_level)

                if video_writer is not None:
                    trails = {d.track_id: self._accumulator.trail(d.track_id) for d in confirmed}
                    annotated_frame = self._annotator.annotate(
                        frame, confirmed, trails, vehicle_count, person_count, traffic_level
                    )
                    video_writer.write(annotated_frame)

                last_frame_index = frame_index
                last_timestamp = timestamp

            if video_writer is not None:
                video_writer.release()

        track_summaries = self._accumulator.finalize()
        track_summaries = stitch_fragmented_tracks(
            track_summaries,
            frame_diagonal=frame_diagonal,
            max_gap_seconds=self._config.tracking.reid_max_gap_seconds,
            max_centroid_distance_ratio=self._config.tracking.reid_max_centroid_distance_ratio,
        )
        overall_level = self._overall_traffic_level()
        metrics = compute_traffic_metrics(
            track_summaries,
            overall_level,
            video_duration_s=last_timestamp,
            frames_processed=last_frame_index + 1,
        )

        logger.info(
            "Pipeline finished: %d vehicles, %d pedestrians, traffic level=%s",
            metrics.total_vehicles,
            metrics.total_pedestrians,
            overall_level.value,
        )

        return RunResult(
            track_summaries=track_summaries,
            metrics=metrics,
            annotated_video_path=annotated_video_path,
        )

    def _overall_traffic_level(self) -> CongestionState:
        if not self._traffic_level_history:
            return CongestionState.LOW
        most_common, _ = Counter(self._traffic_level_history).most_common(1)[0]
        return most_common
