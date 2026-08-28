from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DashboardData:
    tracks: pd.DataFrame
    summary: dict
    annotated_video_path: Path | None


def load_dashboard_data(output_dir: Path) -> DashboardData:
    tracks_path = output_dir / "tracks" / "tracks.csv"
    summary_path = output_dir / "analytics" / "summary.json"

    tracks = pd.read_csv(tracks_path) if tracks_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    annotated_video_path = None
    if summary.get("annotated_video_path"):
        candidate = Path(summary["annotated_video_path"])
        if candidate.exists():
            annotated_video_path = candidate

    return DashboardData(tracks=tracks, summary=summary, annotated_video_path=annotated_video_path)
