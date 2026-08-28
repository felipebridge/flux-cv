from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from traffic_intelligence.schemas.track import TrackSummary

_TRACK_COLUMNS = [
    "track_id",
    "class_id",
    "class_name",
    "mean_confidence",
    "frame_count",
    "first_frame",
    "last_frame",
    "first_timestamp",
    "last_timestamp",
    "first_centroid_x",
    "first_centroid_y",
    "last_centroid_x",
    "last_centroid_y",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tracks_to_dataframe(summaries: list[TrackSummary]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        row = summary.model_dump()
        first_x, first_y = row.pop("first_centroid")
        last_x, last_y = row.pop("last_centroid")
        row["first_centroid_x"] = first_x
        row["first_centroid_y"] = first_y
        row["last_centroid_x"] = last_x
        row["last_centroid_y"] = last_y
        rows.append(row)
    return pd.DataFrame(rows, columns=_TRACK_COLUMNS)


def write_tracks_csv(summaries: list[TrackSummary], path: str | Path) -> None:
    path = Path(path)
    _ensure_parent(path)
    tracks_to_dataframe(summaries).to_csv(path, index=False)


def write_json(data: BaseModel | dict, path: str | Path) -> None:
    path = Path(path)
    _ensure_parent(path)
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
