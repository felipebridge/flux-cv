<h1 align="center">FLUX Computer Vision</h1>

<p align="center">
Real-time vehicle & pedestrian tracking with live traffic congestion classification.
</p>

<p align="center">
  <a href="https://github.com/felipebridge/flux-cv/actions/workflows/ci.yml"><img src="https://github.com/felipebridge/flux-cv/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

<p align="center">
  <img src="docs/assets/screenshot.jpg" alt="Annotated traffic video with vehicle/pedestrian boxes, track IDs, and the live summary panel" width="480">
</p>

## What it does

- **Multi-class tracking** — vehicles (car, bus, truck, motorcycle, bicycle) and pedestrians are
  tracked across frames with ByteTrack/BoT-SORT, so each one is counted once as a distinct
  object, not once per frame.
- **Congestion classification** — LOW / MODERATE / HIGH traffic level from live vehicle density,
  smoothed over time so a momentary spike doesn't flip the reading.
- **Annotated video output** — boxes, track IDs, and a live Vehicles / People / Traffic panel
  burned into the output video.
- **Streamlit dashboard** — visualize exported counts and congestion trends without re-running
  the pipeline.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"

python -m traffic_intelligence run --input data/raw/avenue.mp4
python -m traffic_intelligence dashboard
```

Put your video at `data/raw/<name>.mp4` first. Ultralytics downloads the configured YOLO
checkpoint (`yolo11m.pt` by default) automatically on first run. CUDA is used automatically if
available; otherwise it runs on CPU with no config change.

`analyze --input outputs/tracks/tracks.csv` recomputes counts from a previous run without
re-processing the video.

<details>
<summary>Run with Docker instead</summary>

```bash
docker compose run traffic-intelligence
docker compose up dashboard   # http://localhost:8501
```

</details>

## Output

```
outputs/
├── videos/<name>_annotated.mp4
├── tracks/tracks.csv
└── analytics/{metrics.json, summary.json}
```

## Configuration

All tunable parameters live in [`configs/default.yaml`](configs/default.yaml), validated by
Pydantic on load:

| Setting | Controls |
|---|---|
| `detection.confidence_threshold` | minimum score for a box to reach the tracker |
| `tracking.tracker` | `bytetrack` (fast) or `botsort` (+ optional appearance re-ID) |
| `congestion.density_thresholds` | vehicle-count cutoffs for MODERATE / HIGH |

See [`docs/architecture.md`](docs/architecture.md) for the reasoning behind the defaults,
including the Windows `torch`/`pandas` import-order workaround if you hit a DLL init error.

## Stack

Python 3.11+ · Ultralytics YOLO (detection + ByteTrack/BoT-SORT tracking) · OpenCV · Pydantic ·
Pandas · Streamlit/Altair · pytest

## Limitations

- No labeled ground truth ships with this project, so no precision/recall/mAP is reported — only
  counts derived from tracking.
- Occlusion can still cause a real object to be missed or double-counted; appearance-based
  re-identification and track stitching reduce this but don't eliminate it.
- Congestion thresholds are density cutoffs you set per camera — there's no universal "moderate
  traffic" number that applies to every angle and road width.

## Testing

```bash
pytest -q
```

Covers config validation, tracking/stitching logic, and metrics aggregation — runs in seconds,
no GPU or model weights required.

## License

MIT — see [LICENSE](LICENSE).
