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

Tracks vehicles and pedestrians in traffic video (each counted once, not once per frame),
classifies congestion as LOW / MODERATE / HIGH, and renders an annotated video plus a
Streamlit dashboard over the results.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"

python -m traffic_intelligence run --input data/raw/avenue.mp4
python -m traffic_intelligence dashboard
```

Drop your video in `data/raw/`. YOLO weights download automatically on first run; CUDA is used
if available, CPU otherwise — no config change needed.

```
outputs/videos/<name>_annotated.mp4   outputs/tracks/tracks.csv   outputs/analytics/*.json
```

## Configuration

All parameters — thresholds, tracker choice, congestion cutoffs — live in
[`configs/default.yaml`](configs/default.yaml), validated by Pydantic on load. See
[`docs/architecture.md`](docs/architecture.md) for the reasoning behind them.

## Stack

Python 3.11+ · Ultralytics YOLO (ByteTrack/BoT-SORT) · OpenCV · Pydantic · Pandas ·
Streamlit/Altair · pytest

## Limitations

No labeled ground truth ships with this project, so results are counts, not precision/recall.
Occlusion can still cause a miss or double-count, and congestion thresholds are density cutoffs
tuned per camera, not a universal number.

## Testing

`pytest -q` — config validation, tracking/stitching, and metrics aggregation, no GPU required.

## License

MIT — see [LICENSE](LICENSE).
