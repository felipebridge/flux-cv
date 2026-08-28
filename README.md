<h1 align="center">FLUX Computer Vision</h1>

<p align="center">
Computer vision pipeline that tracks vehicles and people in traffic footage and reports
counts + congestion level.
</p>

<p align="center">
  <img src="docs/assets/screenshot.jpg" alt="Annotated traffic video with vehicle/pedestrian boxes, track IDs, and the live summary panel" width="480">
</p>

## What it does

- Detects and tracks vehicles (car, bus, truck, motorcycle, bicycle) and pedestrians, each
  counted once as a distinct object — not once per frame.
- Classifies traffic congestion (LOW / MODERATE / HIGH) from vehicle density, smoothed over time.
- Renders an annotated video: boxes, track IDs, and a live Vehicles / People / Traffic panel.
- Ships a Streamlit dashboard over the exported results.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

CUDA is used automatically if available; otherwise it runs on CPU with no config change.

## Run

```bash
python -m traffic_intelligence run --input data/raw/avenue.mp4
python -m traffic_intelligence dashboard
```

Put your video at `data/raw/<name>.mp4` first. Ultralytics downloads the configured YOLO
checkpoint (`yolo11m.pt` by default) automatically on first run.

`analyze --input outputs/tracks/tracks.csv` recomputes counts from a previous run without
re-processing the video.

## Output

```
outputs/
├── videos/<name>_annotated.mp4
├── tracks/tracks.csv
└── analytics/{metrics.json, summary.json}
```

## Configuration

All tunable parameters — detection thresholds, tracker choice, congestion density cutoffs — live
in `configs/default.yaml`, validated by Pydantic on load. See
[`docs/architecture.md`](docs/architecture.md) for the reasoning behind the defaults, including
the Windows `torch`/`pandas` import-order workaround if you hit a DLL init error.

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
