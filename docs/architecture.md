# Architecture notes

Implementation decisions that aren't obvious from reading the code, plus one platform-specific
issue worth knowing about before debugging it from scratch.

## Why the pipeline package has no facade `__init__.py`

`pipeline/__init__.py` is intentionally empty. `pipeline.video_source.VideoSource` has no
machine-learning dependency at all (just OpenCV), while `pipeline.runner.PipelineRunner` pulls
in the full tracking stack (Ultralytics, and transitively PyTorch). If the package's
`__init__.py` re-exported both (as every other package in this project does, for convenience),
importing `pipeline.video_source` alone would still execute `pipeline/__init__.py` first —
which would force a PyTorch import just to open a video file. Callers import
`traffic_intelligence.pipeline.video_source` and `traffic_intelligence.pipeline.runner`
directly instead.

## Windows: torch / pandas DLL conflict

On some Windows environments, importing `pandas` before `torch` in the same process causes:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

This is a conflict between the bundled OpenMP/MKL runtimes that pandas' and torch's wheels
each ship on Windows — whichever one initializes its native libraries first in the process
"wins," and the second one's init routine can fail. It reproduces with nothing but:

```python
import pandas
import torch  # fails here, but only if pandas was imported first
```

`cli.py` works around this by structuring the `run` subcommand's local imports so
`pipeline.runner` (which pulls in `tracking.ultralytics_tracker` → `ultralytics` → `torch`) is
imported before `persistence.writers` (which imports `pandas`). Because of this ordering
requirement, `src/traffic_intelligence/cli.py` is exempted from the isort rule in
`pyproject.toml` (`per-file-ignores`) — a strictly alphabetical import order would put
`persistence` before `pipeline` and reintroduce the conflict. Subcommands that never touch
torch (`--help`, `analyze`, `dashboard`) import nothing from the ML stack at all, so they don't
pay the import cost or risk the conflict.

If you still hit this after modifying the CLI, the fix is the same: make sure some module that
imports `torch` (directly or via `ultralytics`) is the first of the two to be imported, anywhere
in the process, before any `pandas` import happens. Reinstalling `torch` or updating to a newer
wheel can also resolve it, since this is ultimately a packaging issue upstream, not a logic bug.

## Why the tracker wraps Ultralytics' ByteTrack/BoT-SORT instead of reimplementing them

Both trackers are Kalman filter + Hungarian assignment, which is easy to get subtly wrong, and
Ultralytics' implementations are already tested against standard MOT benchmarks.

## Why detection and tracking are one call, not two

An earlier version of this project ran a standalone frame-level detector and a separate
tracker. That's redundant: Ultralytics' `.track()` already runs detection internally as part of
producing tracked boxes, so calling a detector first and a tracker second would run the same
YOLO forward pass twice per frame for no benefit. `tracking.UltralyticsTracker` is the only
model-facing component; there is no separate `detection` package.

## Why counting is done via track survival, not frame-by-frame detection counts

`total_vehicles` and `total_pedestrians` count *finalized tracks* (`TrackAccumulator.finalize`),
not raw per-frame detections. A single vehicle produces one detection per frame it's visible in;
counting detections would count the same car dozens of times. Counting is also gated by
`tracking.min_track_seconds` — a track visible for a fraction of a second is more likely a
detector flicker (a false-positive or partially-occluded box that appears and disappears) than a
real, distinct vehicle, so it's dropped before the count is computed rather than inflating it.
This is measured in time, not frame count, so the same setting behaves the same way on a 30fps
and a 60fps video — a frame-count cutoff would silently mean half as much real time on the
faster video. Duration alone accepts a track that spans a long window but was only sporadically
redetected within it, so `tracking.min_visibility_ratio` additionally requires the track to
actually be present in most of the frames within its span.

After filtering, `tracking.stitch_fragmented_tracks` runs once more over the finalized tracks to
merge same-class fragments that are almost certainly the same physical object handed off between
IDs: two tracks merge when the second starts shortly after the first was last seen
(`tracking.reid_max_gap_seconds`) *and* reappears close to where the first disappeared
(`tracking.reid_max_centroid_distance_ratio` of the frame diagonal). This runs after
`TrackAccumulator.finalize`, not inside it, because stitching needs to compare *already-finalized*
tracks against each other (their start/end points), whereas the accumulator only ever sees one
live detection at a time.

## Why traffic level is based on density, not speed

This project doesn't estimate vehicle speed — congestion is classified purely from how many
vehicles are visible in the frame at once (`CongestionClassifier`), smoothed over
`congestion.persistence_frames` so a momentary spike or dip doesn't flip the reported level. The
value reported for the whole video (`traffic_level` in `summary.json`) is the *most common*
per-frame level across the run, not just whatever it happened to be on the last frame.
