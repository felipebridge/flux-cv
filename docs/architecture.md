# Architecture notes

Implementation decisions that aren't obvious from reading the code, plus one platform-specific
issue worth knowing about before debugging it from scratch.

## Why no package has a facade `__init__.py`

Every `__init__.py` under `src/traffic_intelligence/` is empty. `pipeline.video_source.VideoSource`
has no machine-learning dependency at all (just OpenCV), while `pipeline.runner.PipelineRunner` and
`tracking.ultralytics_tracker` pull in the full tracking stack (Ultralytics, and transitively
PyTorch). Python always runs a package's `__init__.py` before any of its submodules, so if it
re-exported symbols for convenience, importing `pipeline.video_source` alone would still execute
`pipeline/__init__.py` first — forcing a PyTorch import just to open a video file. The same applies
to `tracking`: `track_accumulator.py` has zero ML dependencies, but `tracking/__init__.py` used to
eagerly import `UltralyticsTracker` for a re-export nothing in the codebase used, so importing the
accumulator alone silently loaded PyTorch anyway. Callers import each submodule directly —
`traffic_intelligence.pipeline.video_source`, `traffic_intelligence.tracking.track_accumulator`,
etc. — instead of relying on package-level re-exports.

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

`TrackAccumulator.is_confirmed` exposes the same duration+visibility check `finalize` uses, but
queryable per track *while the video is still being processed*. `pipeline.runner` filters
detections down to this confirmed set once per frame and uses that same filtered list for
*everything* the frame produces: the boxes drawn, the trails drawn, the live Vehicles/People
numbers, and the value fed to `CongestionClassifier`. Using one filtered list for all of it, not
separate ones, is what keeps the on-screen number honest — it's not a running total to be
compared to the boxes, it's a live count *of* the boxes, so it can never show a number the video
doesn't visibly back up. Before this, the panel counted raw per-frame detections while the boxes
drawn were also raw, unrelated to that count's own filtering; a detector misfire could each
inflate the number for a frame or two independent of what was drawn. A newly-appeared real object
now has no box and doesn't count for its first `min_track_seconds`, same delay it would see in
the final summary — the live numbers and `summary.json` count by the same rule.

## Why traffic level is based on density, not speed

Congestion is classified purely from how many vehicles are visible in the frame at once
(`CongestionClassifier`), smoothed over `congestion.persistence_frames` so a momentary spike or
dip doesn't flip the reported level. The value reported for the whole video (`traffic_level` in
`summary.json`) is the *most common* per-frame level across the run, not just whatever it
happened to be on the last frame. Per-vehicle speed (below) is reported separately and does not
feed into this classification.

## How per-vehicle speed is estimated, and why it's an approximation

There's no camera calibration for an arbitrary input video — no known distance between two
points in the frame, no ground-plane homography. `analytics.speed.SpeedEstimator` works around
that by assuming a typical real-world width per vehicle class (`speed.reference_widths_m`, e.g.
a car is ~1.8m wide) and inferring a pixels-per-meter scale from the median observed
bounding-box width across confirmed vehicles, once at least `speed.min_calibration_samples`
have been seen. That scale converts each track's pixel displacement between timestamped
positions into km/h.

This is a real estimate, not a measurement from a calibrated sensor: it doesn't model
perspective (a car moving directly toward/away from the camera covers fewer on-screen pixels per
meter than one moving across it) or lens distortion, and it's only as good as the assumed
reference width for whatever's actually in frame. Treat reported speeds as indicative, not
citation-grade.

`TrackAccumulator._speed_summary` (used at `finalize()`) computes speed from consecutive
segments of a track's *full* position history, not a straight line from its first position to
its last — a merely-curved or perspective-foreshortened path would otherwise understate speed.
`TrackAccumulator.current_speed_kmh` (used for the live on-screen label) instead looks at only a
short trailing window, and is called mid-video, before `SpeedEstimator` has necessarily seen
`min_calibration_samples` — so the live number is coarser early in a video and sharpens as
calibration accumulates, while the value in `tracks.csv`/`summary.json` (computed after the full
video is processed) uses the final, most-calibrated scale throughout.

A single noisy segment (a stray detection-box jitter, or a frame where camera-motion
compensation slipped) produces one implausibly high segment speed among otherwise-normal ones.
Reporting that raw value as "max speed" reads as broken, not as a real burst of speed, so
`max_speed_kmh` is the 90th percentile of a track's segment speeds rather than the literal
maximum — it keeps a genuine sustained fast segment while dropping a one-frame outlier.

## Excluding parked vehicles: motion compensation, and why it isn't exact

A parked car isn't stationary on screen if the camera is handheld: `analytics.motion_compensation.
CameraMotionEstimator` estimates the camera's own frame-to-frame motion from sparse optical flow
on background features (RANSAC-fit, so points sitting on moving vehicles don't bias it), and
`pipeline.runner` feeds every detection's centroid through it before handing that position to
`TrackAccumulator`. `TrackAccumulator._has_moved_enough` then requires a vehicle's *compensated*
position to stray at least `tracking.min_movement_ratio` of the frame diagonal from where it
started before counting it as real traffic; below that, it's parked or stopped, not passing
through, and is excluded from both the live boxes/counts and the final summary. This never
applies to pedestrians — a person standing still is still a pedestrian.

This compensation is relative, frame-to-frame visual odometry, not absolute positioning — each
frame's estimate carries a little error, and over a long continuous shot those errors compound
into a residual drift with no natural ceiling. On a ~22s handheld clip this showed up as ~200-350px
of apparent motion for objects that were provably parked the entire time (several independent,
genuinely-stationary vehicles drifted by almost exactly the same amount, which is the signature of
uncorrected camera drift, not per-object noise). Because that drift grows with elapsed time while
`min_movement_ratio` is tuned for the few-second window most real vehicles are visible for,
`TrackAccumulator` applies a stricter bar (`_LONG_TRACK_MOVEMENT_MULTIPLIER`, currently 3x) once a
track has been continuously visible for `_LONG_TRACK_SECONDS` (15s): real through-traffic isn't
usually in frame that long from one camera position, so a vehicle that's still there is almost
certainly parked, no matter how much the compensation error alone could explain. This is a
targeted workaround for a specific failure mode observed in testing, not a general fix for visual
odometry drift -- a fixed camera (no motion to estimate or compensate) or a much shorter clip
wouldn't need it.

A tried-and-reverted alternative: classifying each vehicle by whether *that specific box's* own
feature points matched the RANSAC-fit background transform frame-by-frame (rather than judging
cumulative displacement). It sounded more robust in principle -- no drift to accumulate, since
each frame's classification is judged fresh -- but on a real multi-vehicle street scene it
undercounted moving vehicles by ~70% in testing. When most on-screen motion (many cars, all
roughly following the camera's pan) is similar, RANSAC's "background" fit gets contaminated by
real vehicle motion, so genuinely moving cars get misclassified as background-consistent. It
needs the moving foreground to be a small, clearly-different-motion minority of the tracked
points to work, which doesn't hold for anything but light, sparse traffic. Don't reintroduce this
without solving that failure mode first.
