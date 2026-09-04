from __future__ import annotations

import cv2
import numpy as np

_MAX_CORNERS = 600
_QUALITY_LEVEL = 0.008
_MIN_DISTANCE = 8
_MIN_TRACKED_POINTS = 6
_LK_WIN_SIZE = (21, 21)
_LK_MAX_LEVEL = 3


class CameraMotionEstimator:
    """Estimates the camera's own frame-to-frame motion (pan/rotation/handshake) from sparse
    optical flow on background features, and exposes a running transform that maps a raw
    pixel coordinate in the current frame back into frame 0's coordinate system.

    Handheld video is never perfectly static: everything in frame -- including a parked car --
    drifts on screen as the camera moves. Without compensating for that, a stationary vehicle
    looks like it's moving and every vehicle's estimated speed is off by however much the
    camera itself moved that frame. This estimates that camera motion from the *background*
    (via RANSAC, so a handful of points sitting on moving vehicles don't bias the fit) and lets
    callers undo it before computing displacement or speed.

    This is *relative* visual odometry, not absolute positioning: each frame's estimate carries
    a little error, and composing hundreds of frames of that error accumulates into a real,
    unbounded drift over a long shot (see docs/architecture.md).
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._cumulative_transform = np.eye(2, 3, dtype=np.float64)  # current frame -> frame 0

    def update(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return

        prev_points = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=_MAX_CORNERS, qualityLevel=_QUALITY_LEVEL, minDistance=_MIN_DISTANCE
        )
        if prev_points is None or len(prev_points) < _MIN_TRACKED_POINTS:
            self._prev_gray = gray
            return

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_points, None, winSize=_LK_WIN_SIZE, maxLevel=_LK_MAX_LEVEL
        )
        valid = status.reshape(-1) == 1
        prev_valid = prev_points[valid]
        curr_valid = curr_points[valid]

        if len(prev_valid) >= _MIN_TRACKED_POINTS:
            # Maps this frame's coordinates onto the previous frame's -- i.e. the camera's own
            # motion since the last frame. RANSAC keeps a minority of points sitting on moving
            # vehicles from skewing the background-motion estimate.
            frame_to_previous, _ = cv2.estimateAffinePartial2D(curr_valid, prev_valid, method=cv2.RANSAC)
            if frame_to_previous is not None:
                self._cumulative_transform = self._compose(self._cumulative_transform, frame_to_previous)

        self._prev_gray = gray

    @staticmethod
    def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        outer_3x3 = np.vstack([outer, [0.0, 0.0, 1.0]])
        inner_3x3 = np.vstack([inner, [0.0, 0.0, 1.0]])
        return (outer_3x3 @ inner_3x3)[:2, :]

    def to_reference_frame(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        transformed = self._cumulative_transform @ np.array([x, y, 1.0])
        return float(transformed[0]), float(transformed[1])
