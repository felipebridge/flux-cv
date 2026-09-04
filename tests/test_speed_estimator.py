from __future__ import annotations

import pytest

from traffic_intelligence.analytics.speed import SpeedEstimator


def test_not_calibrated_before_minimum_samples():
    estimator = SpeedEstimator(min_calibration_samples=5)
    for _ in range(4):
        estimator.observe("car", 100.0)

    assert estimator.is_calibrated is False
    assert estimator.meters_per_pixel is None
    assert estimator.speed_kmh(100.0, 1.0) is None


def test_calibrates_after_minimum_samples():
    estimator = SpeedEstimator(min_calibration_samples=3)
    for _ in range(3):
        estimator.observe("car", 90.0)  # 90px == 1.8m -> 0.02 m/px

    assert estimator.is_calibrated is True
    assert estimator.meters_per_pixel == pytest.approx(0.02)


def test_ignores_unknown_class_and_degenerate_widths():
    estimator = SpeedEstimator(min_calibration_samples=1)
    estimator.observe("unknown_class", 100.0)
    estimator.observe("car", 0.0)

    assert estimator.is_calibrated is False


def test_speed_kmh_converts_pixel_distance_to_kmh():
    estimator = SpeedEstimator(min_calibration_samples=1)
    estimator.observe("car", 90.0)  # meters_per_pixel = 0.02

    # 100px in 1s -> 2m in 1s -> 2 m/s -> 7.2 km/h
    assert estimator.speed_kmh(100.0, 1.0) == pytest.approx(7.2)


def test_speed_kmh_none_for_zero_duration():
    estimator = SpeedEstimator(min_calibration_samples=1)
    estimator.observe("car", 90.0)

    assert estimator.speed_kmh(100.0, 0.0) is None


def test_uses_custom_reference_widths():
    estimator = SpeedEstimator(reference_widths_m={"bus": 2.5}, min_calibration_samples=1)
    estimator.observe("bus", 125.0)  # 125px == 2.5m -> 0.02 m/px

    assert estimator.meters_per_pixel == pytest.approx(0.02)
