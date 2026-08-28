from __future__ import annotations

from traffic_intelligence.analytics.congestion import CongestionClassifier
from traffic_intelligence.config.settings import CongestionConfig, CongestionThresholds
from traffic_intelligence.schemas.metrics import CongestionState


def _config(persistence_frames: int = 3) -> CongestionConfig:
    return CongestionConfig(
        density_thresholds=CongestionThresholds(moderate=5, high=10),
        persistence_frames=persistence_frames,
    )


def test_classify_instant_low_density():
    classifier = CongestionClassifier(_config())
    assert classifier.classify_instant(vehicle_density=2) == CongestionState.LOW


def test_classify_instant_moderate_density():
    classifier = CongestionClassifier(_config())
    assert classifier.classify_instant(vehicle_density=5) == CongestionState.MODERATE


def test_classify_instant_high_density():
    classifier = CongestionClassifier(_config())
    assert classifier.classify_instant(vehicle_density=10) == CongestionState.HIGH


def test_update_smooths_a_single_noisy_frame():
    classifier = CongestionClassifier(_config(persistence_frames=3))
    classifier.update(vehicle_density=1)
    classifier.update(vehicle_density=1)
    smoothed_state = classifier.update(vehicle_density=20)

    assert smoothed_state == CongestionState.LOW


def test_update_converges_after_sustained_high_density():
    classifier = CongestionClassifier(_config(persistence_frames=3))
    for _ in range(3):
        state = classifier.update(vehicle_density=20)

    assert state == CongestionState.HIGH
