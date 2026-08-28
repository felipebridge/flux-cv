from __future__ import annotations

from traffic_intelligence.config.settings import DeviceType


def resolve_device(requested: DeviceType) -> str:
    if requested == DeviceType.CPU:
        return "cpu"
    if requested == DeviceType.CUDA:
        return "cuda"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
