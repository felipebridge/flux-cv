"""Sanity-checks a local setup before running the pipeline: Python version,
config validity, and CUDA availability. Run with `python scripts/check_environment.py`
after `pip install -e ".[dev]"`."""

from __future__ import annotations

import sys

from traffic_intelligence.config.settings import ConfigError, load_config


def _check_python_version() -> bool:
    ok = sys.version_info >= (3, 11)
    status = "OK" if ok else "FAIL (requires 3.11+)"
    print(f"Python version: {sys.version.split()[0]} [{status}]")
    return ok


def _check_config() -> bool:
    try:
        load_config("configs/default.yaml")
        print("Config load (configs/default.yaml): OK")
        return True
    except ConfigError as exc:
        print(f"Config load (configs/default.yaml): FAIL ({exc})")
        return False


def _check_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA: available ({torch.cuda.get_device_name(0)})")
        else:
            print("CUDA: not available, pipeline will run on CPU")
    except ImportError:
        print('CUDA: torch not installed yet, run `pip install -e ".[dev]"`')


def main() -> int:
    checks = [_check_python_version(), _check_config()]
    _check_cuda()
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
