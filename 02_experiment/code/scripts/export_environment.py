"""Export reproducibility metadata without probing accelerators or devices."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "geotoken3path.environment.v1",
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "PyYAML": package_version("PyYAML"),
            "pytest": package_version("pytest"),
        },
        "local_gpu_probe": "forbidden_not_run",
        "device_inventory": "not_collected",
        "scientific_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
