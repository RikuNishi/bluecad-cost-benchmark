#!/usr/bin/env python3
"""Print the environment facts that decide which experiments can run here.

Works anywhere (standard library only). Run it on the dev machine and again
on the GPU machine before the first measured run, and keep the output with
the run's metadata.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import os
import platform
import shutil
import sys


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    print(f"cpu cores: {os.cpu_count()}")
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"disk free (cwd): {free_gb:.1f} GB")

    try:
        import torch
    except ImportError:
        print("torch: NOT installed (CPU-only dev environment)")
        print("gpu: none usable from here")
        return 0

    print(f"torch: {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"gpu[{i}]: {props.name}  VRAM {props.total_memory / 1e9:.1f} GB"
                f"  bf16 support: {torch.cuda.is_bf16_supported()}"
            )
    else:
        print("gpu: torch installed but CUDA not available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
