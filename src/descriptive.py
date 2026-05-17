"""
descriptive.py — Stage [07] orchestrator

Runs the three descriptive sub-scripts in sequence:
  1. descriptive_acoustic.py
  2. descriptive_neural.py
  3. descriptive_cross.py

This is NOT a Snakemake node. Each sub-script is a separate Snakemake rule and
can be invoked independently. This orchestrator exists for manual end-to-end
runs and for parity with stage [06].
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "src/descriptive_acoustic.py",
    "src/descriptive_neural.py",
    "src/descriptive_cross.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    for s in SCRIPTS:
        print(f"\n=== Running {s} ===", flush=True)
        cmd = [sys.executable, s, "--config", str(args.config)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {s} returned {result.returncode}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()