"""Stage 05: orchestrator for normalisation.

Runs the two sub-scripts:
  - normalise_acoustic.py  (Lobanov F1/F2/F3 + f0 -> semitones)
  - normalise_neural.py    (PCA d=50 per (model, layer))

Each sub-script is also runnable in standalone for isolated debug / rerun.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_substage(script: Path, config: Path) -> None:
    print(f"\n========== running {script.name} ==========")
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"[normalise] {script.name} failed with exit {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    here = Path(__file__).parent

    run_substage(here / "normalise_acoustic.py", cfg_path)
    run_substage(here / "normalise_neural.py", cfg_path)

    print("\n[normalise] all sub-stages done.")


if __name__ == "__main__":
    main()