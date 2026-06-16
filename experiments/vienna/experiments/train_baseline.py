"""Legacy wrapper for the config-driven Vienna training CLI."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from factual_recall.cli import main


if __name__ == "__main__":
    args = sys.argv[1:] or ["--config", str(PROJECT_ROOT / "configs/runs/vienna_baseline.toml")]
    main(["train", *args])
