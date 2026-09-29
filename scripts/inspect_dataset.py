"""Validate every episode under a collection directory and print statistics.

    python scripts/inspect_dataset.py --data data/debug

Exits non-zero on the first corrupt or shape-invalid episode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data.validate import EpisodeValidationError, format_report, inspect_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    try:
        stats = inspect_dataset(data_dir)
    except (EpisodeValidationError, FileNotFoundError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(format_report(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
