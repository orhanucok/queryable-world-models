"""Logging helpers: console logging, JSONL metric logs, environment stamping.

Every training/eval run is stamped with: the exact config used, package
versions, the git commit hash when available, and a timestamp (spec section
15: log package versions; log git commit hash when available).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure root once with a uniform format."""
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, stream=sys.stderr)
        _configured = True
    return logging.getLogger(name)


def _module_version(module_name: str) -> Optional[str]:
    try:
        import importlib.metadata as _md

        try:
            return _md.version(module_name)
        except _md.PackageNotFoundError:
            return None
    except Exception:
        return None


def log_package_versions() -> Dict[str, str]:
    """Collect package versions relevant to reproducibility (best effort)."""
    import platform

    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("numpy", "yaml", "torch", "pybullet", "pygame", "imageio", "matplotlib"):
        v = _module_version(mod if mod != "yaml" else "PyYAML")
        if v is not None:
            versions[mod] = v
    return versions


def git_commit_hash() -> Optional[str]:
    """Short git commit hash of the current repo, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


class JsonlLogger:
    """Append-only JSONL metric/event logger (one JSON object per line)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("time", datetime.now(timezone.utc).isoformat())
        self._fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def stamp_run(run_dir: Path, cfg: Mapping[str, Any], extra: Optional[dict] = None) -> Path:
    """Write ``run_meta.json`` (config snapshot + versions + git hash) into a run dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": dict(cfg),
        "versions": log_package_versions(),
        "git_commit": git_commit_hash(),
    }
    if extra:
        meta["extra"] = dict(extra)
    out = run_dir / "run_meta.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    return out
