"""STEP 1 sanity tests: config loading/merging/saving, seeding determinism,
device selection, logging stamping."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from qwm.utils.config import get as cfg_get
from qwm.utils.config import load_config, merge, save_config
from qwm.utils.logging import git_commit_hash, log_package_versions
from qwm.utils.seed import make_rng, set_global_seed, spawn_seeds


def test_merge_nested_override():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    # inputs untouched (no aliasing)
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}
    assert override == {"a": {"b": 10}, "e": 4}


def test_load_config_roundtrip(tmp_path: Path):
    cfg = {"train": {"batch_size": 16, "amp": True}, "seed": 0}
    src = tmp_path / "cfg.yaml"
    with src.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    loaded = load_config(src)
    assert loaded == cfg
    snap = save_config(loaded, tmp_path / "snapshot.json")
    assert snap.is_file()
    import json

    with snap.open("r", encoding="utf-8") as f:
        assert json.load(f) == cfg


def test_load_config_fail_loud(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_config_get_dotted():
    cfg = {"train": {"batch_size": 16}, "seed": 7}
    assert cfg_get(cfg, "train.batch_size") == 16
    assert cfg_get(cfg, "train.missing", "dflt") == "dflt"
    assert cfg_get(cfg, "missing.path") is None


def test_make_rng_deterministic_and_distinct():
    a1 = make_rng(123).random(4)
    a2 = make_rng(123).random(4)
    b = make_rng(124).random(4)
    np.testing.assert_array_equal(a1, a2)
    assert not np.allclose(a1, b)


def test_spawn_seeds_deterministic_distinct_and_domain_separated():
    s1 = spawn_seeds(42, 5, "collect")
    s2 = spawn_seeds(42, 5, "collect")
    s3 = spawn_seeds(42, 5, "planning_eval")
    assert s1 == s2
    assert len(set(s1)) == 5
    assert set(s1) != set(s3)
    assert spawn_seeds(42, 0, "x") == []
    with pytest.raises(ValueError):
        spawn_seeds(42, -1, "x")


def test_set_global_seed_idempotent():
    set_global_seed(999)
    x1 = random.random()
    y1 = np.random.random()
    set_global_seed(999)
    x2 = random.random()
    y2 = np.random.random()
    assert x1 == x2
    assert y1 == y2


def test_get_device_lazy_torch():
    torch = pytest.importorskip("torch")
    from qwm.utils.device import get_device

    d = get_device("auto")
    assert str(d) in ("cpu", "cuda")
    assert str(get_device("cpu")) == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")
    with pytest.raises(ValueError):
        get_device("gpu9")


def test_logging_helpers():
    versions = log_package_versions()
    assert "python" in versions and "numpy" in versions
    # git_commit_hash may legitimately be None outside a repo; here we ARE in
    # a repo after STEP 1's commit, but before the first commit it can be
    # None — accept both, only check the type.
    assert git_commit_hash() is None or isinstance(git_commit_hash(), str)
