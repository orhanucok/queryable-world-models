"""Episode storage, loud validation, and the episode-level 80/10/10 split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from qwm.data import schema
from qwm.data.storage import episode_path, load_episode, save_episode
from qwm.data.validate import EpisodeValidationError, validate_episode
from qwm.teleop.gamepad import controller_available, read_action


def _episode(n: int, episode_id: int = 0, time_steps: int | None = None):
    time_steps = schema.EPISODE_LENGTH if time_steps is None else time_steps
    res = schema.DEFAULT_RES
    rng = np.random.default_rng(episode_id + 3)
    obj = rng.uniform(-0.2, 0.2, size=(n, 3)).astype(np.float32)
    obj[:, 2] = 0.78
    query = schema.build_query(
        np.array([0.0, 0.05, 0.92], dtype=np.float32),
        obj,
        np.ones(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.ones(n, dtype=np.float32),
    )
    arrays = {
        "obs_rgb": rng.integers(0, 255, size=(time_steps, 3, res, res), dtype=np.uint8),
        "action": rng.uniform(-1.0, 1.0, size=(time_steps, 4)).astype(np.float32),
        "reward": np.zeros(time_steps, dtype=np.float32),
        "done": np.zeros(time_steps, dtype=np.bool_),
        "task_id": np.full(time_steps, schema.TASK_PUSH_TO_XY, dtype=np.int64),
        "task_goal": np.tile(np.array([0.0, 0.21, -0.3, -1.0], dtype=np.float32), (time_steps, 1)),
        "grip_xyz": np.tile(query["grip_xyz"], (time_steps, 1)),
        "obj_xyz": np.tile(query["obj_xyz"], (time_steps, 1, 1)),
        "obj_upright": np.tile(query["obj_upright"], (time_steps, 1)),
        "obj_contact": np.tile(query["obj_contact"], (time_steps, 1)),
        "obj_on_table": np.tile(query["obj_on_table"], (time_steps, 1)),
        "object_valid_mask": np.tile(query["object_valid_mask"], (time_steps, 1)),
        "target_object_index": np.zeros(time_steps, dtype=np.int64),
        "episode_id": np.full(time_steps, episode_id, dtype=np.int64),
        "timestep": np.arange(time_steps, dtype=np.int64),
    }
    arrays["done"][-1] = True
    meta = {
        "collection_source": "mixed:scripted",
        "seed": 11,
        "n_objects": n,
        "task_id": schema.TASK_PUSH_TO_XY,
        "t2_mode": None,
        "success": False,
        "res": res,
    }
    return arrays, meta


def test_save_load_roundtrip_keeps_pos_origin_pad(tmp_path):
    arrays, meta = _episode(4, episode_id=2)
    path = episode_path(tmp_path, 2)
    save_episode(path, arrays, meta)
    loaded = load_episode(path)
    np.testing.assert_array_equal(loaded["obs_rgb"], arrays["obs_rgb"])
    np.testing.assert_allclose(loaded["obj_xyz"], arrays["obj_xyz"])
    pad = loaded["obj_xyz"][:, 4:]
    np.testing.assert_allclose(pad, np.broadcast_to(schema.POS_ORIGIN, pad.shape))
    assert loaded["meta"]["collection_source"] == "mixed:scripted"
    assert loaded["meta"]["t2_mode"] is None
    summary = validate_episode(path)
    assert summary["episode_id"] == 2
    assert summary["n_objects"] == 4
    np.testing.assert_allclose(summary["action"], arrays["action"])


def test_twenty_object_episode_has_no_pad(tmp_path):
    arrays, meta = _episode(schema.NMAX, episode_id=0)
    path = episode_path(tmp_path, 0)
    save_episode(path, arrays, meta)
    summary = validate_episode(path)
    assert summary["n_objects"] == schema.NMAX
    assert load_episode(path)["object_valid_mask"].all()


def test_pad_zeros_are_rejected(tmp_path):
    arrays, meta = _episode(4, episode_id=1)
    arrays["obj_xyz"][:, 4:] = 0.0
    path = episode_path(tmp_path, 1)
    save_episode(path, arrays, meta)
    with pytest.raises(EpisodeValidationError, match="POS_ORIGIN"):
        validate_episode(path)


def test_short_episode_and_bad_action_fail_loud(tmp_path):
    arrays, meta = _episode(2, episode_id=0, time_steps=3)
    path = episode_path(tmp_path, 0)
    save_episode(path, arrays, meta)
    with pytest.raises(EpisodeValidationError, match="EPISODE_LENGTH"):
        validate_episode(path)

    arrays, meta = _episode(2, episode_id=5)
    arrays["action"][0, 0] = np.float32(1.5)
    path = episode_path(tmp_path, 5)
    save_episode(path, arrays, meta)
    with pytest.raises(EpisodeValidationError, match="action"):
        validate_episode(path)


def test_task_goal_float32_values_roundtrip(tmp_path):
    arrays, meta = _episode(1, episode_id=6)
    path = episode_path(tmp_path, 6)
    save_episode(path, arrays, meta)
    loaded = load_episode(path)
    np.testing.assert_allclose(
        loaded["task_goal"][0],
        np.array([0.0, 0.21, -0.3, -1.0], dtype=np.float32),
    )
    validate_episode(path)


def test_gamepad_check_does_not_crash_without_a_controller():
    assert isinstance(controller_available(), bool)
    if controller_available():
        action = read_action()
        assert action.shape == (schema.ACTION_DIM,)
        assert action.dtype == np.float32
    else:
        with pytest.raises(RuntimeError, match="no gamepad"):
            read_action()


def test_split_counts_and_reproducible_isolation():
    from qwm.data.splits import assign_split, assert_no_leakage, split_counts

    assert split_counts(10) == {"train": 8, "val": 1, "test": 1}
    assert split_counts(200) == {"train": 160, "val": 20, "test": 20}
    first = assign_split(list(range(10)), seed=0)
    second = assign_split(list(range(10)), seed=0)
    assert first == second
    assert_no_leakage(first)
    groups = [set(first["episode_ids"][name]) for name in ("train", "val", "test")]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(range(10))


def test_debug_split_manifest_and_sequence_isolation():
    import torch

    from qwm.data.dataset import SequenceDataset, load_split_datasets, make_loader, normalize_image
    from qwm.data.splits import load_manifest, manifest_path
    from qwm.data.storage import load_episode
    from qwm.utils.config import get as cfg_get
    from qwm.utils.config import load_config

    debug = ROOT / "data" / "debug"
    manifest = load_manifest(manifest_path(debug))
    assert manifest["counts"] == {"train": 8, "val": 1, "test": 1}
    assert manifest["seed"] == 0
    cfg = load_config(ROOT / "configs" / "train.yaml")
    assert cfg_get(cfg, "data.image_norm") == "uint8_div_255"
    length = int(cfg_get(cfg, "data.sequence_length"))
    sets = load_split_datasets(debug, length, manifest)
    id_sets = {name: set(int(i) for i in ds.window_episode_ids()) for name, ds in sets.items()}
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
    assert id_sets["train"] == set(manifest["episode_ids"]["train"])
    assert id_sets["test"] == set(manifest["episode_ids"]["test"])

    sample = sets["train"][0]
    assert sample["obs"].shape == (length, schema.OBS_CHANNELS, schema.DEFAULT_RES, schema.DEFAULT_RES)
    assert sample["obs"].dtype == torch.float32
    assert float(sample["obs"].min()) >= 0.0 and float(sample["obs"].max()) <= 1.0
    assert sample["action"].shape == (length, schema.ACTION_DIM)
    assert sample["query"].shape == (length, schema.QUERY_DIM)
    assert sample["object_valid_mask"].shape == (length, schema.NMAX)
    assert sample["object_valid_mask"].dtype == torch.bool
    assert torch.all(sample["timestep"][1:] - sample["timestep"][:-1] == 1)
    episode_id = int(sample["episode_id"])
    assert episode_id in id_sets["train"]
    raw = load_episode(debug / "episodes" / f"ep_{episode_id:06d}.npz")
    start = int(sample["timestep"][0])
    assert int(sample["timestep"][-1]) < raw["obs_rgb"].shape[0]
    expected_obs = normalize_image(raw["obs_rgb"][start:start + length])
    np.testing.assert_allclose(sample["obs"].numpy(), expected_obs, atol=1e-6)
    expected_query = schema.flatten_query(
        {
            "grip_xyz": raw["grip_xyz"][start],
            "obj_xyz": raw["obj_xyz"][start],
            "obj_upright": raw["obj_upright"][start],
            "obj_contact": raw["obj_contact"][start],
            "obj_on_table": raw["obj_on_table"][start],
        },
        normalized=True,
    )
    np.testing.assert_allclose(sample["query"][0].numpy(), expected_query, atol=1e-5)
    # padded slots are normalized 0, which denormalizes to POS_ORIGIN
    pad = sample["query"][0, schema.OBJ_XYZ].reshape(schema.NMAX, 3)
    n_objects = int(raw["meta"]["n_objects"])
    np.testing.assert_allclose(pad[n_objects:].numpy(), 0.0, atol=1e-5)

    loader = make_loader(sets["train"], batch_size=4, shuffle=False, seed=0)
    batch = next(iter(loader))
    assert batch["obs"].shape[0] == 4
    assert batch["query"].shape == (4, length, schema.QUERY_DIM)
    assert set(int(i) for i in batch["episode_id"]).issubset(id_sets["train"])


def test_sequence_dataset_rejects_short_windows(tmp_path):
    arrays, meta = _episode(2, episode_id=0, time_steps=schema.EPISODE_LENGTH)
    from qwm.data.dataset import SequenceDataset
    from qwm.data.storage import episode_path, save_episode

    save_episode(episode_path(tmp_path, 0), arrays, meta)
    with pytest.raises(ValueError, match="sequence_length"):
        SequenceDataset(tmp_path, [0], sequence_length=1)
