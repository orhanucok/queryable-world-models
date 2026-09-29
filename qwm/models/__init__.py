"""Compact QWM and the latent-only sibling."""
from qwm.models.latent_only import LatentOnly
from qwm.models.qwm import QWM, trainable_parameters

__all__ = ["QWM", "LatentOnly", "build_model", "trainable_parameters"]


def build_model(cfg: dict):
    """Build QWM or latent-only from a model config mapping."""
    model_cfg = cfg["model"] if "model" in cfg else cfg
    name = str(model_cfg["name"])
    if name == "qwm":
        return QWM(model_cfg)
    if name == "latent_only":
        return LatentOnly(model_cfg)
    raise ValueError(f"unknown model name {name!r}")
