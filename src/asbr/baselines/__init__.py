"""Baseline methods for fairness comparison experiments.

Imports are lazy to avoid pulling in heavy dependencies (aif360, fairlearn)
when only a subset of baselines is needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asbr.baselines.expgrad import ExponGradConfig, ExponGradWrapper, build_sensitive_features
    from asbr.baselines.aif360_expgrad import AIF360ExponGradConfig, AIF360ExponGradWrapper
    from asbr.baselines.adversarial import AdversarialDebiasingConfig, AdversarialDebiasingWrapper
    from asbr.baselines.prejudice_remover import PrejudiceRemoverConfig, PrejudiceRemoverWrapper
    from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

__all__ = [
    "ExponGradConfig",
    "ExponGradWrapper",
    "AIF360ExponGradConfig",
    "AIF360ExponGradWrapper",
    "AdversarialDebiasingConfig",
    "AdversarialDebiasingWrapper",
    "PrejudiceRemoverConfig",
    "PrejudiceRemoverWrapper",
    "PyTorchMLPWrapper",
    "build_sensitive_features",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ExponGradConfig": ("asbr.baselines.expgrad", "ExponGradConfig"),
    "ExponGradWrapper": ("asbr.baselines.expgrad", "ExponGradWrapper"),
    "build_sensitive_features": ("asbr.baselines.expgrad", "build_sensitive_features"),
    "AIF360ExponGradConfig": ("asbr.baselines.aif360_expgrad", "AIF360ExponGradConfig"),
    "AIF360ExponGradWrapper": ("asbr.baselines.aif360_expgrad", "AIF360ExponGradWrapper"),
    "AdversarialDebiasingConfig": ("asbr.baselines.adversarial", "AdversarialDebiasingConfig"),
    "AdversarialDebiasingWrapper": ("asbr.baselines.adversarial", "AdversarialDebiasingWrapper"),
    "PrejudiceRemoverConfig": ("asbr.baselines.prejudice_remover", "PrejudiceRemoverConfig"),
    "PrejudiceRemoverWrapper": ("asbr.baselines.prejudice_remover", "PrejudiceRemoverWrapper"),
    "PyTorchMLPWrapper": ("asbr.baselines.pytorch_mlp_wrapper", "PyTorchMLPWrapper"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
