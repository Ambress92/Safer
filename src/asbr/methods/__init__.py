"""Hydra-instantiable method classes with uniform train/predict interface.

Imports are lazy to avoid pulling in heavy dependencies (aif360, fairlearn,
torch) when only a single method is needed.  Hydra configs should use
fully-qualified ``_target_`` paths (e.g. ``asbr.methods.erm.ERMMethod``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asbr.methods.base import TrainedMethod
    from asbr.methods.erm import ERMMethod
    from asbr.methods.expgrad import ExponGradMethod, ExponGradMLPMethod
    from asbr.methods.aif360_expgrad import AIF360ExponGradMethod, AIF360ExponGradMLPMethod
    from asbr.methods.adversarial import AdversarialDebiasingMethod
    from asbr.methods.prejudice_remover import PrejudiceRemoverMethod
    from asbr.methods.sp_penalty import SPPenaltyMethod
    from asbr.methods.asbr import ASBRMethod

__all__ = [
    "TrainedMethod",
    "ERMMethod",
    "ExponGradMethod",
    "ExponGradMLPMethod",
    "AIF360ExponGradMethod",
    "AIF360ExponGradMLPMethod",
    "AdversarialDebiasingMethod",
    "PrejudiceRemoverMethod",
    "SPPenaltyMethod",
    "ASBRMethod",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TrainedMethod": ("asbr.methods.base", "TrainedMethod"),
    "ERMMethod": ("asbr.methods.erm", "ERMMethod"),
    "ExponGradMethod": ("asbr.methods.expgrad", "ExponGradMethod"),
    "ExponGradMLPMethod": ("asbr.methods.expgrad", "ExponGradMLPMethod"),
    "AIF360ExponGradMethod": ("asbr.methods.aif360_expgrad", "AIF360ExponGradMethod"),
    "AIF360ExponGradMLPMethod": ("asbr.methods.aif360_expgrad", "AIF360ExponGradMLPMethod"),
    "AdversarialDebiasingMethod": ("asbr.methods.adversarial", "AdversarialDebiasingMethod"),
    "PrejudiceRemoverMethod": ("asbr.methods.prejudice_remover", "PrejudiceRemoverMethod"),
    "SPPenaltyMethod": ("asbr.methods.sp_penalty", "SPPenaltyMethod"),
    "ASBRMethod": ("asbr.methods.asbr", "ASBRMethod"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
