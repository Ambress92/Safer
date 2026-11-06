"""Shared Pareto plotting utilities.

This module centralizes:
- method styling
- sweep-key definitions
- results.csv aggregation
- common CLI parsing helpers

It is used by both the legacy AUC Pareto plot and the newer single-panel
utility plots (e.g. log loss vs MSEE).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


AUTO_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
)

AUTO_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "8", "d", "<", ">")


METHOD_COLOR = {
    "ERM": "#333333",
    "SP-Penalty": "#D55E00",
    "EO-Penalty": "#D55E00",
    "EO-Penalty-TwoPhase": "#d4006e",
    "EO-Penalty-Joint": "#D55E00",
    "ASBR-LRT-Soft-EO-TwoPhase": "#7a52c7",
    "ASBR-LRT-Gated-EO-TwoPhase-NoSW": "#8c564b",
    "ASBR-LRT-Joint": "#4e79a7",
    "ASBR-LRT-Joint-EO": "#f28e2b",
    "ASBR-LRT-Gated-SP-Joint": "#59a14f",
    "ASBR-LRT-Gated-SP-TwoPhase": "#1a7340",
    "ASBR-LRT-Gated-EO-Joint": "#F0E442",
    "ASBR-LRT-Gated-EO-TwoPhase": "#c26a00",
    "ASBR-LRT-Gated-EO-Progressive": "#7b3f00",
    "ASBR-LRT-Gated-SP-Progressive": "#2c7bb6",
    "ASBR-LRT-Unsq-Joint": "#9ecae1",
    "ASBR-LRT-Unsq-Joint-EO": "#f5b97a",
    "ASBR-LRT-Unsq-EO-Progressive": "#c9852a",
    "ASBR-LRT-Soft-EO-Progressive": "#5e3c99",
    "fairret-norm2": "#b07aa1",
    "fairret-norm1": "#76b7b2",
    "fairret-norm2-eo": "#564B7C",
    "fairret-norm1-eo": "#009E73",
    "fairret-norm2-sp": "#564B7C",
    "fairret-norm1-sp": "#009E73",
    "ExponGrad": "#f1ce63",
    "ExponGrad-EO": "#c9a400",
    "PrejudiceRemover": "#9c755f",
    "AdvDebiasing-EO": "#CC79A7",
    "ASBR-LRT-Target-EO-TwoPhase": "#006d77",
    "ASBR-LRT-Gated-EO-TwoPhase-NoHinge": "#83c5be",
    "ASBR-LRT-Gated-EO-Joint-NoGate": "#ff7f0e",
    "ASBR-LRT-Gated-EO-Joint-LRTStat": "#9467bd",
    "ASBR-DirectMSEE-EO-Joint": "#009e73",
    "ASBR-DirectMSEE-EO-Joint-RMS": "#0072b2",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail": "#e69f00",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Warmup": "#cc79a7",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-WarmupGuard": "#7f3c8d",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Ramp": "#56b4e9",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-RampGuard": "#009e9e",
    "ASBR-LRT-Gated-EO-Joint-RMS": "#2ca02c",
    "ASBR-LRT-Gated-EO-Joint-OvR": "#1f77b4",
    "ASBR-LRT-Gated-EO-Joint-OvR-RMS": "#8c564b",
    "ASBR-DirectLRT-EO-Joint": "#d62728",
    "ASBR-DirectLRT-EO-Joint-Tail001": "#ff9896",
    "ASBR-DirectLRT-EO-Joint-Tail003": "#c49c94",
    "ASBR-DirectLRT-EO-Joint-Guard": "#6b6ecf",
    "ASBR-DirectLRT-EO-Joint-GuardAll": "#17becf",
    "ASBR-DirectLRT-Joint": "#0072B2",
    "ASBR-DirectLRT-SP-Joint": "#0072B2",
    "ASBR-DirectLRT-Joint-Ramp30": "#bcbd22",
    "ASBR-DirectLRT-TwoPhase": "#8c564b",
    "ASBR-DirectLRT-EO-Joint-RampGuard": "#bcbd22",
}

METHOD_MARKER = {
    "ERM": "*",
    "SP-Penalty": "s",
    "EO-Penalty": "s",
    "EO-Penalty-TwoPhase": "D",
    "EO-Penalty-Joint": "o",
    "ASBR-LRT-Soft-EO-TwoPhase": "D",
    "ASBR-LRT-Gated-EO-TwoPhase-NoSW": "^",
    "ASBR-LRT-Joint": "o",
    "ASBR-LRT-Joint-EO": "D",
    "ASBR-LRT-Gated-SP-Joint": "^",
    "ASBR-LRT-Gated-SP-TwoPhase": "^",
    "ASBR-LRT-Gated-EO-Joint": "v",
    "ASBR-LRT-Gated-EO-TwoPhase": "v",
    "ASBR-LRT-Gated-EO-Progressive": "D",
    "ASBR-LRT-Gated-SP-Progressive": "D",
    "ASBR-LRT-Unsq-Joint": "o",
    "ASBR-LRT-Unsq-Joint-EO": "o",
    "ASBR-LRT-Unsq-EO-Progressive": "D",
    "ASBR-LRT-Soft-EO-Progressive": "D",
    "fairret-norm2": "v",
    "fairret-norm1": "P",
    "fairret-norm2-eo": "v",
    "fairret-norm1-eo": "P",
    "fairret-norm2-sp": "v",
    "fairret-norm1-sp": "P",
    "ExponGrad": "X",
    "ExponGrad-EO": "X",
    "PrejudiceRemover": "<",
    "AdvDebiasing-EO": "h",
    "ASBR-LRT-Target-EO-TwoPhase": "P",
    "ASBR-LRT-Gated-EO-TwoPhase-NoHinge": "^",
    "ASBR-LRT-Gated-EO-Joint-NoGate": "o",
    "ASBR-LRT-Gated-EO-Joint-LRTStat": "D",
    "ASBR-DirectMSEE-EO-Joint": "X",
    "ASBR-DirectMSEE-EO-Joint-RMS": "*",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail": "X",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Warmup": "P",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-WarmupGuard": "H",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Ramp": "8",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-RampGuard": "8",
    "ASBR-LRT-Gated-EO-Joint-RMS": "D",
    "ASBR-LRT-Gated-EO-Joint-OvR": "h",
    "ASBR-LRT-Gated-EO-Joint-OvR-RMS": "d",
    "ASBR-DirectLRT-EO-Joint": "o",
    "ASBR-DirectLRT-EO-Joint-Tail001": "s",
    "ASBR-DirectLRT-EO-Joint-Tail003": "^",
    "ASBR-DirectLRT-EO-Joint-Guard": "P",
    "ASBR-DirectLRT-EO-Joint-GuardAll": "X",
    "ASBR-DirectLRT-Joint": "X",
    "ASBR-DirectLRT-SP-Joint": "X",
    "ASBR-DirectLRT-Joint-Ramp30": "D",
    "ASBR-DirectLRT-TwoPhase": "P",
    "ASBR-DirectLRT-EO-Joint-RampGuard": "D",
}

METHOD_LABEL = {
    "ERM": "ERM",
    "ASBR-DirectLRT-Joint": "SAFER",
    "ASBR-DirectLRT-Joint-NoDenseTail": "SAFER-NoDenseTail",
    "ASBR-DirectLRT-Joint-NoSelectivity": "SAFER-NoSelectivity",
    "ASBR-DirectLRT-Joint-NoThreshSmooth": "SAFER-NoThreshSmooth",
    "SP-Penalty": "EO-Penalty",
    "EO-Penalty": "EO-Penalty",
    "EO-Penalty-TwoPhase": "EO-Penalty TwoPhase",
    "EO-Penalty-Joint": "EO-Penalty",
    "ASBR-LRT-Soft-EO-TwoPhase": "ASBR-LRT-Soft-EO TwoPhase",
    "ASBR-LRT-Gated-EO-TwoPhase-NoSW": "ASBR-Gated-EO TwoPhase, no size weight",
    "ASBR-LRT-Joint": "ASBR-LRT SP",
    "ASBR-LRT-Joint-EO": "ASBR-LRT EO",
    "ASBR-LRT-Gated-SP-Joint": "ASBR-Gated-SP",
    "ASBR-LRT-Gated-SP-TwoPhase": "ASBR-Gated-SP TwoPhase",
    "ASBR-LRT-Gated-EO-Joint": "ASBR-Gated-EO",
    "ASBR-LRT-Gated-EO-TwoPhase": "Gated-EO TwoPhase",
    "ASBR-LRT-Gated-EO-Progressive": "Gated-EO PLT",
    "ASBR-LRT-Gated-SP-Progressive": "ASBR-Gated-SP Progressive",
    "ASBR-LRT-Unsq-Joint": "ASBR-LRT-Unsq SP",
    "ASBR-LRT-Unsq-Joint-EO": "ASBR-LRT-Unsq EO",
    "ASBR-LRT-Unsq-EO-Progressive": "ASBR-LRT-Unsq-EO PLT",
    "ASBR-LRT-Soft-EO-Progressive": "ASBR-LRT-Soft-EO PLT",
    "fairret-norm2": "fairret NormLoss p=2 SP",
    "fairret-norm1": "fairret NormLoss p=1 SP",
    "fairret-norm2-eo": "FAIRRET Norm 2",
    "fairret-norm1-eo": "FAIRRET Norm 1",
    "fairret-norm2-sp": "FAIRRET Norm 2",
    "fairret-norm1-sp": "FAIRRET Norm 1",
    "ExponGrad": "ExponGrad SP",
    "ExponGrad-EO": "ExponGrad EO",
    "PrejudiceRemover": "Prejudice Remover",
    "AdvDebiasing-EO": "Adversarial Debiasing",
    "ASBR-LRT-Target-EO-TwoPhase": "ASBR-LRT-Target-EO TwoPhase",
    "ASBR-LRT-Gated-EO-TwoPhase-NoHinge": "Gated-EO TwoPhase, no hinge",
    "ASBR-LRT-Gated-EO-Joint-NoGate": "ASBR-Gated-EO, no gate",
    "ASBR-LRT-Gated-EO-Joint-LRTStat": "ASBR-Gated-EO, LRT statistic hinge",
    "ASBR-DirectMSEE-EO-Joint": "ASBR Direct-MSEE EO",
    "ASBR-DirectMSEE-EO-Joint-RMS": "ASBR Direct-MSEE EO, RMS",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail": "ASBR Direct-MSEE EO, threshold tail",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Warmup": "ASBR Direct-MSEE EO, threshold tail + warmup",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-WarmupGuard": "ASBR Direct-MSEE EO, threshold tail + warmup + BCE guard",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-Ramp": "ASBR Direct-MSEE EO, threshold tail + lambda ramp",
    "ASBR-DirectMSEE-EO-Joint-ThresholdTail-RampGuard": "Direct-MSEE EO",
    "ASBR-LRT-Gated-EO-Joint-RMS": "ASBR-Gated-EO, RMS aggregation",
    "ASBR-LRT-Gated-EO-Joint-OvR": "ASBR-Gated-EO, overall reference",
    "ASBR-LRT-Gated-EO-Joint-OvR-RMS": "ASBR-Gated-EO, OvR + RMS",
    "ASBR-DirectLRT-EO-Joint": "Direct-LRT MSEE, no tail",
    "ASBR-DirectLRT-EO-Joint-Tail001": "Direct-LRT MSEE, tail=0.001",
    "ASBR-DirectLRT-EO-Joint-Tail003": "Direct-LRT MSEE, tail=0.003",
    "ASBR-DirectLRT-EO-Joint-Guard": "Direct-LRT, BCE guard $\\lambda \\geq 1$",
    "ASBR-DirectLRT-EO-Joint-GuardAll": "Direct-LRT",
    "ASBR-DirectLRT-Joint-Ramp30": "Direct-LRT, $\\lambda$ ramp",
    "ASBR-DirectLRT-TwoPhase": "Direct-LRT TwoPhase",
    "ASBR-DirectLRT-SP-Joint": "SAFER",
    "ASBR-DirectLRT-EO-Joint-RampGuard": "Direct-LRT, $\\lambda$ ramp",
}

# Methods whose sweep parameter is target_reduction rather than lambda.
# _annotate_lambd uses this to show "target=X" instead of "λ=X".
TARGET_REDUCTION_METHODS: frozenset[str] = frozenset({
    "ASBR-LRT-Target-EO-TwoPhase",
})

# Methods whose sweep parameter is alpha rather than lambda.
ALPHA_METHODS: frozenset[str] = frozenset({
    "AdvDebiasing-EO",
})


def _stable_index(value: str, modulus: int) -> int:
    """Return a deterministic index for styling fallback cycles."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def get_method_color(method: str) -> str:
    """Return a curated or deterministic fallback color for a method."""
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    return AUTO_COLORS[_stable_index(method, len(AUTO_COLORS))]


def get_method_marker(method: str) -> str:
    """Return a curated or deterministic fallback marker for a method."""
    if method in METHOD_MARKER:
        return METHOD_MARKER[method]
    return AUTO_MARKERS[_stable_index(method, len(AUTO_MARKERS))]


def get_method_label(method: str) -> str:
    """Return a curated label, falling back to the method directory name."""
    return METHOD_LABEL.get(method, method)


SWEEP_SPECS: dict[str, tuple[str, str | None]] = {
    "sp-sweep": ("SP-Penalty", None),
    "eo-penalty-sweep": ("EO-Penalty", None),
    "eo-penalty-twophase-sweep": ("EO-Penalty-TwoPhase", "eopenaltytwophase"),
    "eo-penalty-joint-sweep": ("EO-Penalty-Joint", "eopenaltyjoint"),
    "soft-eo-twophase-sweep": ("ASBR-LRT-Soft-EO-TwoPhase", None),
    "asbr-sweep": ("ASBR-LRT-Joint", None),
    "asbr-eo-sweep": ("ASBR-LRT-Joint-EO", "asbrlrtjoint"),
    "asbr-unsq-sweep": ("ASBR-LRT-Unsq-Joint", None),
    "asbr-unsq-eo-sweep": ("ASBR-LRT-Unsq-Joint-EO", None),
    "gated-sp-joint-sweep": ("ASBR-LRT-Gated-SP-Joint", "asbrlrtgatedspjoint"),
    "gated-sp-two-phase-sweep": ("ASBR-LRT-Gated-SP-TwoPhase", "asbrlrtgatedsp"),
    "gated-eo-joint-sweep": ("ASBR-LRT-Gated-EO-Joint", "asbrlrtgated"),
    "gated-eo-two-phase-sweep": ("ASBR-LRT-Gated-EO-TwoPhase", "asbrlrtgated"),
    "gated-eo-two-phase-nosw-sweep": ("ASBR-LRT-Gated-EO-TwoPhase-NoSW", "twophasenosw"),
    "gated-eo-two-phase-soft-sweep": ("ASBR-LRT-Gated-EO-TwoPhase-Soft", "twophasesoft"),
    "gated-eo-two-phase-nohinge-sweep": ("ASBR-LRT-Gated-EO-TwoPhase-NoHinge", "twophasenohinge"),
    "target-eo-two-phase-sweep": ("ASBR-LRT-Target-EO-TwoPhase", "targeteotwophase"),
    "gated-eo-progressive-sweep": ("ASBR-LRT-Gated-EO-Progressive", None),
    "soft-eo-progressive-sweep": ("ASBR-LRT-Soft-EO-Progressive", None),
    "asbr-unsq-eo-progressive-sweep": ("ASBR-LRT-Unsq-EO-Progressive", None),
    "fairret-sweep": ("fairret-norm2", None),
    "fairret1-sweep": ("fairret-norm1", None),
    "fairret-eo-sweep": ("fairret-norm2-eo", "fairretnorm2"),
    "fairret1-eo-sweep": ("fairret-norm1-eo", "fairretnorm1"),
    "gated-eo-joint-nogate-sweep": ("ASBR-LRT-Gated-EO-Joint-NoGate", "jointnogate"),
    "gated-eo-joint-lrtstat-sweep": ("ASBR-LRT-Gated-EO-Joint-LRTStat", "jointlrtstat"),
    "direct-msee-eo-joint-sweep": ("ASBR-DirectMSEE-EO-Joint", "directmsee"),
    "direct-msee-eo-joint-rms-sweep": ("ASBR-DirectMSEE-EO-Joint-RMS", "directmsee"),
    "direct-msee-eo-joint-thresholdtail-sweep": ("ASBR-DirectMSEE-EO-Joint-ThresholdTail", "thresholdtail"),
    "direct-msee-eo-joint-thresholdtail-warmup-sweep": ("ASBR-DirectMSEE-EO-Joint-ThresholdTail-Warmup", "thresholdtailwarmup"),
    "direct-msee-eo-joint-thresholdtail-warmupguard-sweep": ("ASBR-DirectMSEE-EO-Joint-ThresholdTail-WarmupGuard", "thresholdtailwarmupguard"),
    "direct-msee-eo-joint-thresholdtail-ramp-sweep": ("ASBR-DirectMSEE-EO-Joint-ThresholdTail-Ramp", "thresholdtailramp"),
    "direct-msee-eo-joint-thresholdtail-rampguard-sweep": ("ASBR-DirectMSEE-EO-Joint-ThresholdTail-RampGuard", "thresholdtailrampguard"),
    "gated-eo-joint-rms-sweep": ("ASBR-LRT-Gated-EO-Joint-RMS", "jointrms"),
    "gated-eo-joint-ovr-sweep": ("ASBR-LRT-Gated-EO-Joint-OvR", "jointovr"),
    "gated-eo-joint-ovr-rms-sweep": ("ASBR-LRT-Gated-EO-Joint-OvR-RMS", "jointovrrms"),
    "direct-lrt-eo-joint-sweep": ("ASBR-DirectLRT-EO-Joint", "directlrteojoint"),
    "direct-lrt-eo-joint-tail001-sweep": ("ASBR-DirectLRT-EO-Joint-Tail001", "directlrteojointtail001"),
    "direct-lrt-eo-joint-tail003-sweep": ("ASBR-DirectLRT-EO-Joint-Tail003", "directlrteojointtail003"),
    "direct-lrt-eo-joint-guard-sweep": ("ASBR-DirectLRT-EO-Joint-Guard", "directlrteojointguard"),
    "direct-lrt-eo-joint-guardall-sweep": ("ASBR-DirectLRT-EO-Joint-GuardAll", "directlrteojointguardall"),
    "direct-lrt-joint-sweep": ("ASBR-DirectLRT-Joint", "directlrtjoint"),
    "direct-lrt-joint-ramp30-sweep": ("ASBR-DirectLRT-Joint-Ramp30", "directlrtjointramp30"),
    "direct-lrt-twophase-sweep": ("ASBR-DirectLRT-TwoPhase", "directlrttwophase"),
    "direct-lrt-eo-joint-rampguard-sweep": ("ASBR-DirectLRT-EO-Joint-RampGuard", "directlrteojointrampguard"),
    "expgrad-sweep": ("ExponGrad", None),
    "expgrad-eo-sweep": ("ExponGrad-EO", "expongrad"),
}


def _aggregate_split_metric(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return mean and standard error over split means for one metric column."""
    if "split" in df.columns:
        split_means = df.groupby("split").mean(numeric_only=True)
        n_splits = len(split_means)
    else:
        split_means = df
        n_splits = len(df)
    vals = split_means[col]
    return float(vals.mean()), float(vals.std() / max(n_splits ** 0.5, 1))


def _load_one(csv_path: str | Path) -> dict:
    """Load one results.csv and return aggregated metrics."""
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        split_means = df.groupby("split").mean(numeric_only=True)
        n_splits = len(split_means)
    else:
        split_means = df
        n_splits = len(df)

    def _agg(col: str) -> tuple[float, float]:
        vals = split_means[col]
        return float(vals.mean()), float(vals.std() / max(n_splits ** 0.5, 1))

    stats: dict = {"n": len(df), "n_splits": n_splits}
    for col, key in [
        ("auc_test", "auc"),
        ("accuracy_test", "accuracy"),
        ("log_loss_test", "log_loss"),
        ("mean_sp_test", "mean_sp"),
        ("max_sp_test", "max_sp"),
        ("saft_mspe", "saft_mspe"),
        ("saft_violation_rate", "saft_violation_rate"),
        ("saft_mean_sp_violations", "saft_mean_sp_violations"),
    ]:
        if col in split_means.columns:
            m, se = _agg(col)
            stats[key] = m
            stats[key + "_std"] = se
        else:
            stats[key] = float("nan")
            stats[key + "_std"] = float("nan")

    for col in ("mean_eo_test", "max_eo_test"):
        key = col.replace("_test", "")
        if col in split_means.columns:
            m, se = _agg(col)
            stats[key] = m
            stats[key + "_std"] = se
        else:
            stats[key] = float("nan")
            stats[key + "_std"] = float("nan")

    for col in ("saft_msee", "saft_eo_violation_rate"):
        if col in split_means.columns:
            m, se = _agg(col)
            stats[col] = m
            stats[col + "_std"] = se
        else:
            stats[col] = float("nan")
            stats[col + "_std"] = float("nan")

    return stats


def _extract_lambd(path: Path) -> float | None:
    """Infer sweep value from a path fragment like lambd=5.0, epsilon=0.1, alpha=1, or target_reduction=0.5."""
    for part in path.parts:
        match = re.search(r"(?:lambd|epsilon|alpha|target_reduction)=([0-9.eE+\-]+)", part)
        if match:
            return float(match.group(1))
    return None


def _load_progressive(csv_path: str | Path, method_name: str) -> list[dict]:
    """Load one progressive results.csv and return one entry per lambda."""
    df = pd.read_csv(csv_path)
    if "lambd" not in df.columns:
        raise ValueError(f"No 'lambd' column in {csv_path} - not a progressive results file.")

    entries: list[dict] = []
    for lambd, group in df.groupby("lambd"):
        if "split" in group.columns:
            split_means = group.groupby("split").mean(numeric_only=True)
            n_splits = len(split_means)
        else:
            split_means = group
            n_splits = len(group)

        def _agg(col: str) -> tuple[float, float]:
            vals = split_means[col]
            return float(vals.mean()), float(vals.std() / max(n_splits ** 0.5, 1))

        stats: dict = {"n": len(group), "n_splits": n_splits}
        for col, key in [
            ("auc_test", "auc"),
            ("accuracy_test", "accuracy"),
            ("log_loss_test", "log_loss"),
            ("mean_sp_test", "mean_sp"),
            ("max_sp_test", "max_sp"),
            ("saft_mspe", "saft_mspe"),
            ("saft_violation_rate", "saft_violation_rate"),
            ("saft_mean_sp_violations", "saft_mean_sp_violations"),
        ]:
            if col in split_means.columns:
                m, se = _agg(col)
                stats[key] = m
                stats[key + "_std"] = se
            else:
                stats[key] = float("nan")
                stats[key + "_std"] = float("nan")

        for col in ("mean_eo_test", "max_eo_test"):
            key = col.replace("_test", "")
            if col in split_means.columns:
                m, se = _agg(col)
                stats[key] = m
                stats[key + "_std"] = se
            else:
                stats[key] = float("nan")
                stats[key + "_std"] = float("nan")

        for col in ("saft_msee", "saft_eo_violation_rate"):
            if col in split_means.columns:
                m, se = _agg(col)
                stats[col] = m
                stats[col + "_std"] = se
            else:
                stats[col] = float("nan")
                stats[col + "_std"] = float("nan")

        entries.append({
            "method": method_name,
            "lambd": float(lambd),
            "path": str(csv_path),
            **stats,
        })
    return entries


def _discover_sweep(sweep_dir: str, method_name: str, path_filter: str | None = None) -> list[dict]:
    """Scan a sweep directory for lambda-specific results.csv files."""
    root = Path(sweep_dir)
    entries: list[dict] = []
    needle = None if path_filter == "" else (path_filter or method_name).lower().replace("-", "").replace("_", "")
    for csv in root.rglob("results.csv"):
        lambd = _extract_lambd(csv)
        if lambd is None:
            continue
        csv_str = str(csv).lower().replace("-", "").replace("_", "")
        if needle is not None and needle not in csv_str:
            continue
        try:
            stats = _load_one(csv)
            entries.append({"method": method_name, "lambd": lambd, "path": str(csv), **stats})
        except Exception as exc:
            print(f"  [warn] Could not read {csv}: {exc}")
    return entries


def _parse_sweep_spec(spec: str) -> tuple[str, str] | None:
    """Parse a generic sweep spec of the form METHOD=DIR."""
    method, sep, sweep_dir = spec.partition("=")
    if not sep or not method.strip() or not sweep_dir.strip():
        print(f"  [warn] Skipping malformed --sweep entry {spec!r}; expected METHOD=DIR.")
        return None
    return method.strip(), sweep_dir.strip()


def _method_allowed(
    method: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> bool:
    """Return whether a method name passes include/exclude regex filters."""
    if include_patterns and not any(re.search(pattern, method) for pattern in include_patterns):
        return False
    if exclude_patterns and any(re.search(pattern, method) for pattern in exclude_patterns):
        return False
    return True


def _discover_sweep_root(
    sweep_root: str,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
) -> list[dict]:
    """Scan child directories under a dataset sweep root, using each child name as the method."""
    root = Path(sweep_root)
    if not root.exists():
        print(f"  [warn] Sweep root does not exist: {sweep_root}")
        return []
    if not root.is_dir():
        print(f"  [warn] Sweep root is not a directory: {sweep_root}")
        return []

    entries: list[dict] = []
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        method_name = method_dir.name
        if not _method_allowed(method_name, include_methods, exclude_methods):
            continue
        found = _discover_sweep(str(method_dir), method_name, path_filter="")
        if found:
            print(f"{method_name}: found {len(found)} entries in {method_dir}")
            entries.extend(found)
    return entries


def _filter_curves_by_erm_metric(
    curve_data: pd.DataFrame,
    erm_row: pd.Series | None,
    metric_col: str | None = "saft_msee",
) -> pd.DataFrame:
    """Drop points above ERM on the plotted fairness metric, unless that removes a full curve."""
    if metric_col is None or curve_data.empty or erm_row is None or metric_col not in curve_data.columns:
        return curve_data

    erm_metric = erm_row.get(metric_col)
    if erm_metric is None or pd.isna(erm_metric):
        return curve_data

    filtered_parts: list[pd.DataFrame] = []
    removed = 0
    kept_original = 0
    for method, method_df in curve_data.groupby("method", sort=False):
        keep_mask = method_df[metric_col].isna() | (method_df[metric_col] <= erm_metric)
        filtered_df = method_df[keep_mask]
        if filtered_df.empty:
            filtered_parts.append(method_df)
            kept_original += len(method_df)
        else:
            filtered_parts.append(filtered_df)
            removed += len(method_df) - len(filtered_df)

    if removed:
        print(
            f"{metric_col} filter: removed {removed} point(s) above ERM ({erm_metric:.4f}); "
            f"kept {kept_original} point(s) from curves that would become empty"
        )

    return pd.concat(filtered_parts, ignore_index=True) if filtered_parts else curve_data


def build_data(
    sweeps: Mapping[str, str | list[str]],
    extras: list[str] | None = None,
    lambd_min: Mapping[str, float] | None = None,
    lambd_max: Mapping[str, float] | None = None,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
    erm_filter_metric: str | None = "saft_msee",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Assemble plotting data from sweep directories and extra entries."""
    all_entries: list[dict] = []

    for key, dirs in sweeps.items():
        dir_list = [dirs] if isinstance(dirs, str) else list(dirs)

        if key == "progressive-csv":
            for csv_path in dir_list:
                df_check = pd.read_csv(csv_path, nrows=1)
                method_name_in_csv = (
                    df_check["method"].iloc[0] if "method" in df_check.columns else "Progressive"
                )
                found = _load_progressive(csv_path, method_name_in_csv)
                print(f"Progressive CSV ({method_name_in_csv}): {len(found)} lambda entries in {csv_path}")
                all_entries.extend(found)
            continue

        if key == "sweep-root":
            for sweep_root in dir_list:
                all_entries.extend(
                    _discover_sweep_root(
                        sweep_root,
                        include_methods=include_methods,
                        exclude_methods=exclude_methods,
                    )
                )
            continue

        if key == "sweep":
            for spec in dir_list:
                parsed = _parse_sweep_spec(spec)
                if parsed is None:
                    continue
                method_name, sweep_dir = parsed
                if not _method_allowed(method_name, include_methods, exclude_methods):
                    continue
                found = _discover_sweep(sweep_dir, method_name, path_filter="")
                print(f"{method_name}: found {len(found)} entries in {sweep_dir}")
                all_entries.extend(found)
            continue

        if key not in SWEEP_SPECS:
            valid_keys = ["sweep-root", "sweep", "progressive-csv", *SWEEP_SPECS]
            print(f"  [warn] Unknown sweep key {key!r} - skipping. Valid keys: {valid_keys}")
            continue

        method_name, path_filter = SWEEP_SPECS[key]
        if not _method_allowed(method_name, include_methods, exclude_methods):
            continue
        for sweep_dir in dir_list:
            found = _discover_sweep(sweep_dir, method_name, path_filter)
            print(f"{method_name}: found {len(found)} entries in {sweep_dir}")
            all_entries.extend(found)

    erm_row: pd.Series | None = None
    for extra in extras or []:
        parts = extra.split(",", 2)
        if len(parts) != 3:
            print(f"  [warn] Skipping malformed extra entry: {extra!r}")
            continue
        method, lambd_str, path = parts
        try:
            stats = _load_one(path)
        except Exception as exc:
            print(f"  [warn] Could not read {path}: {exc}")
            continue
        entry = {"method": method, "lambd": float(lambd_str), "path": path, "_from_extra": True, **stats}
        all_entries.append(entry)
        if method == "ERM":
            erm_row = pd.Series(entry)

    if lambd_min or lambd_max:
        before = len(all_entries)
        all_entries = [
            entry for entry in all_entries
            if entry.get("_from_extra", False) or (
                entry["method"] not in (lambd_min or {}) or entry["lambd"] >= (lambd_min or {})[entry["method"]]
            ) and (
                entry["method"] not in (lambd_max or {}) or entry["lambd"] <= (lambd_max or {})[entry["method"]]
            )
        ]
        print(f"lambda filter: kept {len(all_entries)}/{before} entries")

    empty_cols = ["method", "lambd", "path", "n", "auc", "auc_std", "log_loss", "log_loss_std"]
    data = pd.DataFrame(all_entries) if all_entries else pd.DataFrame(columns=empty_cols)
    curve_data = data[data["method"] != "ERM"].copy() if not data.empty else data
    curve_data = _filter_curves_by_erm_metric(curve_data, erm_row, erm_filter_metric)
    return curve_data, erm_row


def _annotate_lambd(ax, x, y, lambd, color, method: str = "") -> None:
    """Annotate one plot point with its lambda or target_reduction value."""
    if method in TARGET_REDUCTION_METHODS:
        prefix = "target"
    elif method in ALPHA_METHODS:
        prefix = "α"
    else:
        prefix = "λ"
    val = f"{lambd:g}" if isinstance(lambd, float) else str(lambd)
    ax.annotate(
        f"{prefix}={val}",
        (x, y),
        textcoords="offset points",
        xytext=(-10, -12),
        fontsize=6.5,
        color=color,
        alpha=0.85,
    )


def add_common_plot_args(parser: argparse.ArgumentParser, *, include_mode: bool = True) -> None:
    """Add the shared sweep/input flags used by plotting scripts."""
    if include_mode:
        parser.add_argument(
            "--mode",
            choices=["sp", "eo"],
            default="sp",
            help=(
                "Fairness axis to plot. "
                "'sp': use MSPE / SAFT SP violation rate. "
                "'eo': use MSEE / SAFT EO violation rate."
            ),
        )

    for key in SWEEP_SPECS:
        help_text = None
        if key == "eo-penalty-twophase-sweep":
            help_text = "Trainer-matched EO-Penalty (two-phase) sweep directory."
        elif key == "gated-eo-two-phase-nosw-sweep":
            help_text = "ASBR-Gated-EO-TwoPhase with use_size_weight=false sweep directory."
        parser.add_argument(f"--{key}", action="append", default=[], metavar="DIR", help=help_text)

    parser.add_argument(
        "--progressive-csv",
        action="append",
        default=[],
        metavar="CSV",
        help="Path to a progressive-training results.csv (must have a 'lambd' column).",
    )
    parser.add_argument(
        "--sweep-root",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Dataset sweep root whose child directories are method names. "
            "Each child is scanned recursively for lambda-specific results.csv files."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="METHOD=DIR",
        help=(
            "Generic method sweep directory. Example: "
            "'ASBR-New-Method=multirun/run_tag/compas/ASBR-New-Method'."
        ),
    )
    parser.add_argument(
        "--include-method",
        action="append",
        default=[],
        metavar="REGEX",
        help="Only include methods whose names match at least one regex. Applies to generic and legacy sweeps.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=[],
        metavar="REGEX",
        help="Exclude methods whose names match any regex. Applies after --include-method.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="METHOD,LAMBD,PATH",
        help="Extra point: 'ERM,0,outputs/.../results.csv'. Can be repeated.",
    )
    parser.add_argument("--lambd-min", action="append", default=[], metavar="METHOD=VALUE")
    parser.add_argument("--lambd-max", action="append", default=[], metavar="METHOD=VALUE")
    parser.add_argument("--output", default="docs/figures/pareto_adult.pdf")
    parser.add_argument("--dataset-label", default="Adult")
    parser.add_argument("--no-erm-halfline", action="store_true", default=False,
                        help="Disable vertical dashed line at half the ERM fairness metric.")
    parser.add_argument("--no-highlight-best", action="store_true", default=False,
                        help="Disable highlighting the best point below the ERM half-line.")


def sweeps_from_args(args: argparse.Namespace) -> dict[str, list[str]]:
    """Build the sweeps dict expected by build_data() from parsed CLI args."""
    sweeps: dict[str, list[str]] = {}
    if getattr(args, "sweep_root", []):
        sweeps["sweep-root"] = args.sweep_root
    if getattr(args, "sweep", []):
        sweeps["sweep"] = args.sweep
    for key in SWEEP_SPECS:
        attr = key.replace("-", "_")
        values = getattr(args, attr, [])
        if values:
            sweeps[key] = values
    if getattr(args, "progressive_csv", []):
        sweeps["progressive-csv"] = args.progressive_csv
    return sweeps


def parse_lambd_bounds(items: list[str]) -> dict[str, float]:
    """Parse repeated METHOD=VALUE CLI entries into a dict."""
    bounds: dict[str, float] = {}
    for item in items:
        method, _, value = item.partition("=")
        bounds[method.strip()] = float(value.strip())
    return bounds
