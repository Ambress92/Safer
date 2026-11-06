"""Per-subgroup EO diagnostic: confirm or reject Hypothesis 2 for an EO loss.

Trains ERM and a chosen EO fairness method on a single Adult split, then compares
per-subgroup SAFT EO results to count how many groups:

  - gained a violation  (not-sig → sig):  pushes MSEE up
  - lost  a violation   (sig → not-sig):  reduces MSEE
  - stayed significant
  - stayed non-significant

If Hypothesis 2 holds for the chosen loss, the number of "gained" groups
should exceed "lost" at low lambda, explaining the net MSEE increase relative
to ERM.

Usage::

    python -m asbr.experiments.eo_subgroup_diagnostic \
        [--method eo_sq|safer_gated_eo|safer_unsq_eo|safer_nogate_eo|gated_eo_sq] \
        [--lambd 1.0] [--split 0] [--seed 0]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import scipy.stats
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from asbr.data.adult import AdultDataset
from asbr.data.intersections import build_subgroup_masks
from asbr.data.loader import prepare_split
from asbr.experiments.config import TrainingConfig
from asbr.fairness.saft import compute_msee, saft_eo_audit
from asbr.methods.asbr import ASBRMethod
from asbr.methods.erm import ERMMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TRAINING = TrainingConfig(
    hidden_dims=(64, 32),
    dropout=0.2,
    lr=1e-3,
    patience=10,
    max_epochs=200,
    batch_size=256,
    max_grad_norm=1.0,
    weight_decay=5e-4,
    lr_reduce_factor=0.1,
    min_lr=1e-5,
)


def _build_method(method: str, lambd: float, trainer: str) -> ASBRMethod:
    if method == "eo_sq":
        return ASBRMethod(
            name="EO-Sq-Penalty",
            trainer=trainer,
            loss_type="lrt",
            asbr={
                "alpha": 0.05,
                "lambd": lambd,
                "kappa": 0.5,
                "epsilon": 1e-8,
                "fairness_metric": "eo",
                "use_hinge": False,
                "use_hybrid": False,
                "use_size_weight": False,
                "use_sp_magnitude": True,
            },
        )
    if method == "gated_eo_sq":
        return ASBRMethod(
            name="Gated-EO-Sq",
            trainer=trainer,
            loss_type="lrt",
            asbr={
                "alpha": 0.05,
                "lambd": lambd,
                "kappa": 0.5,
                "epsilon": 1e-8,
                "fairness_metric": "eo",
                "use_hinge": True,
                "use_hybrid": False,
                "use_size_weight": False,
                "use_sp_magnitude": True,
            },
        )
    if method == "safer_gated_eo":
        return ASBRMethod(
            name="ASBR-LRT-Gated-EO-Joint" if trainer == "joint" else "ASBR-LRT-Gated-EO-TwoPhase",
            trainer=trainer,
            loss_type="lrt",
            asbr={
                "alpha": 0.05,
                "lambd": lambd,
                "n_tilde": 30,
                "kappa": 0.5,
                "epsilon": 1e-8,
                "fairness_metric": "eo",
                "use_hinge": True,
                "use_hybrid": True,
                "use_size_weight": True,
            },
        )
    if method == "safer_unsq_eo":
        return ASBRMethod(
            name="ASBR-LRT-Unsq-Joint-EO" if trainer == "joint" else "ASBR-LRT-Unsq-EO-TwoPhase",
            trainer=trainer,
            loss_type="lrt",
            asbr={
                "alpha": 0.05,
                "lambd": lambd,
                "n_tilde": 30,
                "kappa": 0.5,
                "epsilon": 1e-8,
                "fairness_metric": "eo",
                "use_hinge": True,
                "use_hybrid": True,
                "use_size_weight": True,
                "use_squared_hinge": False,
            },
        )
    if method == "safer_nogate_eo":
        return ASBRMethod(
            name="ASBR-LRT-NoGate-EO-Joint" if trainer == "joint" else "ASBR-LRT-NoGate-EO-TwoPhase",
            trainer=trainer,
            loss_type="lrt",
            asbr={
                "alpha": 0.05,
                "lambd": lambd,
                "n_tilde": 30,
                "kappa": 0.5,
                "epsilon": 1e-8,
                "fairness_metric": "eo",
                "use_hinge": True,
                "use_hybrid": True,
                "use_size_weight": True,
                "hinge_alpha": 1.0,
            },
        )
    raise ValueError(f"Unknown method {method!r}")


def _training_config_for(trainer: str) -> TrainingConfig:
    """Mirror run_hydra's auto-override: no gradient clipping for two-phase."""
    if trainer != "two_phase":
        return _DEFAULT_TRAINING
    return TrainingConfig(
        hidden_dims=_DEFAULT_TRAINING.hidden_dims,
        dropout=_DEFAULT_TRAINING.dropout,
        lr=_DEFAULT_TRAINING.lr,
        patience=_DEFAULT_TRAINING.patience,
        max_epochs=_DEFAULT_TRAINING.max_epochs,
        batch_size=_DEFAULT_TRAINING.batch_size,
        max_grad_norm=0.0,
        weight_decay=_DEFAULT_TRAINING.weight_decay,
        lr_reduce_factor=_DEFAULT_TRAINING.lr_reduce_factor,
        min_lr=_DEFAULT_TRAINING.min_lr,
    )


def _z_from_p(p_value: float) -> float:
    """Convert a two-sided p-value to a |Z| score."""
    p_clipped = max(p_value, 1e-15)
    return float(scipy.stats.norm.ppf(1.0 - p_clipped / 2.0))


def _excess(eo_estimate: float, ci_lower: float, ci_upper: float) -> float:
    """Exact per-subgroup MSEE contribution, matching compute_msee()."""
    if eo_estimate >= 0:
        return max(0.0, ci_lower)
    else:
        return max(0.0, -ci_upper)


@dataclass
class SubgroupChange:
    name: str
    eo_before: float
    eo_after: float
    ci_lower_before: float
    ci_upper_before: float
    ci_lower_after: float
    ci_upper_after: float
    z_before: float
    z_after: float
    sig_before: bool
    sig_after: bool

    @property
    def category(self) -> str:
        if self.sig_before and self.sig_after:
            return "stayed_sig"
        if not self.sig_before and not self.sig_after:
            return "stayed_nonsig"
        if not self.sig_before and self.sig_after:
            return "gained"
        return "lost"

    @property
    def excess_before(self) -> float:
        """Exact MSEE contribution (matches compute_msee formula)."""
        return _excess(self.eo_before, self.ci_lower_before, self.ci_upper_before)

    @property
    def excess_after(self) -> float:
        return _excess(self.eo_after, self.ci_lower_after, self.ci_upper_after)


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(
    method: str = "eo_sq",
    trainer: str = "joint",
    lambd: float = 1.0,
    split_idx: int = 0,
    seed: int = 0,
) -> None:
    method_obj = _build_method(method, lambd, trainer)
    method_name = method_obj.name
    training_config = _training_config_for(trainer)
    print(f"\n{'=' * 65}")
    print(
        f"EO Subgroup Diagnostic  |  {method_name}  |  λ={lambd}  |  split={split_idx}  |  seed={seed}"
    )
    print(f"{'=' * 65}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading Adult dataset ...")
    dataset = AdultDataset()
    sss = StratifiedShuffleSplit(n_splits=split_idx + 1, test_size=0.4, random_state=42)
    splits = list(sss.split(dataset.X, dataset.y))
    train_idx, test_idx = splits[split_idx]

    X_tr_full = dataset.X[train_idx]
    y_tr_full = dataset.y[train_idx]
    A_tr_full = dataset.A.iloc[train_idx].reset_index(drop=True)
    X_test    = dataset.X[test_idx]
    y_test    = dataset.y[test_idx]
    A_test    = dataset.A.iloc[test_idx].reset_index(drop=True)

    idx_tr, idx_vl = train_test_split(
        np.arange(len(X_tr_full)),
        test_size=0.2,
        stratify=y_tr_full,
        random_state=seed,
    )
    X_tr, X_vl = X_tr_full[idx_tr], X_tr_full[idx_vl]
    y_tr, y_vl = y_tr_full[idx_tr], y_tr_full[idx_vl]
    A_tr = A_tr_full.iloc[idx_tr].reset_index(drop=True)

    X_tr_s, X_vl_s, scaler = prepare_split(X_tr, X_vl)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    masks_test = build_subgroup_masks(A_test, level=3, min_size=2)
    print(f"    n_train={len(X_tr)}, n_val={len(X_vl)}, n_test={len(X_test)}, n_subgroups={len(masks_test)}")

    # ── 2. Train ERM ──────────────────────────────────────────────────────────
    print("\n[2/4] Training ERM ...")
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    erm = ERMMethod(name="ERM")
    erm.train(X_tr_s, y_tr, X_vl_s, y_vl, A_tr, intersection_level=3, training_config=training_config)
    y_prob_erm, y_pred_erm = erm.predict(X_test_s)

    eo_results_erm = saft_eo_audit(y_pred_erm, y_test, masks_test, alpha=0.05, n_tilde=30, seed=seed)
    msee_erm = compute_msee(eo_results_erm)
    logloss_erm = log_loss(y_test, y_prob_erm)
    sig_erm   = {r.subgroup for r in eo_results_erm if r.reject_h0}
    erm_by_name = {r.subgroup: r for r in eo_results_erm}
    print(
        f"    ERM log-loss={logloss_erm:.4f}, MSEE={msee_erm:.4f}, significant groups={len(sig_erm)}/{len(eo_results_erm)}"
    )

    # ── 3. Train chosen EO method at λ=lambd ─────────────────────────────────
    print(f"\n[3/4] Training {method_name} (λ={lambd}) ...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    method_obj.train(
        X_tr_s, y_tr, X_vl_s, y_vl, A_tr, intersection_level=3, training_config=training_config
    )
    y_prob_method, y_pred_method = method_obj.predict(X_test_s)

    eo_results_method = saft_eo_audit(
        y_pred_method, y_test, masks_test, alpha=0.05, n_tilde=30, seed=seed
    )
    msee_method = compute_msee(eo_results_method)
    logloss_method = log_loss(y_test, y_prob_method)
    sig_method = {r.subgroup for r in eo_results_method if r.reject_h0}
    method_by_name = {r.subgroup: r for r in eo_results_method}
    print(
        f"    {method_name} log-loss={logloss_method:.4f}, MSEE={msee_method:.4f}, significant groups={len(sig_method)}/{len(eo_results_method)}"
    )

    # ── 4. Per-subgroup change analysis ───────────────────────────────────────
    print(f"\n[4/4] Per-subgroup crossing analysis ...")
    common = set(erm_by_name.keys()) & set(method_by_name.keys())
    erm_only = set(erm_by_name.keys()) - common
    method_only = set(method_by_name.keys()) - common
    if erm_only or method_only:
        print(f"    NOTE: {len(erm_only)} groups eligible in ERM only, "
              f"{len(method_only)} in {method_name} only — excluded from accounting.")

    changes: list[SubgroupChange] = []
    for name in sorted(common):
        r_b = erm_by_name[name]
        r_a = method_by_name[name]
        changes.append(SubgroupChange(
            name=name,
            eo_before=r_b.eo_estimate,
            eo_after=r_a.eo_estimate,
            ci_lower_before=r_b.ci_lower,
            ci_upper_before=r_b.ci_upper,
            ci_lower_after=r_a.ci_lower,
            ci_upper_after=r_a.ci_upper,
            z_before=_z_from_p(r_b.p_value),
            z_after=_z_from_p(r_a.p_value),
            sig_before=r_b.reject_h0,
            sig_after=r_a.reject_h0,
        ))

    gained     = [c for c in changes if c.category == "gained"]
    lost       = [c for c in changes if c.category == "lost"]
    stayed_sig = [c for c in changes if c.category == "stayed_sig"]
    stayed_ns  = [c for c in changes if c.category == "stayed_nonsig"]

    # ── MSEE accounting (exact, over common subgroups) ────────────────────────
    n_common = len(changes)
    msee_common_before = sum(c.excess_before for c in changes) / n_common
    msee_common_after  = sum(c.excess_after  for c in changes) / n_common

    delta_gained  =  sum(c.excess_after                    for c in gained)    / n_common
    delta_lost    = -sum(c.excess_before                   for c in lost)      / n_common
    delta_sig     =  sum(c.excess_after - c.excess_before  for c in stayed_sig) / n_common
    delta_nonsig  =  sum(c.excess_after - c.excess_before  for c in stayed_ns)  / n_common
    delta_accounted = delta_gained + delta_lost + delta_sig + delta_nonsig
    # Sanity check: reconstructed delta should equal msee_common_after - msee_common_before
    accounting_error = abs(delta_accounted - (msee_common_after - msee_common_before))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  {'Category':<20} {'n':>4}   {'mean excess before':>18}   {'mean excess after':>17}   {'ΔMSEE contrib':>13}")
    print(f"{'─' * 72}")
    for label, bucket, delta in [
        ("gained (↑ MSEE)",   gained,    delta_gained),
        ("lost   (↓ MSEE)",   lost,      delta_lost),
        ("stayed_sig",         stayed_sig, delta_sig),
        ("stayed_nonsig",      stayed_ns,  delta_nonsig),
    ]:
        if bucket:
            ex_b = np.mean([c.excess_before for c in bucket])
            ex_a = np.mean([c.excess_after  for c in bucket])
        else:
            ex_b = ex_a = float("nan")
        print(f"  {label:<20} {len(bucket):>4}   {ex_b:>18.5f}   {ex_a:>17.5f}   {delta:>+13.5f}")
    print(f"{'─' * 72}")
    print(f"  {'Total common':>24}   {n_common:>4}   {msee_common_before:>18.5f}   {msee_common_after:>17.5f}   {delta_accounted:>+13.5f}")
    print(f"  Accounting error (should be ~0): {accounting_error:.2e}")
    print(f"{'─' * 72}")
    print(f"  log-loss: ERM={logloss_erm:.4f}  →  {method_name}(λ={lambd})={logloss_method:.4f}  "
          f"(Δ={logloss_method - logloss_erm:+.4f})")
    print(f"  MSEE (all eligible groups):  ERM={msee_erm:.5f}  →  {method_name}={msee_method:.5f}  "
          f"(Δ={msee_method - msee_erm:+.5f})")
    print(f"  MSEE (common groups only):   ERM={msee_common_before:.5f}  →  {method_name}={msee_common_after:.5f}  "
          f"(Δ={msee_common_after - msee_common_before:+.5f})")

    # ── Near-threshold analysis ───────────────────────────────────────────────
    z_alpha = 1.96
    delta_z = 0.5  # "border zone" width
    border_gained = [c for c in gained if (z_alpha - delta_z) <= c.z_before <= z_alpha]
    border_lost   = [c for c in lost   if  z_alpha            <= c.z_before <= (z_alpha + delta_z)]

    print(f"\n  Near-threshold breakdown (border zone ±{delta_z} around z_α=1.96):")
    print(f"    gained: {len(gained)} total, {len(border_gained)} were border (Z_erm ∈ [{z_alpha-delta_z:.2f},{z_alpha:.2f}])")
    print(f"    lost:   {len(lost)}   total, {len(border_lost)} were border (Z_erm ∈ [{z_alpha:.2f},{z_alpha+delta_z:.2f}])")

    # ── Z-score distribution for gained groups ────────────────────────────────
    if gained:
        print(f"\n  Z-score distribution for 'gained' groups (ERM Z_before):")
        z_vals = sorted([c.z_before for c in gained])
        z_buckets = [
            ("[0.0, 0.5)",   [z for z in z_vals if z < 0.5]),
            ("[0.5, 1.0)",   [z for z in z_vals if 0.5 <= z < 1.0]),
            ("[1.0, 1.5)",   [z for z in z_vals if 1.0 <= z < 1.5]),
            ("[1.5, 1.96)",  [z for z in z_vals if 1.5 <= z < z_alpha]),
        ]
        for label, bucket in z_buckets:
            bar = "█" * len(bucket)
            print(f"    Z_erm {label:<12}  n={len(bucket):>3}  {bar}")

    # ── Top gained groups (sorted by MSEE added) ──────────────────────────────
    if gained:
        print(f"\n  Top-10 'gained' groups (largest excess_after, i.e. MSEE added):")
        gained_sorted = sorted(gained, key=lambda c: c.excess_after, reverse=True)[:10]
        print(f"  {'Subgroup':<45} {'EO_erm':>7} {'Z_erm':>7} {'EO_new':>7} {'Z_new':>7} {'Δexcess':>8}")
        for c in gained_sorted:
            print(f"  {c.name:<45} {c.eo_before:>+7.3f} {c.z_before:>7.2f} "
                  f"{c.eo_after:>+7.3f} {c.z_after:>7.2f} {c.excess_after - c.excess_before:>+8.4f}")

    # ── Lost groups (sorted by MSEE removed) ─────────────────────────────────
    if lost:
        print(f"\n  'Lost' groups (violations eliminated, sorted by excess_before removed):")
        lost_sorted = sorted(lost, key=lambda c: c.excess_before, reverse=True)
        print(f"  {'Subgroup':<45} {'EO_erm':>7} {'Z_erm':>7} {'EO_new':>7} {'Z_new':>7} {'Δexcess':>8}")
        for c in lost_sorted:
            print(f"  {c.name:<45} {c.eo_before:>+7.3f} {c.z_before:>7.2f} "
                  f"{c.eo_after:>+7.3f} {c.z_after:>7.2f} {c.excess_after - c.excess_before:>+8.4f}")

    # ── Stayed_sig degradation callout ────────────────────────────────────────
    if stayed_sig:
        degraded = [c for c in stayed_sig if c.excess_after > c.excess_before]
        improved = [c for c in stayed_sig if c.excess_after < c.excess_before]
        print(f"\n  stayed_sig ({len(stayed_sig)} groups): "
              f"{len(degraded)} degraded, {len(improved)} improved, "
              f"{len(stayed_sig) - len(degraded) - len(improved)} unchanged")
        print(f"    Net MSEE change from stayed_sig: {delta_sig:+.5f}")
        if degraded:
            worst = sorted(degraded, key=lambda c: c.excess_after - c.excess_before, reverse=True)[:5]
            print(f"    Top-5 degraded (largest excess increase):")
            print(f"    {'Subgroup':<45} {'EO_erm':>7} {'Z_erm':>7} {'EO_new':>7} {'Z_new':>7} {'Δexcess':>8}")
            for c in worst:
                print(f"    {c.name:<45} {c.eo_before:>+7.3f} {c.z_before:>7.2f} "
                      f"{c.eo_after:>+7.3f} {c.z_after:>7.2f} {c.excess_after - c.excess_before:>+8.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    net_delta = msee_common_after - msee_common_before
    print(f"\n{'=' * 72}")
    print("Hypothesis 2 assessment (proxy-metric mismatch):")
    print(f"  Net Δ MSEE (common groups): {net_delta:+.5f}")
    print(f"    Boundary crossings: +{len(gained)} gained, -{len(lost)} lost  "
          f"(net count: {len(gained) - len(lost):+d})")
    print(f"    MSEE from boundary crossings:  "
          f"gained={delta_gained:+.5f}, lost={delta_lost:+.5f}  (net={delta_gained+delta_lost:+.5f})")
    print(f"    MSEE from stayed_sig changing:  {delta_sig:+.5f}")
    print(f"    MSEE from stayed_nonsig (≈0?):  {delta_nonsig:+.5f}")
    if net_delta > 1e-6:
        dominant = max(
            [("boundary crossings", delta_gained + delta_lost),
             ("stayed_sig degradation", delta_sig)],
            key=lambda x: abs(x[1]),
        )
        print(f"\n  ✓ CONFIRMED: MSEE increased by {net_delta:.5f}.")
        print(f"    Primary driver: {dominant[0]} (Δ={dominant[1]:+.5f})")
        if abs(delta_sig) > abs(delta_gained + delta_lost):
            print(f"    NOTE: stayed_sig degradation dominates — "
                  f"the penalty worsens already-certified groups more than it creates new ones.")
    elif net_delta < -1e-6:
        print(f"\n  ✗ NOT CONFIRMED: MSEE decreased by {abs(net_delta):.5f}.")
    else:
        print(f"\n  ~ NEUTRAL: no meaningful MSEE change.")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EO subgroup diagnostic for Hypothesis 2")
    parser.add_argument(
        "--method",
        type=str,
        default="eo_sq",
        choices=["eo_sq", "gated_eo_sq", "safer_gated_eo", "safer_unsq_eo", "safer_nogate_eo"],
        help="EO loss to diagnose (default: eo_sq)",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="joint",
        choices=["joint", "two_phase"],
        help="Training procedure for the chosen EO method (default: joint)",
    )
    parser.add_argument("--lambd", type=float, default=1.0, help="Lambda for the chosen EO method (default: 1.0)")
    parser.add_argument("--split", type=int, default=0, help="Split index (default: 0)")
    parser.add_argument("--seed",  type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    run_diagnostic(
        method=args.method,
        trainer=args.trainer,
        lambd=args.lambd,
        split_idx=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
