# SAFER: Significance-Aware Fairness Regularization

Code for the NeurIPS 2026 paper **Overcoming the Resolution Limit:
Significance-Aware Regularization for Intersectional Fairness**.

Antonio Ferrara, André Panisson, Francesco Cozzi, Alan Perotti, Francesco Bonchi  
Intesa Sanpaolo AI Research, Turin, Italy

This repository contains the implementation and experiment scripts for
Significance-Aware Fairness Regularization (SAFER), an in-processing approach
for sparse intersectional fairness. SAFER regularizes subgroup disparities
according to their statistical evidence, so noisy small-sample disparities do
not receive the same optimization pressure as well-supported disparities.

The main reproducibility interface is the set of shell scripts in `scripts/`.
They run Hydra experiment sweeps, collect results under `outputs/` and
`multirun/`, and generate Pareto plots under `docs/figures/`.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[baselines,datasets,experiments]"
```

For development checks:

```bash
pip install -e ".[baselines,datasets,experiments,dev]"
pytest tests/
```

The Adult, COMPAS, and German loaders use `sklearn.datasets.fetch_openml`.
Those datasets are downloaded through OpenML on first use and cached by
scikit-learn.

## Experiment Scripts

All experiment entrypoints live in `scripts/`. They share
`scripts/common/experiment.sh`, which handles step selection, rerun behavior,
and plotting.

| Script | Experiment |
|---|---|
| `scripts/adult_eo_main_l4_test90_no_size_weight.sh` | Adult, Equal Opportunity, level-4 intersections, 90% test split, main Pareto comparison |
| `scripts/adult_eo_design_ablation_l4_test90.sh` | Adult, Equal Opportunity, level-4 intersections, threshold/selectivity/tail ablations |
| `scripts/compas_eo_main_l3_test80.sh` | COMPAS, Equal Opportunity, level-3 intersections, 80% test split, main Pareto comparison |
| `scripts/compas_eo_main_l3_test50.sh` | COMPAS, Equal Opportunity, level-3 intersections, 50% test split, main Pareto comparison |
| `scripts/compas_sp_main_l3_test80.sh` | COMPAS, Statistical Parity, level-3 intersections, 80% test split |
| `scripts/compas_eo_design_ablation_l3_test80.sh` | COMPAS, Equal Opportunity, level-3 intersections, 80% test split, design ablations |
| `scripts/compas_eo_design_ablation_l3_test50.sh` | COMPAS, Equal Opportunity, level-3 intersections, 50% test split, design ablations |
| `scripts/german_eo_main_l3_test50.sh` | German Credit, Equal Opportunity, level-3 intersections, 50% test split, main Pareto comparison |
| `scripts/german_eo_design_ablation_l3_test50.sh` | German Credit, Equal Opportunity, level-3 intersections, 50% test split, design ablations |
| `scripts/taiwan_credit_eo_main_l4_test50.sh` | Taiwan Credit, Equal Opportunity, level-4 intersections, 50% test split, additional main comparison |

Each script defaults to `RUN_METHODS=plot`, so it will only regenerate the plot
from existing result CSVs. To reproduce a full experiment sweep, set
`RUN_METHODS=all`.

```bash
RUN_METHODS=all bash scripts/compas_eo_main_l3_test80.sh
```

To run selected steps:

```bash
RUN_METHODS=erm,direct_lrt,fairret_norm2,fairret_norm1,eo_penalty_joint,plot \
  bash scripts/compas_eo_main_l3_test80.sh
```

To list valid step names for a script:

```bash
RUN_METHODS=list bash scripts/compas_eo_main_l3_test80.sh
```

Useful controls:

| Variable | Meaning |
|---|---|
| `RUN_METHODS` | Comma-separated step list, or `all`, or `plot` |
| `DRY_RUN=1` | Print commands without executing them |
| `FORCE_RERUN=1` | Ignore existing result CSVs and recompute selected steps |
| `VERBOSE_SKIPS=1` | Print skipped steps |

The scripts set `CUDA_VISIBLE_DEVICES=""` to force CPU execution. Remove or
override that line if you want to run the PyTorch models on a GPU.

## Outputs

Single ERM runs write to:

```text
outputs/{dataset}/ERM/{run_tag}/results.csv
```

Lambda sweeps write to:

```text
multirun/{run_tag}/{dataset}/{method}/.../results.csv
```

Plots are written under:

```text
docs/figures/
```

The tracked figures in `docs/figures/` correspond to outputs produced by these
experiment scripts.

## Running One Hydra Job Directly

The scripts are the recommended reproduction interface, but individual jobs can
also be launched with Hydra:

```bash
python -m asbr.experiments.run_hydra \
  method=erm dataset=compas dataset.intersection_level=3 \
  n_splits=2 n_seeds=1 test_size=0.8 val_size=0.1
```

Example lambda sweep:

```bash
python -m asbr.experiments.run_hydra --multirun \
  method=asbr_direct_lrt_joint dataset=compas dataset.intersection_level=3 \
  method.asbr.lambd=0.05,0.1,0.2,0.5,1 \
  n_splits=100 n_seeds=3 test_size=0.8 val_size=0.1
```

## Repository Map

```text
conf/                  Hydra configs for datasets, methods, and training
src/asbr/data/         Dataset loaders and intersection construction
src/asbr/fairness/     SAFER losses and fairness/statistical primitives
src/asbr/methods/      Method wrappers for SAFER and baselines
src/asbr/experiments/  Hydra runner, aggregation, and plotting code
scripts/               Reproducible experiment entrypoints
docs/figures/          Generated experiment figures
tests/                 Unit tests
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{ferrara2026safer,
  title     = {Overcoming the Resolution Limit: Significance-Aware Regularization for Intersectional Fairness},
  author    = {Ferrara, Antonio and Panisson, Andr{\'e} and Cozzi, Francesco and Perotti, Alan and Bonchi, Francesco},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
