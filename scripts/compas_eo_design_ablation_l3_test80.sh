source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM

# Ablations of three design choices in ASBR-DirectLRT-Joint:
#   (a) Threshold-smoothed predictions: direct_pred_temperature=0.05 (smooth
#       approximation of the hard 0.5 threshold) vs. null (raw calibrated
#       probabilities feed the fairness counts).
#   (b) Softplus selectivity: hinge_alpha=0.05 → z_hinge≈1.96 (only groups
#       with significant excess contribute) vs. hinge_alpha=1.0 → z_hinge=0
#       (excess = abs_metric, softplus stays in its linear regime, all groups
#       always active).
#   (c) Dense tail: dense_tail_weight=0.003 (small √Λ-shaped term over all
#       groups) vs. 0.0 (significant-excess hinge only).

EXP_DATASET=compas
EXP_INTERSECTION_LEVEL=3
N_SPLITS=100
N_SEEDS=3
TEST_SIZE=0.8
VAL_SIZE=0.1

RUN_TAG="compas_l3_2080_ablation"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

LAMBDA_DIRECT_LRT="0.05,0.1,0.2,0.5,1,2,5,10,20,100"
LAMBDA_DIRECT_LRT_NOSEL="0.05,0.1,0.2,0.3,0.4,0.5,1,2,5,10,100"

# Examples:
#   RUN_METHODS=plot bash scripts/compas_eo_design_ablation_l3_test80.sh
#   RUN_METHODS=direct_lrt,direct_lrt_no_thresh,plot bash scripts/compas_eo_design_ablation_l3_test80.sh
#   FORCE_RERUN=1 RUN_METHODS=direct_lrt_no_thresh bash scripts/compas_eo_design_ablation_l3_test80.sh
#   RUN_METHODS=all bash scripts/compas_eo_design_ablation_l3_test80.sh
RUN_METHODS="${RUN_METHODS:-plot}"
PLOT_INCLUDE_METHODS="ASBR-DirectLRT-Joint$|ASBR-DirectLRT-Joint-NoThreshSmooth|ASBR-DirectLRT-Joint-NoSelectivity|ASBR-DirectLRT-Joint-NoDenseTail"
AVAILABLE_STEPS="
  erm
  direct_lrt
  direct_lrt_no_thresh
  direct_lrt_no_selectivity
  direct_lrt_no_dense_tail
  plot
"

experiment_init

run_erm
run_direct_lrt_joint "${LAMBDA_DIRECT_LRT}"
run_direct_lrt_joint_no_thresh_smooth "${LAMBDA_DIRECT_LRT}"
run_direct_lrt_joint_no_selectivity "${LAMBDA_DIRECT_LRT_NOSEL}"
run_direct_lrt_joint_no_dense_tail "${LAMBDA_DIRECT_LRT}"
plot_logloss_eo "docs/figures/logloss_pareto_compas_eo_${RUN_TAG}.pdf"
