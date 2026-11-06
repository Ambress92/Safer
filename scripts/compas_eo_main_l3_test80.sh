source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM

EXP_DATASET=compas
EXP_INTERSECTION_LEVEL=3
N_SPLITS=100
N_SEEDS=3
TEST_SIZE=0.8
VAL_SIZE=0.1

# Reuses completed results from the exploratory 20/80 COMPAS run.
RUN_TAG="compas_${N_SPLITS}splits_level4_2080_nosw_final"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

LAMBDA_DIRECT_LRT="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100,200,500,1000"
LAMBDA_DIRECT_LRT_NOSEL_NOSOFTPLUS="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100"
LAMBDA_FAIRRET_NORM2="0.005,0.01,0.02,0.03,0.05,0.06,0.08,0.1,0.2,0.3,0.4,0.5,1,2,5,10,20,50,100,500,1000"
LAMBDA_FAIRRET_NORM1="0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.2,1,10"
LAMBDA_EO_PENALTY_JOINT="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100,200,500,1000"

# Examples:
#   RUN_METHODS=plot bash scripts/compas_eo_main_l3_test80.sh
#   RUN_METHODS=direct_lrt,direct_lrt_no_selectivity_no_softplus,plot bash scripts/compas_eo_main_l3_test80.sh
#   FORCE_RERUN=1 RUN_METHODS=direct_lrt bash scripts/compas_eo_main_l3_test80.sh
#   RUN_METHODS=all bash scripts/compas_eo_main_l3_test80.sh
RUN_METHODS="${RUN_METHODS:-plot}"
# PLOT_INCLUDE_METHODS="ASBR-DirectLRT-Joint$|ASBR-DirectLRT-Joint-NoSelectivity-NoSoftplus|fairret-norm2-eo|fairret-norm1-eo|EO-Penalty$"
PLOT_INCLUDE_METHODS="ASBR-DirectLRT-Joint$|fairret-norm2-eo|fairret-norm1-eo|EO-Penalty$"
AVAILABLE_STEPS="
  erm
  direct_lrt direct_lrt_no_selectivity_no_softplus
  fairret_norm2 fairret_norm1
  eo_penalty_joint
  plot
"

experiment_init

run_erm
run_direct_lrt_joint "${LAMBDA_DIRECT_LRT}"
run_direct_lrt_joint_no_selectivity_no_softplus "${LAMBDA_DIRECT_LRT_NOSEL_NOSOFTPLUS}"
run_fairret_norm2_eo fairret_norm2 "${LAMBDA_FAIRRET_NORM2}"
run_fairret_norm1_eo "${LAMBDA_FAIRRET_NORM1}"
run_eo_penalty_joint "${LAMBDA_EO_PENALTY_JOINT}"
plot_logloss_eo "docs/figures/logloss_pareto_compas_eo_${RUN_TAG}.pdf"
