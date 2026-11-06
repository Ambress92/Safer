source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM

EXP_DATASET=compas
EXP_INTERSECTION_LEVEL=3
N_SPLITS=30
N_SEEDS=3
TEST_SIZE=0.8
VAL_SIZE=0.1

RUN_TAG="compas_l3_2080_sp"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

LAMBDA_DIRECT_LRT="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100"
LAMBDA_SP_PENALTY="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100,200,500,1000"
LAMBDA_FAIRRET_NORM2="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.5,1,2,5,10"
LAMBDA_FAIRRET_NORM1="0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.2,1,10"

# Examples:
#   RUN_METHODS=plot bash scripts/compas_sp_main_l3_test80.sh
#   RUN_METHODS=gated_sp_joint,fairret_norm2,fairret_norm1,plot bash scripts/compas_sp_main_l3_test80.sh
#   FORCE_RERUN=1 RUN_METHODS=gated_sp_joint bash scripts/compas_sp_main_l3_test80.sh
#   RUN_METHODS=all bash scripts/compas_sp_main_l3_test80.sh
RUN_METHODS="${RUN_METHODS:-plot}"
PLOT_INCLUDE_METHODS="ASBR-DirectLRT-SP-Joint|fairret-norm2-sp|fairret-norm1-sp|SP-Penalty$"
AVAILABLE_STEPS="
  erm
  direct_lrt
  fairret_norm2 fairret_norm1
  sp_penalty
  plot
"

experiment_init

run_erm
run_direct_lrt_sp_joint "${LAMBDA_DIRECT_LRT}"
run_fairret_norm2_sp fairret_norm2 "${LAMBDA_FAIRRET_NORM2}"
run_fairret_norm1_sp "${LAMBDA_FAIRRET_NORM1}"
run_sp_penalty "${LAMBDA_SP_PENALTY}"
plot_logloss_sp "docs/figures/logloss_pareto_compas_sp_${RUN_TAG}.pdf"
