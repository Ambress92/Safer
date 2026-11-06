source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM

EXP_DATASET=german
EXP_INTERSECTION_LEVEL=3
N_SPLITS=100
N_SEEDS=3
TEST_SIZE=0.5
VAL_SIZE=0.1

RUN_TAG="german_l3_5050"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

LAMBDA_DIRECT_LRT="1,2,5,10,20,50,100"
LAMBDA_FAIRRET_NORM2="0.005,0.01,0.02,0.03,0.06,0.1,0.15,0.2,0.3,0.5,1,10,100"
LAMBDA_FAIRRET_NORM1="0.001,0.01,0.02,0.03,0.04,0.05,0.1,1,10"
LAMBDA_EO_PENALTY_JOINT="0.1,0.5,1,2,5,10,20,30,50,80,100,200,500,1000"

# Examples:
#   RUN_METHODS=plot bash scripts/german_eo_main_l3_test50.sh
#   RUN_METHODS=direct_lrt,fairret_norm2,fairret_norm1,plot bash scripts/german_eo_main_l3_test50.sh
#   FORCE_RERUN=1 RUN_METHODS=direct_lrt bash scripts/german_eo_main_l3_test50.sh
#   RUN_METHODS=all bash scripts/german_eo_main_l3_test50.sh
RUN_METHODS="${RUN_METHODS:-plot}"
PLOT_INCLUDE_METHODS="ASBR-DirectLRT-Joint|fairret-norm2-eo|fairret-norm1-eo|EO-Penalty$"
# PLOT_INCLUDE_METHODS="fairret-norm1-eo"
AVAILABLE_STEPS="
  erm
  direct_lrt
  fairret_norm2 fairret_norm1
  eo_penalty_joint
  plot
"

experiment_init

run_erm
run_direct_lrt_joint "${LAMBDA_DIRECT_LRT}"
run_fairret_norm2_eo fairret_norm2 "${LAMBDA_FAIRRET_NORM2}"
run_fairret_norm1_eo "${LAMBDA_FAIRRET_NORM1}"
run_eo_penalty_joint "${LAMBDA_EO_PENALTY_JOINT}"
plot_logloss_eo "docs/figures/logloss_pareto_german_eo_${RUN_TAG}.pdf"
