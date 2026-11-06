source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM
export PYTHONWARNINGS="ignore::UserWarning"

EXP_DATASET=taiwan_credit
EXP_INTERSECTION_LEVEL=4
N_SPLITS=30
N_SEEDS=3
TEST_SIZE=0.5
VAL_SIZE=0.1

RUN_TAG="${N_SPLITS}splits_taiwan_credit_5050"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

LAMBDA_DIRECT_LRT="0.01,0.1,1,10,100"
LAMBDA_EO_PENALTY_JOINT="1,10,100,1000,10000"
LAMBDA_FAIRRET_NORM2="0.01,0.1,1,10,100"
LAMBDA_FAIRRET_NORM1="0.001,0.01,0.1,1,10"
EPSILON_EXPGRAD_EO="0.005,0.01,0.05,0.1"
LAMBDA_PREJUDICE_REMOVER="0.01,0.1,1,10"
ALPHA_ADVERSARIAL_DEBIASING_EO="0.1,0.5,1,5,10"

# Examples:
#   RUN_METHODS=plot bash scripts/taiwan_credit_eo_main_l4_test50.sh
#   RUN_METHODS=erm,direct_lrt,plot bash scripts/taiwan_credit_eo_main_l4_test50.sh
#   RUN_METHODS=all bash scripts/taiwan_credit_eo_main_l4_test50.sh

RUN_METHODS="${RUN_METHODS:-plot}"
PLOT_INCLUDE_METHODS="^ASBR-DirectLRT-Joint$|^EO-Penalty-Joint$|^fairret-norm2-eo$|^fairret-norm1-eo$|^ExponGrad-EO$|^PrejudiceRemover$|^AdvDebiasing-EO$"
AVAILABLE_STEPS="
  erm
  direct_lrt
  eo_penalty_joint
  fairret_norm2 fairret_norm1
  expgrad_eo 
  prejudice_remover
  adversarial_debiasing
  plot
"

experiment_init

run_erm
run_direct_lrt_joint "${LAMBDA_DIRECT_LRT}"
run_eo_penalty_joint "${LAMBDA_EO_PENALTY_JOINT}"
run_fairret_norm2_eo fairret_norm2 "${LAMBDA_FAIRRET_NORM2}"
run_fairret_norm1_eo "${LAMBDA_FAIRRET_NORM1}"
run_expgrad_eo "${EPSILON_EXPGRAD_EO}"
run_prejudice_remover "${LAMBDA_PREJUDICE_REMOVER}"
run_adversarial_debiasing_eo "${ALPHA_ADVERSARIAL_DEBIASING_EO}"
plot_logloss_eo "docs/figures/taiwan_credit_eo_${RUN_TAG}.pdf" \
    --dataset-label "Taiwan Credit"
    # --no-erm-halfline
    # --no-highlight-best
