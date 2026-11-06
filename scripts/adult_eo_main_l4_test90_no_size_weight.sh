source .venv/bin/activate
source scripts/common/experiment.sh

export CUDA_VISIBLE_DEVICES=""  # force CPU to avoid OOM

EXP_DATASET=adult
EXP_INTERSECTION_LEVEL=4
N_SPLITS=100
N_SEEDS=3
TEST_SIZE=0.9
VAL_SIZE=0.1

RUN_TAG="${N_SPLITS}splits_level4_1090_nosw_final"
ERM_OUT="outputs/${EXP_DATASET}/ERM/${RUN_TAG}"
SWEEP_DIR="multirun/${RUN_TAG}"

# LAMBDA_DIRECT_LRT="0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,15,20,30,50,80,100"
LAMBDA_DIRECT_LRT="0.05,0.1,0.2,0.3,0.5,1,2,5,10,20,30,50,80,100"
LAMBDA_GATED_EO_TWOPHASE_NOSW="8,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,180,200,220,240,260,280,300,350,400"
LAMBDA_GATED_EO_TWOPHASE_NOHINGE="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10,12,14,16,18,20,25,30"
LAMBDA_EO_PENALTY_JOINT="10,20,50,100,200,500,1000,2000,4000"
# LAMBDA_EO_PENALTY_JOINT="10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,600,800,1000,2000,4000"
LAMBDA_EO_PENALTY_TWOPHASE="8,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,180,200,220,240,260,280,300,350,400"
LAMBDA_GATED_EO_JOINT="10,20,40,60,100,100,200,400,600,800,1000,2000,4000,6000,10000"
LAMBDA_FAIRRET_NORM2="0.01,0.1,0.5,1,2,5,10,20,50,100,500,1000"
LAMBDA_FAIRRET_NORM1="0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.2,1,10"
EPSILON_EXPGRAD_EO="0.005,0.01,0.02,0.05,0.1"
LAMBDA_PREJUDICE_REMOVER="0.1,0.5,1,2,5,10"
ALPHA_ADVERSARIAL_DEBIASING_EO="0.1,0.5,1,5,10"

# Examples:
#   RUN_METHODS=plot bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
#   RUN_METHODS=direct_lrt_twophase,plot bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
#   RUN_METHODS=direct_lrt_ramp30,plot bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
#   RUN_METHODS=twophase_all,eo_penalty_all,plot bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
#   FORCE_RERUN=1 RUN_METHODS=direct_lrt bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
#   RUN_METHODS=all bash scripts/adult_eo_main_l4_test90_no_size_weight.sh
RUN_METHODS="${RUN_METHODS:-plot}"
PLOT_INCLUDE_METHODS="^ASBR-DirectLRT-Joint$|^EO-Penalty-Joint$|^fairret-norm2-eo$|^fairret-norm1-eo$"
# PLOT_INCLUDE_METHODS="^ASBR-DirectLRT-Joint$|^EO-Penalty-Joint$|^fairret-norm2-eo$|^fairret-norm1-eo$|^AdvDebiasing-EO$|^ExponGrad-EO$"
# PLOT_INCLUDE_METHODS="ASBR-DirectLRT-Joint|ASBR-DirectLRT-Joint-Ramp30|ASBR-DirectLRT-TwoPhase|ASBR-LRT-Gated-EO-TwoPhase-NoSW|ASBR-LRT-Gated-EO-TwoPhase-NoHinge|EO-Penalty-TwoPhase|ASBR-LRT-Gated-EO-Joint|fairret-norm2-eo|fairret-norm1-eo|^AdvDebiasing-EO$"
AVAILABLE_STEPS="
  erm
  direct_lrt direct_lrt_ramp30 direct_lrt_twophase
  gated_eo_twophase_nosw gated_eo_twophase_nohinge
  eo_penalty_joint eo_penalty_twophase
  gated_eo_joint
  fairret_norm2 fairret_norm1
  expgrad_eo prejudice_remover adversarial_debiasing
  plot
"

experiment_init

run_erm
run_direct_lrt_joint "${LAMBDA_DIRECT_LRT}"
run_direct_lrt_joint_ramp30 "${LAMBDA_DIRECT_LRT}"
run_direct_lrt_twophase "${LAMBDA_DIRECT_LRT}"
run_gated_eo_twophase_nosw "${LAMBDA_GATED_EO_TWOPHASE_NOSW}"
run_gated_eo_twophase_nohinge "${LAMBDA_GATED_EO_TWOPHASE_NOHINGE}"
run_eo_penalty_joint "${LAMBDA_EO_PENALTY_JOINT}"
run_eo_penalty_twophase "${LAMBDA_EO_PENALTY_TWOPHASE}"
run_gated_eo_joint "${LAMBDA_GATED_EO_JOINT}"
run_fairret_norm2_eo fairret_norm2 "${LAMBDA_FAIRRET_NORM2}"
run_fairret_norm1_eo "${LAMBDA_FAIRRET_NORM1}"
run_expgrad_eo "${EPSILON_EXPGRAD_EO}"
run_prejudice_remover "${LAMBDA_PREJUDICE_REMOVER}"
run_adversarial_debiasing_eo "${ALPHA_ADVERSARIAL_DEBIASING_EO}"
plot_logloss_eo "docs/figures/adult_eo_${RUN_TAG}.pdf" \
    --lambd-max 'ASBR-LRT-Gated-EO-TwoPhase-NoSW=300' \
    --lambd-max 'ASBR-LRT-Gated-EO-TwoPhase-NoHinge=20' \
    --lambd-max 'EO-Penalty-TwoPhase=300'
