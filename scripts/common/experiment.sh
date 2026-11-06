# Shared helpers for Hydra experiment scripts.
#
# Scripts using this library should define:
#   EXP_DATASET, EXP_INTERSECTION_LEVEL, N_SPLITS, N_SEEDS, TEST_SIZE, VAL_SIZE,
#   RUN_TAG, ERM_OUT, SWEEP_DIR, AVAILABLE_STEPS
# before calling experiment_init.

EXCLUDE_COMMON="dataset,dataset.intersection_level,method,method.name,n_splits,n_seeds,test_size,val_size,force_rerun"
EXCLUDE_DIRECT_LRT="${EXCLUDE_COMMON},training"
EXCLUDE_DIRECT_LRT_RAMP="${EXCLUDE_DIRECT_LRT},training.fairness_ramp_epochs"
EXCLUDE_DIRECT_LRT_SP="${EXCLUDE_DIRECT_LRT},method.asbr.fairness_metric"
EXCLUDE_DIRECT_LRT_NOTHRESH="${EXCLUDE_DIRECT_LRT},method.asbr.direct_pred_temperature"
EXCLUDE_DIRECT_LRT_NOSEL="${EXCLUDE_DIRECT_LRT},method.asbr.hinge_alpha"
EXCLUDE_DIRECT_LRT_NOSEL_NOSOFTPLUS="${EXCLUDE_DIRECT_LRT},method.asbr.hinge_alpha,method.asbr.direct_excess_activation"
EXCLUDE_DIRECT_LRT_NOTAIL="${EXCLUDE_DIRECT_LRT},method.asbr.dense_tail_weight"
EXCLUDE_FAIRRET="${EXCLUDE_COMMON},method.fairness_metric,training"
EXCLUDE_NOSW="${EXCLUDE_COMMON},method.asbr.use_size_weight"
EXCLUDE_NOHINGE="${EXCLUDE_COMMON},method.asbr.use_size_weight,method.asbr.hinge_alpha"
EXCLUDE_EOPENALTY="${EXCLUDE_COMMON},training"
EXCLUDE_GATED_EO_JOINT="${EXCLUDE_COMMON},method.asbr.fairness_metric"
EXCLUDE_EXPGRAD="${EXCLUDE_COMMON}"
EXCLUDE_PREJUDICE_REMOVER="${EXCLUDE_COMMON}"
EXCLUDE_ADVERSARIAL="${EXCLUDE_COMMON},method.constraints"

experiment_init() {
    RUN_METHODS="${RUN_METHODS:-plot}"
    DRY_RUN="${DRY_RUN:-0}"
    FORCE_RERUN="${FORCE_RERUN:-false}"
    VERBOSE_SKIPS="${VERBOSE_SKIPS:-0}"

    case "${FORCE_RERUN}" in
        1|true|TRUE|yes|YES)
            FORCE_RERUN=true
            ;;
        *)
            FORCE_RERUN=false
            ;;
    esac

    if [[ "${RUN_METHODS}" == "list" ]]; then
        echo "Available RUN_METHODS step IDs:"
        echo "${AVAILABLE_STEPS}"
        echo "Group aliases: all, fairret_all, twophase_all, eo_penalty_all"
        exit 0
    fi

    build_plot_filter_args
}

step_enabled() {
    local step="$1"
    local selected="${RUN_METHODS//,/ }"
    local item

    for item in ${selected}; do
        case "${item}" in
            all)
                return 0
                ;;
            "${step}")
                return 0
                ;;
            fairret_all)
                [[ "${step}" == fairret* ]] && return 0
                ;;
            twophase_all)
                [[ "${step}" == gated_eo_twophase* ]] && return 0
                ;;
            eo_penalty_all)
                [[ "${step}" == eo_penalty* ]] && return 0
                ;;
        esac
    done

    return 1
}

run_step() {
    local step="$1"
    shift

    if ! step_enabled "${step}"; then
        if [[ "${VERBOSE_SKIPS}" == "1" ]]; then
            echo "==> Skipping ${step}"
        fi
        return 0
    fi

    echo "==> Running ${step}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '    '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi

    "$@"
}

build_plot_filter_args() {
    PLOT_FILTER_ARGS=()
    local pattern
    local include_patterns="${PLOT_INCLUDE_METHODS//,/ }"
    local exclude_patterns="${PLOT_EXCLUDE_METHODS//,/ }"

    for pattern in ${include_patterns}; do
        [[ -n "${pattern}" ]] && PLOT_FILTER_ARGS+=(--include-method "${pattern}")
    done

    for pattern in ${exclude_patterns}; do
        [[ -n "${pattern}" ]] && PLOT_FILTER_ARGS+=(--exclude-method "${pattern}")
    done
}

run_erm() {
    run_step erm python -m asbr.experiments.run_hydra \
        method=erm \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.run.dir=${ERM_OUT}
}

run_direct_lrt_joint() {
    local lambdas="$1"
    run_step direct_lrt python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT}]"
}

run_direct_lrt_sp_joint() {
    local lambdas="$1"
    run_step direct_lrt python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-SP-Joint \
        method.asbr.fairness_metric=sp \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_SP}]"
}

run_direct_lrt_joint_no_thresh_smooth() {
    local lambdas="$1"
    run_step direct_lrt_no_thresh python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-Joint-NoThreshSmooth \
        method.asbr.direct_pred_temperature=null \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_NOTHRESH}]"
}

run_direct_lrt_joint_no_selectivity() {
    local lambdas="$1"
    run_step direct_lrt_no_selectivity python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-Joint-NoSelectivity \
        method.asbr.hinge_alpha=1.0 \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_NOSEL}]"
}

run_direct_lrt_joint_no_selectivity_no_softplus() {
    local lambdas="$1"
    run_step direct_lrt_no_selectivity_no_softplus python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-Joint-NoSelectivity-NoSoftplus \
        method.asbr.hinge_alpha=1.0 \
        method.asbr.direct_excess_activation=linear \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_NOSEL_NOSOFTPLUS}]"
}

run_direct_lrt_joint_no_dense_tail() {
    local lambdas="$1"
    run_step direct_lrt_no_dense_tail python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-Joint-NoDenseTail \
        method.asbr.dense_tail_weight=0.0 \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_NOTAIL}]"
}

run_direct_lrt_joint_ramp30() {
    local lambdas="$1"
    run_step direct_lrt_ramp30 python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_joint \
        method.name=ASBR-DirectLRT-Joint-Ramp30 \
        training=joint_bce_guarded \
        training.fairness_ramp_epochs=30 \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT_RAMP}]"
}

run_direct_lrt_twophase() {
    local lambdas="$1"
    run_step direct_lrt_twophase python -m asbr.experiments.run_hydra --multirun \
        method=asbr_direct_lrt_twophase \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_DIRECT_LRT}]"
}

run_fairret_norm2_eo() {
    local step="$1"
    local lambdas="$2"
    run_step "${step}" python -m asbr.experiments.run_hydra --multirun \
        method=fairret_norm2_joint \
        method.name=fairret-norm2-eo \
        method.fairness_metric=eo \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_FAIRRET}]"
}

run_fairret_norm1_eo() {
    local lambdas="$1"
    run_step fairret_norm1 python -m asbr.experiments.run_hydra --multirun \
        method=fairret_norm1_joint \
        method.name=fairret-norm1-eo \
        method.fairness_metric=eo \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_FAIRRET}]"
}

run_gated_eo_twophase_nosw() {
    local lambdas="$1"
    run_step gated_eo_twophase_nosw python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_eo_twophase \
        method.name=ASBR-LRT-Gated-EO-TwoPhase-NoSW \
        method.asbr.use_size_weight=false \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_NOSW}]"
}

run_gated_eo_twophase_nohinge() {
    local lambdas="$1"
    run_step gated_eo_twophase_nohinge python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_eo_twophase \
        method.name=ASBR-LRT-Gated-EO-TwoPhase-NoHinge \
        method.asbr.hinge_alpha=1.0 \
        method.asbr.use_size_weight=false \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_NOHINGE}]"
}

run_gated_eo_joint() {
    local lambdas="$1"
    run_step gated_eo_joint python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_sp_joint \
        method.name=ASBR-LRT-Gated-EO-Joint \
        method.asbr.fairness_metric=eo \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_GATED_EO_JOINT}]"
}

run_eo_penalty_joint() {
    local lambdas="$1"
    run_step eo_penalty_joint python -m asbr.experiments.run_hydra --multirun \
        method=eo_penalty_joint \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_EOPENALTY}]"
}

run_eo_penalty_twophase() {
    local lambdas="$1"
    run_step eo_penalty_twophase python -m asbr.experiments.run_hydra --multirun \
        method=eo_penalty_twophase \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_EOPENALTY}]"
}

run_expgrad_eo() {
    local epsilons="$1"
    run_step expgrad_eo python -m asbr.experiments.run_hydra --multirun \
        method=expgrad_eo \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.constraint=tpr \
        method.epsilon=${epsilons} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_EXPGRAD}]"
}

run_prejudice_remover() {
    local lambdas="$1"
    run_step prejudice_remover python -m asbr.experiments.run_hydra --multirun \
        method=prejudice_remover \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_PREJUDICE_REMOVER}]"
}

run_adversarial_debiasing_eo() {
    local alphas="$1"
    run_step adversarial_debiasing python -m asbr.experiments.run_hydra --multirun \
        method=adversarial \
        method.name=AdvDebiasing-EO \
        method.constraints=equalized_odds \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.alpha=${alphas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_ADVERSARIAL}]"
}

plot_logloss_eo() {
    local output="$1"
    shift
    run_step plot python -m asbr.experiments.logloss_plot \
        --mode eo \
        --sweep-root ${SWEEP_DIR}/${EXP_DATASET} \
        "${PLOT_FILTER_ARGS[@]}" \
        "$@" \
        --extra "ERM,0,${ERM_OUT}/results.csv" \
        --output "${output}"
}

# ---------------------------------------------------------------------------
# SP variants
# ---------------------------------------------------------------------------

run_gated_sp_joint() {
    local lambdas="$1"
    run_step gated_sp_joint python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_sp_joint \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_COMMON}]"
}

run_gated_sp_twophase_nosw() {
    local lambdas="$1"
    run_step gated_sp_twophase_nosw python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_sp \
        method.name=ASBR-LRT-Gated-SP-TwoPhase-NoSW \
        method.asbr.use_size_weight=false \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_NOSW}]"
}

run_gated_sp_twophase_nohinge() {
    local lambdas="$1"
    run_step gated_sp_twophase_nohinge python -m asbr.experiments.run_hydra --multirun \
        method=asbr_lrt_gated_sp \
        method.name=ASBR-LRT-Gated-SP-TwoPhase-NoHinge \
        method.asbr.hinge_alpha=1.0 \
        method.asbr.use_size_weight=false \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.asbr.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_NOHINGE}]"
}

run_fairret_norm2_sp() {
    local step="$1"
    local lambdas="$2"
    run_step "${step}" python -m asbr.experiments.run_hydra --multirun \
        method=fairret_norm2_joint \
        method.name=fairret-norm2-sp \
        method.fairness_metric=sp \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_FAIRRET}]"
}

run_fairret_norm1_sp() {
    local lambdas="$1"
    run_step fairret_norm1 python -m asbr.experiments.run_hydra --multirun \
        method=fairret_norm1_joint \
        method.name=fairret-norm1-sp \
        method.fairness_metric=sp \
        training=joint_bce_guarded \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_FAIRRET}]"
}

run_sp_penalty() {
    local lambdas="$1"
    run_step sp_penalty python -m asbr.experiments.run_hydra --multirun \
        method=sp_penalty \
        dataset=${EXP_DATASET} dataset.intersection_level=${EXP_INTERSECTION_LEVEL} \
        method.lambd=${lambdas} \
        n_splits=${N_SPLITS} n_seeds=${N_SEEDS} \
        force_rerun=${FORCE_RERUN} \
        test_size=${TEST_SIZE} val_size=${VAL_SIZE} \
        hydra.sweep.dir=${SWEEP_DIR} \
        'hydra.sweep.subdir=${dataset.name}/${method.name}/${hydra.job.override_dirname}' \
        "hydra.job.config.override_dirname.exclude_keys=[${EXCLUDE_COMMON}]"
}

plot_logloss_sp() {
    local output="$1"
    shift
    run_step plot python -m asbr.experiments.logloss_plot \
        --mode sp \
        --sweep-root ${SWEEP_DIR}/${EXP_DATASET} \
        "${PLOT_FILTER_ARGS[@]}" \
        "$@" \
        --extra "ERM,0,${ERM_OUT}/results.csv" \
        --output "${output}"
}
