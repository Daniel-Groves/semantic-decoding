#!/bin/bash --login

# --- Slurm Options ---
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mcts_tune_S1
#SBATCH -o mcts_tune_S1.o%j

echo "========================================================"
echo "MCTS HYPERPARAMETER TUNING — S1 FULL DECODE (6 configs)"
echo "========================================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load tools/gcc/git/2.43.0
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

BASELINE_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/final_paper/baseline
RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech
SCORES_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech

# Common args
COMMON="--subject S1 --experiment perceived_speech --task wheretheressmoke"
MODELS="--encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz"

run_config() {
    local SUFFIX=$1
    local CPUCT=$2
    local GAMMA=$3
    local DEPTH=$4

    echo ""
    echo "========================================================"
    echo "Config ${SUFFIX}: c_puct=${CPUCT}, gamma=${GAMMA}, depth=${DEPTH}"
    echo "Started at $(date)"
    echo "========================================================"

    python3 -u run_mcts_decoder.py \
        ${COMMON} \
        --beam_width 10 \
        --simulations 30 \
        --depth ${DEPTH} \
        --c_puct ${CPUCT} \
        --gamma ${GAMMA} \
        --suffix ${SUFFIX} \
        ${MODELS}

    echo "=== Evaluating ${SUFFIX} ==="
    cp ${RESULTS_DIR}/wheretheressmoke_${SUFFIX}.npz ${RESULTS_DIR}/wheretheressmoke.npz

    python3 -u evaluate_predictions.py \
        --subject S1 --experiment perceived_speech \
        --task wheretheressmoke

    cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_${SUFFIX}.npz

    echo "Config ${SUFFIX} finished at $(date)"
}

# A: c_puct=1.0, gamma=0.7, depth=2
run_config tune_a 1.0 0.7 2

# B: c_puct=1.5, gamma=0.7, depth=2
run_config tune_b 1.5 0.7 2

# C: c_puct=2.5, gamma=0.4, depth=2
run_config tune_c 2.5 0.4 2

# D: c_puct=2.5, gamma=0.9, depth=2
run_config tune_d 2.5 0.9 2

# E: c_puct=1.5, gamma=0.7, depth=3
run_config tune_e 1.5 0.7 3

# F: c_puct=1.0, gamma=0.4, depth=3
run_config tune_f 1.0 0.4 3

echo ""
echo "========================================================"
echo "ALL TUNING CONFIGS COMPLETE"
echo "========================================================"
echo "Job finished on $(date)"
echo ""
echo "Score files:"
for s in tune_a tune_b tune_c tune_d tune_e tune_f; do
    echo "  ${SCORES_DIR}/wheretheressmoke_${s}.npz"
done
