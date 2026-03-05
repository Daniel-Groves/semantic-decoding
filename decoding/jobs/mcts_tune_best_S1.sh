#!/bin/bash --login

# --- Slurm Options ---
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-6
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mcts_best_S1
#SBATCH -o mcts_best_S1.o%j

# ============================================================
# PHASE 2: Full S1 decode for top 2 configs from Phase 1
# ============================================================
# INSTRUCTIONS: After Phase 1 completes, fill in the winning
# hyperparams below based on BERTScore results, then submit.
# ============================================================

echo "========================================================"
echo "MCTS TUNING PHASE 2 — FULL S1 DECODE (top 2 configs)"
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

MODELS="--encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz"

# ============================================================
# CONFIG 1: Fill in from Phase 1 winner
# ============================================================
SUFFIX_1="tune_best1"
CPUCT_1=1.5    # <-- UPDATE from Phase 1
GAMMA_1=0.7    # <-- UPDATE from Phase 1
DEPTH_1=2      # <-- UPDATE from Phase 1

echo ""
echo "=== Config 1: c_puct=${CPUCT_1}, gamma=${GAMMA_1}, depth=${DEPTH_1} ==="
echo "Started at $(date)"

python3 -u run_mcts_decoder.py \
    --subject S1 --experiment perceived_speech --task wheretheressmoke \
    --beam_width 10 --simulations 30 \
    --depth ${DEPTH_1} --c_puct ${CPUCT_1} --gamma ${GAMMA_1} \
    --suffix ${SUFFIX_1} \
    ${MODELS}

echo "=== Evaluating ${SUFFIX_1} ==="
cp ${RESULTS_DIR}/wheretheressmoke_${SUFFIX_1}.npz ${RESULTS_DIR}/wheretheressmoke.npz
python3 -u evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_${SUFFIX_1}.npz

echo "Config 1 finished at $(date)"

# ============================================================
# CONFIG 2: Fill in from Phase 1 runner-up
# ============================================================
SUFFIX_2="tune_best2"
CPUCT_2=1.0    # <-- UPDATE from Phase 1
GAMMA_2=0.4    # <-- UPDATE from Phase 1
DEPTH_2=3      # <-- UPDATE from Phase 1

echo ""
echo "=== Config 2: c_puct=${CPUCT_2}, gamma=${GAMMA_2}, depth=${DEPTH_2} ==="
echo "Started at $(date)"

python3 -u run_mcts_decoder.py \
    --subject S1 --experiment perceived_speech --task wheretheressmoke \
    --beam_width 10 --simulations 30 \
    --depth ${DEPTH_2} --c_puct ${CPUCT_2} --gamma ${GAMMA_2} \
    --suffix ${SUFFIX_2} \
    ${MODELS}

echo "=== Evaluating ${SUFFIX_2} ==="
cp ${RESULTS_DIR}/wheretheressmoke_${SUFFIX_2}.npz ${RESULTS_DIR}/wheretheressmoke.npz
python3 -u evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_${SUFFIX_2}.npz

echo "Config 2 finished at $(date)"

echo ""
echo "========================================================"
echo "PHASE 2 COMPLETE"
echo "========================================================"
echo "Score files:"
echo "  ${SCORES_DIR}/wheretheressmoke_${SUFFIX_1}.npz"
echo "  ${SCORES_DIR}/wheretheressmoke_${SUFFIX_2}.npz"
echo "Job finished on $(date)"
