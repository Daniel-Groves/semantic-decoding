#!/bin/bash --login

# --- Slurm Options ---
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-6
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mcts_mid_S1
#SBATCH -o mcts_mid_S1.o%j

echo "========================================================"
echo "MCTS DECODER — MID CONFIG (width=10, sims=30, depth=2)"
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

echo ""
echo "=== Running MCTS decoder ==="
python3 -u run_mcts_decoder.py \
    --subject S1 \
    --experiment perceived_speech \
    --task wheretheressmoke \
    --beam_width 10 \
    --simulations 30 \
    --depth 2 \
    --c_puct 2.5 \
    --gamma 0.7 \
    --suffix mcts_mid \
    --encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz \
    --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz

RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech
SCORES_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech

echo ""
echo "=== Evaluating results ==="
cp ${RESULTS_DIR}/wheretheressmoke_mcts_mid.npz ${RESULTS_DIR}/wheretheressmoke.npz

python3 -u evaluate_predictions.py \
    --subject S1 --experiment perceived_speech \
    --task wheretheressmoke

cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_mcts_mid.npz

echo ""
echo "========================================================"
echo "MCTS MID CONFIG COMPLETE"
echo "========================================================"
echo "Job finished on $(date)"
