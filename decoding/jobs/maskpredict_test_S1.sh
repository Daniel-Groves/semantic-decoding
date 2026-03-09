#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mp_test_S1
#SBATCH -o maskpred_test_S1.o%j

echo "MASK-PREDICT QUICK TEST — S1 (2 iterations, 50 words)"
echo "Job started on $(date)"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

BASELINE_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/final_paper/baseline
RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech

# Use MCTS mid result as starting point (same encoding model, 1589 words)
python3 -u run_maskpredict_decoder.py \
    --subject S1 \
    --experiment perceived_speech \
    --task wheretheressmoke \
    --baseline ${RESULTS_DIR}/wheretheressmoke_mcts_mid.npz \
    --n_iterations 2 \
    --mask_fraction 0.10 \
    --top_k 10 \
    --suffix maskpred_test \
    --encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz \
    --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz

echo "Job finished on $(date)"
