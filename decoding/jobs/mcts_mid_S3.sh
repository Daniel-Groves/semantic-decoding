#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-6
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --job-name=mcts_mid_S3
#SBATCH -o mcts_mid_S3.o%j

echo "MCTS mid (10/30/2) — S3 wheretheressmoke"
echo "Job started on $(date)"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

BASELINE_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/final_paper/baseline

python3 -u run_mcts_decoder.py \
    --subject S3 \
    --experiment perceived_speech \
    --task wheretheressmoke \
    --beam_width 10 --simulations 30 --depth 2 \
    --c_puct 2.5 --gamma 0.7 \
    --suffix mcts_mid \
    --encoding_model ${BASELINE_DIR}/S3_baseline_encoding_model.npz \
    --word_rate_model ${BASELINE_DIR}/S3_baseline_word_rate_auditory.npz

RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S3/perceived_speech
SCORES_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S3/perceived_speech

cp ${RESULTS_DIR}/wheretheressmoke_mcts_mid.npz ${RESULTS_DIR}/wheretheressmoke.npz

python3 -u evaluate_predictions.py \
    --subject S3 --experiment perceived_speech \
    --task wheretheressmoke

cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_mcts_mid.npz

echo "Job finished on $(date)"
