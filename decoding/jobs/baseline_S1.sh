#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-6
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=baseline_S1
#SBATCH -o baseline_S1.o%j

echo "Re-running beam search baseline for S1"
echo "Job started on $(date)"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

BASELINE_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/final_paper/baseline
MODELS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/S1
RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech

# Temporarily swap encoding model to GPT-1 baseline
mv ${MODELS_DIR}/encoding_model_perceived.npz ${MODELS_DIR}/encoding_model_perceived.npz.bak
mv ${MODELS_DIR}/word_rate_model_auditory.npz ${MODELS_DIR}/word_rate_model_auditory.npz.bak
ln -s ${BASELINE_DIR}/S1_baseline_encoding_model.npz ${MODELS_DIR}/encoding_model_perceived.npz
ln -s ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz ${MODELS_DIR}/word_rate_model_auditory.npz

python3 -u run_decoder.py \
    --subject S1 \
    --experiment perceived_speech \
    --task wheretheressmoke

# Restore original models
rm ${MODELS_DIR}/encoding_model_perceived.npz ${MODELS_DIR}/word_rate_model_auditory.npz
mv ${MODELS_DIR}/encoding_model_perceived.npz.bak ${MODELS_DIR}/encoding_model_perceived.npz
mv ${MODELS_DIR}/word_rate_model_auditory.npz.bak ${MODELS_DIR}/word_rate_model_auditory.npz

# Save a permanent copy
cp ${RESULTS_DIR}/wheretheressmoke.npz ${RESULTS_DIR}/wheretheressmoke_baseline.npz

echo "Job finished on $(date)"
