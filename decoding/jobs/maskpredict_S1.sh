#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-3
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=maskpred_S1
#SBATCH -o maskpred_S1.o%j

echo "========================================================"
echo "MASK-PREDICT DECODER — S1"
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

# Baseline result to refine
BASELINE_RESULT=${RESULTS_DIR}/wheretheressmoke_baseline.npz

echo ""
echo "=== Running mask-predict decoder (BERT + brain) ==="
python3 -u run_maskpredict_decoder.py \
    --subject S1 \
    --experiment perceived_speech \
    --task wheretheressmoke \
    --baseline ${BASELINE_RESULT} \
    --n_iterations 10 \
    --mask_fraction 0.15 \
    --alpha 1.0 \
    --beta 1.0 \
    --top_k 30 \
    --suffix maskpred \
    --encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz \
    --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz

echo ""
echo "=== Evaluating ==="
cp ${RESULTS_DIR}/wheretheressmoke_maskpred.npz ${RESULTS_DIR}/wheretheressmoke.npz

python3 -u evaluate_predictions.py \
    --subject S1 --experiment perceived_speech \
    --task wheretheressmoke

cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_maskpred.npz

echo ""
echo "=== Running BERT-only ablation (no brain) ==="
python3 -u run_maskpredict_decoder.py \
    --subject S1 \
    --experiment perceived_speech \
    --task wheretheressmoke \
    --baseline ${BASELINE_RESULT} \
    --n_iterations 10 \
    --mask_fraction 0.15 \
    --alpha 1.0 \
    --top_k 30 \
    --no_brain \
    --suffix maskpred_nobrain \
    --encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz \
    --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz

echo ""
echo "=== Evaluating BERT-only ==="
cp ${RESULTS_DIR}/wheretheressmoke_maskpred_nobrain.npz ${RESULTS_DIR}/wheretheressmoke.npz

python3 -u evaluate_predictions.py \
    --subject S1 --experiment perceived_speech \
    --task wheretheressmoke

cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_maskpred_nobrain.npz

echo ""
echo "========================================================"
echo "MASK-PREDICT COMPLETE"
echo "========================================================"
echo "Job finished on $(date)"
