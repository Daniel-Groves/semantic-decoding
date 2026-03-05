#!/bin/bash --login

# --- Slurm Options ---
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=tune_eval_S1
#SBATCH -o mcts_tune_eval_S1.o%j

echo "========================================================"
echo "MCTS TUNING — EVALUATE ALL 6 PROXY CONFIGS (S1)"
echo "========================================================"
echo "Job started on $(date)"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load tools/gcc/git/2.43.0
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech
SCORES_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech

for SUFFIX in tune_a tune_b tune_c tune_d tune_e tune_f; do
    echo ""
    echo "=== Evaluating ${SUFFIX} ==="

    if [ ! -f "${RESULTS_DIR}/wheretheressmoke_${SUFFIX}.npz" ]; then
        echo "SKIP: ${RESULTS_DIR}/wheretheressmoke_${SUFFIX}.npz not found"
        continue
    fi

    cp ${RESULTS_DIR}/wheretheressmoke_${SUFFIX}.npz ${RESULTS_DIR}/wheretheressmoke.npz

    python3 -u evaluate_predictions.py \
        --subject S1 --experiment perceived_speech \
        --task wheretheressmoke

    cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_${SUFFIX}.npz

    echo "${SUFFIX} done at $(date)"
done

echo ""
echo "========================================================"
echo "ALL EVALS COMPLETE — Comparing scores"
echo "========================================================"

python3 << 'PYEOF'
import numpy as np, os

scores_dir = "/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech"
configs = ["tune_a", "tune_b", "tune_c", "tune_d", "tune_e", "tune_f"]
labels = {
    "tune_a": "c=1.0 g=0.7 d=2",
    "tune_b": "c=1.5 g=0.7 d=2",
    "tune_c": "c=2.5 g=0.4 d=2",
    "tune_d": "c=2.5 g=0.9 d=2",
    "tune_e": "c=1.5 g=0.7 d=3",
    "tune_f": "c=1.0 g=0.4 d=3",
}

header = f"{'Config':<8} {'Params':<20} {'BLEU-1':>8} {'METEOR':>8} {'BERTScore':>10}"
print(header)
print("-" * 60)

for cfg in configs:
    path = os.path.join(scores_dir, f"wheretheressmoke_{cfg}.npz")
    if not os.path.exists(path):
        print(f"{cfg:<8} {labels[cfg]:<20} {'MISSING':>8}")
        continue
    data = np.load(path, allow_pickle=True)
    ws = data["window_scores"].item()
    bleu = np.nanmean([v for k, v in ws.items() if "BLEU" in k[1]])
    meteor = np.nanmean([v for k, v in ws.items() if "METEOR" in k[1]])
    bert = np.nanmean([v for k, v in ws.items() if "BERTScore" in k[1]])
    print(f"{cfg:<8} {labels[cfg]:<20} {bleu:>8.4f} {meteor:>8.4f} {bert:>10.4f}")
PYEOF

echo ""
echo "Job finished on $(date)"
