#!/bin/bash --login

# --- Slurm Options ---
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mcts_tune2_S1
#SBATCH -o mcts_tune2_S1.o%j

echo "========================================================"
echo "MCTS TUNING PHASE 2 — S1 FULL DECODE (3 configs)"
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

COMMON="--subject S1 --experiment perceived_speech --task wheretheressmoke"
MODELS="--encoding_model ${BASELINE_DIR}/S1_baseline_encoding_model.npz --word_rate_model ${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz"

run_config() {
    local SUFFIX=$1
    local CPUCT=$2
    local GAMMA=$3
    local DEPTH=$4
    local SIMS=$5

    echo ""
    echo "========================================================"
    echo "Config ${SUFFIX}: c_puct=${CPUCT}, gamma=${GAMMA}, depth=${DEPTH}, sims=${SIMS}"
    echo "Started at $(date)"
    echo "========================================================"

    python3 -u run_mcts_decoder.py \
        ${COMMON} \
        --beam_width 10 \
        --simulations ${SIMS} \
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

# G: c_puct=1.5, gamma=0.4, depth=2, sims=30
run_config tune_g 1.5 0.4 2 30

# H: c_puct=1.5, gamma=0.5, depth=2, sims=30
run_config tune_h 1.5 0.5 2 30

# I: c_puct=1.5, gamma=0.4, depth=2, sims=50
run_config tune_i 1.5 0.4 2 50

echo ""
echo "========================================================"
echo "ALL TUNING PHASE 2 CONFIGS COMPLETE"
echo "========================================================"
echo "Job finished on $(date)"

python3 << 'PYEOF'
import numpy as np, os

scores_dir = "/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech"
configs = ["tune_g", "tune_h", "tune_i"]
labels = {
    "tune_g": "c=1.5 g=0.4 d=2 s=30",
    "tune_h": "c=1.5 g=0.5 d=2 s=30",
    "tune_i": "c=1.5 g=0.4 d=2 s=50",
}

header = f"{'Config':<8} {'Params':<24} {'BLEU-1':>8} {'METEOR':>8} {'BERTScore':>10}"
print(header)
print("-" * 64)

for cfg in configs:
    path = os.path.join(scores_dir, f"wheretheressmoke_{cfg}.npz")
    if not os.path.exists(path):
        print(f"{cfg:<8} {labels[cfg]:<24} {'MISSING':>8}")
        continue
    data = np.load(path, allow_pickle=True)
    ws = data["window_scores"].item()
    bleu = np.nanmean([v for k, v in ws.items() if k[1] == "BLEU"])
    meteor = np.nanmean([v for k, v in ws.items() if k[1] == "METEOR"])
    bert = np.nanmean([v for k, v in ws.items() if k[1] == "BERT"])
    print(f"{cfg:<8} {labels[cfg]:<24} {bleu:>8.4f} {meteor:>8.4f} {bert:>10.4f}")

print()
print("Baselines:")
print(f"{'base':<8} {'beam search':<24} {'0.2337':>8} {'0.1721':>8} {'0.8074':>10}")
print(f"{'mid':<8} {'c=2.5 g=0.7 d=2 s=30':<24} {'0.2269':>8} {'0.1634':>8} {'0.8062':>10}")
print(f"{'tune_b':<8} {'c=1.5 g=0.7 d=2 s=30':<24} {'0.2371':>8} {'0.1683':>8} {'0.8045':>10}")
print(f"{'tune_f':<8} {'c=1.0 g=0.4 d=3 s=30':<24} {'0.2334':>8} {'0.1692':>8} {'0.8053':>10}")
PYEOF
