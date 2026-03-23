#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-6
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --job-name=mp_tune_S1
#SBATCH -o mp_tune_S1.o%j

echo "========================================================"
echo "MASK-PREDICT HYPERPARAMETER SWEEP — S1"
echo "========================================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"

module load apps/binapps/anaconda3/2023.09
source activate decoding-env
module load libs/cuda/12.4.1

cd /mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/decoding

BASELINE_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/models/final_paper/baseline
RESULTS_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/results/S1/perceived_speech
SCORES_DIR=/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech
BASELINE_RESULT=${RESULTS_DIR}/wheretheressmoke_baseline.npz
EM=${BASELINE_DIR}/S1_baseline_encoding_model.npz
WR=${BASELINE_DIR}/S1_baseline_word_rate_auditory.npz

run_config() {
    local NAME=$1 ITER=$2 FRAC=$3 ALPHA=$4 BETA=$5 TOPK=$6
    echo ""
    echo "=== Config ${NAME}: iter=${ITER} frac=${FRAC} alpha=${ALPHA} beta=${BETA} top_k=${TOPK} ==="
    echo "Started: $(date)"

    python3 -u run_maskpredict_decoder.py \
        --subject S1 --experiment perceived_speech --task wheretheressmoke \
        --baseline ${BASELINE_RESULT} \
        --n_iterations ${ITER} --mask_fraction ${FRAC} \
        --alpha ${ALPHA} --beta ${BETA} --top_k ${TOPK} \
        --suffix mp_${NAME} \
        --encoding_model ${EM} --word_rate_model ${WR}

    cp ${RESULTS_DIR}/wheretheressmoke_mp_${NAME}.npz ${RESULTS_DIR}/wheretheressmoke.npz
    python3 -u evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
    cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_mp_${NAME}.npz

    echo "Finished ${NAME}: $(date)"
}

# --- Evaluate baseline first for comparison ---
echo ""
echo "=== Evaluating baseline ==="
cp ${RESULTS_DIR}/wheretheressmoke_baseline.npz ${RESULTS_DIR}/wheretheressmoke.npz
python3 -u evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
cp ${SCORES_DIR}/wheretheressmoke.npz ${SCORES_DIR}/wheretheressmoke_baseline.npz

# --- Axis 1: beta (brain weight) with default mask_fraction=0.15 ---
# A: baseline config (already have this but re-run for consistency)
run_config A 10 0.15 1.0 1.0 30

# B-D: increase brain weight
run_config B 10 0.15 1.0 2.0 30
run_config C 10 0.15 1.0 3.0 30
run_config D 10 0.15 1.0 5.0 30

# E: decrease brain weight (BERT dominant)
run_config E 10 0.15 2.0 1.0 30

# --- Axis 2: mask_fraction (conservative masking) ---
run_config F 10 0.05 1.0 1.0 30
run_config G 10 0.05 1.0 2.0 30
run_config H 10 0.05 1.0 3.0 30
run_config I 10 0.10 1.0 2.0 30
run_config J 10 0.10 1.0 3.0 30

# --- Axis 3: top_k (more candidates for brain to pick from) ---
run_config K 10 0.15 1.0 2.0 50
run_config L 10 0.05 1.0 2.0 50

# --- Axis 4: more iterations ---
run_config M 20 0.15 1.0 2.0 30
run_config N 20 0.05 1.0 2.0 30

# --- Axis 5: very conservative + strong brain ---
run_config O 10 0.03 1.0 3.0 30
run_config P 10 0.03 1.0 5.0 30
run_config Q 5 0.05 1.0 5.0 30

echo ""
echo "========================================================"
echo "SWEEP COMPLETE — SUMMARY"
echo "========================================================"

python3 << 'PYEOF'
import numpy as np

scores_dir = "scores/S1/perceived_speech"
configs = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q"]
labels = {
    "A": "b=1.0 f=0.15 k=30",
    "B": "b=2.0 f=0.15 k=30",
    "C": "b=3.0 f=0.15 k=30",
    "D": "b=5.0 f=0.15 k=30",
    "E": "a=2.0 b=1.0 f=0.15",
    "F": "b=1.0 f=0.05 k=30",
    "G": "b=2.0 f=0.05 k=30",
    "H": "b=3.0 f=0.05 k=30",
    "I": "b=2.0 f=0.10 k=30",
    "J": "b=3.0 f=0.10 k=30",
    "K": "b=2.0 f=0.15 k=50",
    "L": "b=2.0 f=0.05 k=50",
    "M": "b=2.0 f=0.15 i=20",
    "N": "b=2.0 f=0.05 i=20",
    "O": "b=3.0 f=0.03 k=30",
    "P": "b=5.0 f=0.03 k=30",
    "Q": "b=5.0 f=0.05 i=5",
}

# Also load baseline
print(f"{'Config':<6} {'Description':<25} {'WER':>6} {'BLEU-1':>7} {'METEOR':>7} {'BERT':>7}")
print("-" * 65)

try:
    d = np.load(f"{scores_dir}/wheretheressmoke.npz", allow_pickle=True)
    ss = d["story_scores"].item()
    # This might be the last config's scores, so load baseline separately
except:
    pass

# Load baseline scores
try:
    d = np.load(f"/mnt/iusers01/fse-ugpgt01/compsci01/u05730dg/semantic-decoding/scores/S1/perceived_speech/wheretheressmoke_baseline.npz", allow_pickle=True)
    ss = d["story_scores"].item()
    wer = float(np.mean(ss[("wheretheressmoke", "WER")]))
    bleu = float(np.mean(ss[("wheretheressmoke", "BLEU")]))
    meteor = float(np.mean(ss[("wheretheressmoke", "METEOR")]))
    bert = float(np.mean(ss[("wheretheressmoke", "BERT")]))
    print(f"{'BASE':<6} {'beam search baseline':<25} {wer:>6.3f} {bleu:>7.3f} {meteor:>7.3f} {bert:>7.3f}")
except Exception as e:
    print(f"{'BASE':<6} ERROR: {e}")

print("-" * 65)

for cfg in configs:
    try:
        d = np.load(f"{scores_dir}/wheretheressmoke_mp_{cfg}.npz", allow_pickle=True)
        ss = d["story_scores"].item()
        wer = float(np.mean(ss[("wheretheressmoke", "WER")]))
        bleu = float(np.mean(ss[("wheretheressmoke", "BLEU")]))
        meteor = float(np.mean(ss[("wheretheressmoke", "METEOR")]))
        bert = float(np.mean(ss[("wheretheressmoke", "BERT")]))
        print(f"{cfg:<6} {labels[cfg]:<25} {wer:>6.3f} {bleu:>7.3f} {meteor:>7.3f} {bert:>7.3f}")
    except Exception as e:
        print(f"{cfg:<6} ERROR: {e}")

PYEOF

echo ""
echo "Job finished on $(date)"
