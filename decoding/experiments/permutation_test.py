# Paired permutation test between two decoders' window-level scores
import numpy as np
import argparse


def paired_permutation_test(scores_a, scores_b, n_perms=10000, seed=42):
    np.random.seed(seed)
    diff = scores_b - scores_a
    observed = diff.mean()

    count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(diff))
        if (diff * signs).mean() >= observed:
            count += 1

    return observed, count / n_perms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_a", type=str, required=True)
    parser.add_argument("--scores_b", type=str, required=True)
    parser.add_argument("--task", type=str, default="wheretheressmoke")
    parser.add_argument("--n_perms", type=int, default=10000)
    args = parser.parse_args()

    ws_a = np.load(args.scores_a, allow_pickle=True)['window_scores'].item()
    ws_b = np.load(args.scores_b, allow_pickle=True)['window_scores'].item()

    for metric in ['WER', 'BLEU', 'METEOR', 'BERT']:
        a = ws_a[(args.task, metric)]
        b = ws_b[(args.task, metric)]
        diff, p = paired_permutation_test(a, b, args.n_perms)
        sig = "YES" if p < 0.05 else "NO"
        print(f"{metric}: A={a.mean():.4f} B={b.mean():.4f} diff={diff:+.4f} p={p:.4f} sig={sig}")
