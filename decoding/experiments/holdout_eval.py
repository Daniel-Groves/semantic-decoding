# Leave-one-session-out evaluation comparing GPT-1 vs GPT2-XL encoding models
# Trains on 14 sessions, tests on held-out session, compares correlations
import os
import sys
import numpy as np
import json
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from GPT import GPT
from StimulusModel import LMFeatures, MixedLMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge_corr

np.random.seed(42)


def evaluate_holdout(train_stories, test_stories, features, subject, alphas):
    # Train on train_stories, test on test_stories, return per-voxel correlations
    rstim, tr_stats, word_stats = get_stim(train_stories, features)
    rresp = get_resp(subject, train_stories, stack=True)
    pstim = get_stim(test_stories, features, tr_stats=tr_stats)
    presp = get_resp(subject, test_stories, stack=True)

    Rcorrs = np.array(ridge_corr(rstim, pstim, rresp, presp, alphas, use_corr=False))
    return Rcorrs.max(0)  # best alpha per voxel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--gpt1_layer", type=int, default=9)
    parser.add_argument("--gpt2xl_layers", nargs="+", type=int, default=[24, 42])
    parser.add_argument("--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    alphas = config.ALPHAS

    # GPT-1
    print("Loading GPT-1...")
    gpt1 = GPT(model_name="openai-gpt", device=config.GPT_DEVICE)
    gpt1_features = LMFeatures(model=gpt1, layer=args.gpt1_layer, context_words=config.GPT_WORDS)

    gpt1_results = {}
    for sess in args.sessions:
        train_stories = [s for si in args.sessions if si != sess for s in sess_to_story[str(si)]]
        test_stories = sess_to_story[str(sess)]
        print(f"  Session {sess} holdout ({len(test_stories)} test, {len(train_stories)} train)")
        corrs = evaluate_holdout(train_stories, test_stories, gpt1_features, args.subject, alphas)
        top = np.sort(corrs)[-config.VOXELS:]
        gpt1_results[sess] = top.mean()
        print(f"    GPT-1: mean_top={top.mean():.4f}")
        gc.collect()

    del gpt1, gpt1_features; gc.collect()

    # GPT2-XL
    print("Loading GPT2-XL...")
    gpt2xl = GPT(model_name="gpt2-xl", device=config.GPT_DEVICE)
    if len(args.gpt2xl_layers) > 1:
        gpt2xl_features = MixedLMFeatures(model=gpt2xl, layers=args.gpt2xl_layers, context_words=config.GPT_WORDS)
    else:
        gpt2xl_features = LMFeatures(model=gpt2xl, layer=args.gpt2xl_layers[0], context_words=config.GPT_WORDS)

    gpt2xl_results = {}
    for sess in args.sessions:
        train_stories = [s for si in args.sessions if si != sess for s in sess_to_story[str(si)]]
        test_stories = sess_to_story[str(sess)]
        print(f"  Session {sess} holdout ({len(test_stories)} test, {len(train_stories)} train)")
        corrs = evaluate_holdout(train_stories, test_stories, gpt2xl_features, args.subject, alphas)
        top = np.sort(corrs)[-config.VOXELS:]
        gpt2xl_results[sess] = top.mean()
        print(f"    GPT2-XL: mean_top={top.mean():.4f}")
        gc.collect()

    # Summary
    print(f"\n{'Session':<10} {'GPT-1':<12} {'GPT2-XL':<12} {'Winner'}")
    gpt2xl_wins = 0
    for sess in args.sessions:
        g1, g2 = gpt1_results[sess], gpt2xl_results[sess]
        winner = "GPT2-XL" if g2 > g1 else "GPT-1"
        if g2 > g1: gpt2xl_wins += 1
        print(f"{sess:<10} {g1:<12.4f} {g2:<12.4f} {winner}")

    g1_avg = np.mean(list(gpt1_results.values()))
    g2_avg = np.mean(list(gpt2xl_results.values()))
    print(f"\nAverage:   {g1_avg:.4f}      {g2_avg:.4f}")
    print(f"GPT2-XL wins {gpt2xl_wins}/{len(args.sessions)} sessions")
