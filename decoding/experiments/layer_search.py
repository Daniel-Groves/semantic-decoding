# Single-layer search: trains encoding model for each layer and reports correlations
import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import bootstrap_ridge

np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--layers", nargs="+", type=int, default=[18, 24, 30, 36, 42, 48])
    parser.add_argument("--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--nboots", type=int, default=10)
    args = parser.parse_args()

    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    print(f"Layer search for {args.subject}, {len(stories)} stories, layers {args.layers}")

    gpt = GPT(model_name=config.GPT_MODEL_NAME, device=config.GPT_DEVICE)
    rresp = get_resp(args.subject, stories, stack=True)

    results = {}
    for layer in args.layers:
        print(f"--- Layer {layer} ---")
        features = LMFeatures(model=gpt, layer=layer, context_words=config.GPT_WORDS)
        rstim, tr_stats, word_stats = get_stim(stories, features)

        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
        weights, alphas, bscorrs = bootstrap_ridge(
            rstim, rresp, use_corr=False, alphas=config.ALPHAS,
            nboots=args.nboots, chunklen=config.CHUNKLEN, nchunks=nchunks
        )

        bscorrs = bscorrs.mean(2).max(0)
        top_corrs = np.sort(bscorrs)[-config.VOXELS:]
        results[layer] = {
            'max_corr': bscorrs.max(),
            'mean_top_corr': top_corrs.mean(),
        }
        print(f"  Max corr: {results[layer]['max_corr']:.4f}, Mean top-{config.VOXELS}: {results[layer]['mean_top_corr']:.4f}")

    # Summary
    print(f"\n{'Layer':<10} {'Max Corr':<12} {'Mean Top Corr':<15}")
    for layer in sorted(results.keys()):
        r = results[layer]
        print(f"{layer:<10} {r['max_corr']:<12.4f} {r['mean_top_corr']:<15.4f}")
