# Mixed-layer search: tests combinations of layers by concatenating features
import os
import gc
import numpy as np
import json
import argparse

import config
from GPT import GPT
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import bootstrap_ridge

np.random.seed(42)


class MixedLMFeatures:
    # Extract and concatenate features from multiple GPT layers
    def __init__(self, model, layers, context_words=config.GPT_WORDS):
        self.model = model
        self.layers = layers
        self.context_words = context_words

    def make_stim(self, words):
        context_array = self.model.get_story_array(words, self.context_words)
        all_layer_embs = []
        for layer in self.layers:
            embs = self.model.get_hidden(context_array, layer=layer)
            layer_stim = np.vstack([
                embs[0, :self.context_words],
                embs[:context_array.shape[0] - self.context_words, self.context_words]
            ])
            all_layer_embs.append(layer_stim)
        return np.hstack(all_layer_embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--nboots", type=int, default=10)
    args = parser.parse_args()

    # Layer combinations to test
    layer_combos = [
        (6, 18), (6, 36), (12, 24), (12, 36), (18, 36), (18, 42),
        (24, 36), (24, 42), (6, 24, 42), (12, 18, 36), (18, 36, 42),
        (6, 18, 30, 42), (12, 24, 36, 48), (6, 12, 24, 36),
    ]

    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    print(f"Mixed layer search for {args.subject}, {len(stories)} stories, {len(layer_combos)} combos")

    gpt = GPT(model_name=config.GPT_MODEL_NAME, device=config.GPT_DEVICE)
    rresp = get_resp(args.subject, stories, stack=True)

    results = {}
    for layers in layer_combos:
        layers_str = "+".join(map(str, layers))
        print(f"--- Layers {layers_str} ---")

        features = MixedLMFeatures(model=gpt, layers=layers, context_words=config.GPT_WORDS)
        rstim, tr_stats, word_stats = get_stim(stories, features)

        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
        weights, alphas, bscorrs = bootstrap_ridge(
            rstim, rresp, use_corr=False, alphas=config.ALPHAS,
            nboots=args.nboots, chunklen=config.CHUNKLEN, nchunks=nchunks
        )

        bscorrs = bscorrs.mean(2).max(0)
        top_corrs = np.sort(bscorrs)[-config.VOXELS:]
        results[layers_str] = {
            'n_layers': len(layers),
            'feature_dim': rstim.shape[1],
            'max_corr': bscorrs.max(),
            'mean_top_corr': top_corrs.mean(),
        }
        print(f"  Features: {rstim.shape[1]}, Max corr: {bscorrs.max():.4f}, Mean top: {top_corrs.mean():.4f}")

        del rstim, tr_stats, word_stats, weights, alphas, bscorrs, top_corrs
        gc.collect()

    # Summary sorted by mean_top_corr
    print(f"\n{'Layers':<20} {'#Layers':<8} {'Features':<10} {'Max Corr':<12} {'Mean Top Corr':<15}")
    for layers_str, r in sorted(results.items(), key=lambda x: x[1]['mean_top_corr'], reverse=True):
        print(f"{layers_str:<20} {r['n_layers']:<8} {r['feature_dim']:<10} {r['max_corr']:<12.4f} {r['mean_top_corr']:<15.4f}")
