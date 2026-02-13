import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
from StimulusModel import LMFeatures, MixedLMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int,
        default = [2, 3, 4, 5, 6, 7, 8, 9])  # All available sessions (82 stories)
    parser.add_argument("--layers", nargs = "+", type = int, default = None,
        help = "GPT layers to use (multiple = concatenated mixed features)")
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    
    print(f"Loaded {len(stories)} training stories from {len(args.sessions)} sessions")
    print(f"Stories: {', '.join(stories[:5])}{'...' if len(stories) > 5 else ''}")

    # load gpt
    print("Loading GPT model...")
    gpt = GPT(model_name=config.GPT_MODEL_NAME, device=config.GPT_DEVICE)
    if args.layers and len(args.layers) > 1:
        features = MixedLMFeatures(model = gpt, layers = args.layers, context_words = config.GPT_WORDS)
        print(f"GPT model loaded (mixed layers {args.layers}, {config.GPT_WORDS} context words)")
    else:
        layer = args.layers[0] if args.layers else config.GPT_LAYER
        features = LMFeatures(model = gpt, layer = layer, context_words = config.GPT_WORDS)
        print(f"GPT model loaded (layer {layer}, {config.GPT_WORDS} context words)")
    
    # estimate encoding model
    print("Getting stimulus features...")
    rstim, tr_stats, word_stats = get_stim(stories, features)
    print(f"Getting brain responses for subject {args.subject}...")
    rresp = get_resp(args.subject, stories, stack = True)
    print(f"Data shapes: stimulus {rstim.shape}, response {rresp.shape}")
    
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    print(f"Running bootstrap ridge regression ({config.NBOOTS} boots, {nchunks} chunks, {len(config.ALPHAS)} alphas)...")
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    print(f"Selected top {config.VOXELS} voxels (max correlation: {bscorrs[vox].max():.3f})")
    del rstim, rresp
    
    # estimate noise model
    print("Building noise model...")
    stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in stories}
    resp_dict = get_resp(args.subject, stories, stack = False, vox = vox)
    noise_model = np.zeros([len(vox), len(vox)])
    
    print(f"Processing {len(stories)} leave-one-out iterations...")
    for i, hstory in enumerate(stories):
        if i % 5 == 0:  # Print progress every 5 stories
            print(f"  Progress: {i+1}/{len(stories)} stories ({100*(i+1)/len(stories):.1f}%)")
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    print("Noise model completed!")
    del stim_dict, resp_dict
    
    # save
    print("Saving encoding model...")
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))
    print(f"Encoding model saved to {save_location}/encoding_model_{args.gpt}.npz")
    print("Training complete!")