#PCA fitting for brain-conditioned decoder.
# Fits PCA on all training fMRI data and saves the projection matrix.

import os
import json
import argparse

import config
from brain_data_utils import fit_pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sessions", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--n_components", type=int, default=256)
    args = parser.parse_args()

    # Get training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    print(f"Fitting PCA on {len(stories)} stories for subject {args.subject}")

    save_path = os.path.join(config.MODEL_DIR, args.subject,
                             f"pca_{args.n_components}.npz")

    pca = fit_pca(args.subject, stories, n_components=args.n_components,
                  save_path=save_path)
    print("Done")
