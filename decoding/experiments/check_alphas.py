# Check regularisation alpha distribution in encoding models
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to encoding model .npz")
    args = parser.parse_args()

    em = np.load(args.model)
    alphas = em['alphas']
    print(f"Voxels: {len(alphas)}")
    print(f"Unique alphas: {np.unique(alphas)}")
    print(f"Median: {np.median(alphas)}")
    print(f"At max ({alphas.max():.0f}): {(alphas >= alphas.max() * 0.99).mean()*100:.1f}%")
    if alphas.max() > 1000:
        print(f"Above 1000: {(alphas > 1000).mean()*100:.1f}%")
