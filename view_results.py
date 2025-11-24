#!/usr/bin/env python3

import numpy as np
import sys
import os
import argparse

def display_results(npz_file):
    """Display evaluation results in a human-readable format"""
    
    if not os.path.exists(npz_file):
        print(f"Error: File {npz_file} not found")
        return
    
    print(f"Loading results from: {npz_file}")
    print("=" * 60)
    
    # Load the results
    results = np.load(npz_file, allow_pickle=True)
    
    print("Available data keys:", list(results.keys()))
    print()
    
    # Extract the score dictionaries
    window_scores = results['window_scores'].item()
    window_zscores = results['window_zscores'].item()
    story_scores = results['story_scores'].item()
    story_zscores = results['story_zscores'].item()
    
    print("SEMANTIC DECODING EVALUATION RESULTS")
    print("=" * 60)
    
    for key in window_scores.keys():
        reference, metric = key
        
        print(f"\nReference Task: {reference}")
        print(f"Metric: {metric}")
        print("-" * 40)
        
        # Get story-level scores
        story_score = story_scores[key]
        story_zscore = story_zscores[key]
        
        # Handle array vs scalar
        if hasattr(story_score, 'shape') and story_score.shape:
            story_score = float(story_score.mean())
        else:
            story_score = float(story_score)
            
        if hasattr(story_zscore, 'shape') and story_zscore.shape:
            story_zscore = float(story_zscore.mean())
        else:
            story_zscore = float(story_zscore)
        
        # Get window-level statistics
        window_mean = np.mean(window_scores[key])
        window_std = np.std(window_scores[key])
        window_zscore_mean = np.mean(window_zscores[key])
        window_zscore_std = np.std(window_zscores[key])
        
        print(f"Story-level performance:")
        print(f"  Score: {story_score:.4f}")
        print(f"  Z-score: {story_zscore:.4f}")
        
        print(f"\nWindow-level performance ({len(window_scores[key])} windows):")
        print(f"  Mean score: {window_mean:.4f} ± {window_std:.4f}")
        print(f"  Mean Z-score: {window_zscore_mean:.4f} ± {window_zscore_std:.4f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if metric == "WER":
            accuracy = story_score
            print(f"  Word accuracy: {accuracy:.1%}")
            print(f"  Word error rate: {1-accuracy:.1%}")
            if story_zscore > 2:
                print("  ✓ Significantly better than chance (Z > 2)")
            elif story_zscore > 0:
                print("  + Better than chance")
            else:
                print("  - Worse than chance")
        
        print()
    
    print("=" * 60)
    print("Notes:")
    print("- Scores are computed over sliding time windows")
    print("- Z-scores compare performance to null/random baselines")
    print("- Higher scores indicate better reconstruction")
    print("- Z-score > 2 indicates statistical significance")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display semantic decoding evaluation results')
    parser.add_argument('--subject', type=str, default='S1', 
                        help='Subject ID (default: S1)')
    parser.add_argument('--experiment', type=str, default='perceived_movie',
                        choices=['perceived_movie', 'perceived_speech', 'perceived_multispeaker', 'imagined_speech'],
                        help='Experiment type (default: perceived_movie)')
    parser.add_argument('--task', type=str, default='laluna',
                        help='Task name (default: laluna)')
    parser.add_argument('--file', type=str, default=None,
                        help='Direct path to .npz file (overrides other options)')
    
    args = parser.parse_args()
    
    if args.file:
        npz_file = args.file
    else:
        npz_file = f"scores/{args.subject}/{args.experiment}/{args.task}.npz"
    
    display_results(npz_file)
