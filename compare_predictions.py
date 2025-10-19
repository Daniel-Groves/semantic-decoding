#!/usr/bin/env python3

import numpy as np
import sys
import os
import argparse

# Add decoding directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'decoding'))

from utils_eval import load_transcript

def compare_predictions(subject, experiment, task):
    """Compare decoder predictions with reference transcript"""
    
    # Load predictions
    pred_path = f"results/{subject}/{experiment}/{task}.npz"
    if not os.path.exists(pred_path):
        print(f"Error: Predictions file {pred_path} not found")
        return
        
    pred_data = np.load(pred_path)
    pred_words = pred_data["words"]
    pred_times = pred_data["times"]
    
    # Load reference
    try:
        ref_data = load_transcript(experiment, task)
        ref_words = ref_data["words"]
        ref_times = ref_data["times"]
    except Exception as e:
        print(f"Error loading reference: {e}")
        return
    
    print(f"COMPARISON: {task.upper()}")
    print("=" * 60)
    print(f"Predicted words: {len(pred_words)}")
    print(f"Reference words: {len(ref_words)}")
    print(f"Time range - Predicted: {pred_times[0]:.1f}s to {pred_times[-1]:.1f}s")
    print(f"Time range - Reference: {ref_times[0]:.1f}s to {ref_times[-1]:.1f}s")
    print()
    
    # Show sample text
    print("SAMPLE COMPARISON (first 50 words):")
    print("-" * 40)
    print("PREDICTED:")
    print(" ".join(pred_words[:50]))
    print()
    print("REFERENCE:")
    print(" ".join(ref_words[:50]))
    print()
    
    # Show end sample
    print("SAMPLE COMPARISON (last 20 words):")
    print("-" * 40)
    print("PREDICTED:")
    print(" ".join(pred_words[-20:]))
    print()
    print("REFERENCE:")
    print(" ".join(ref_words[-20:]))
    print()
    
    # Simple word overlap analysis
    pred_set = set(pred_words)
    ref_set = set(ref_words)
    
    overlap = len(pred_set & ref_set)
    pred_unique = len(pred_set - ref_set)
    ref_unique = len(ref_set - pred_set)
    
    print("VOCABULARY ANALYSIS:")
    print("-" * 40)
    print(f"Unique words in predictions: {len(pred_set)}")
    print(f"Unique words in reference: {len(ref_set)}")
    print(f"Overlapping words: {overlap}")
    print(f"Words only in predictions: {pred_unique}")
    print(f"Words only in reference: {ref_unique}")
    print(f"Vocabulary overlap: {overlap / len(ref_set):.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare decoder predictions with reference transcript')
    parser.add_argument('--subject', type=str, default='S1',
                        help='Subject ID (default: S1)')
    parser.add_argument('--experiment', type=str, default='perceived_movie',
                        choices=['perceived_movie', 'perceived_speech', 'perceived_multispeaker', 'imagined_speech'],
                        help='Experiment type (default: perceived_movie)')
    parser.add_argument('--task', type=str, default='laluna',
                        help='Task name (default: laluna)')
    
    args = parser.parse_args()
    
    compare_predictions(args.subject, args.experiment, args.task)