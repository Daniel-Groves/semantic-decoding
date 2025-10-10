#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append('/Users/danielgroves/GitRepos/semantic-decoding/decoding')

from utils_eval import load_transcript, windows, segment_data, WER
import config
import json

def quick_test():
    print("Testing evaluation components...")
    
    # Load reference and predictions
    ref_data = load_transcript('perceived_movie', 'laluna')
    pred_data = np.load(os.path.join(config.RESULT_DIR, 'S1', 'perceived_movie', 'laluna.npz'))
    
    print(f"✓ Loaded data: ref={len(ref_data['words'])}, pred={len(pred_data['words'])}")
    
    # Load eval segments and create windows
    with open(os.path.join(config.DATA_TEST_DIR, 'eval_segments.json'), 'r') as f:
        eval_segments = json.load(f)
    
    window_cutoffs = windows(*eval_segments['laluna'], config.WINDOW)
    print(f"✓ Generated {len(window_cutoffs)} windows")
    
    # Segment both datasets
    ref_windows = segment_data(ref_data['words'], ref_data['times'], window_cutoffs)
    pred_windows = segment_data(pred_data['words'], pred_data['times'], window_cutoffs)
    
    print(f"✓ Segmented data: {len(ref_windows)} ref windows, {len(pred_windows)} pred windows")
    
    # Test WER calculation on just the first few windows
    print("\nTesting WER calculation on first 3 windows...")
    wer_metric = WER()
    
    for i in range(3):
        ref_seg = ref_windows[i]
        pred_seg = pred_windows[i]
        print(f"Window {i}: {len(ref_seg)} ref words, {len(pred_seg)} pred words")
        
        try:
            # Test on single window
            score = wer_metric.score([ref_seg], [pred_seg])
            print(f"  ✓ WER score: {score[0]:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n✓ All tests passed! The evaluation should work.")
    return True

if __name__ == "__main__":
    quick_test()