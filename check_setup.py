#!/usr/bin/env python3
"""
Verify that all required dependencies are installed and working.
Run this after pip installing requirements.txt
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_versions():
    """Check versions of critical packages"""
    print("\nChecking critical package versions:")
    
    try:
        import jiwer
        # jiwer 2.2.0 doesn't have __version__, so we test functionality
        from jiwer import wer
        result = wer("hello world", "hello")
        print(f"✓ jiwer: Working (WER test passed)")
    except Exception as e:
        print(f"✗ jiwer: {e}")
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except:
        print("✗ torch: Not available")
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except:
        print("✗ transformers: Not available")
    
    try:
        import h5py
        print(f"✓ h5py: {h5py.__version__}")
    except:
        print("✗ h5py: Not available")

def main():
    print("Semantic Decoding Dependencies Check")
    print("=" * 40)
    
    # Core packages
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"), 
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("h5py", "h5py"),
        ("nltk", "nltk"),
        ("jiwer", "jiwer"),
        ("datasets", "datasets"),
        ("evaluate", "evaluate"),
        ("bert_score", "bert-score"),
    ]
    
    success_count = 0
    for module, package in packages:
        if check_import(module, package):
            success_count += 1
    
    print(f"\n{success_count}/{len(packages)} packages imported successfully")
    
    if success_count == len(packages):
        print("All dependencies are working!")
        check_versions()
        print("\nReady to run semantic decoding pipeline!")
    else:
        print("Some dependencies are missing. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()