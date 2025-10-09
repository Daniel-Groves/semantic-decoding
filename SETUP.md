# Semantic Decoding Setup Instructions

This repository reproduces the semantic decoding pipeline from "Semantic reconstruction of continuous language from non-invasive brain recordings".

## Quick Setup

### 1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download data (if needed)
The repository expects data in:
- `data_train_raw/` - Training data from OpenNeuro
- `data_test_raw/` - Test data from OpenNeuro

## Key Dependencies

**Critical for evaluation**: 
- `jiwer==2.2.0` (newer versions have compatibility issues)
- `h5py==3.14.0` (for reading HDF5 brain data files)
- `torch==2.8.0` (for neural networks and GPT features)

## Pipeline Steps

1. **Train Encoding Models**: `python3 decoding/train_EM.py`
2. **Train Word Rate Models**: `python3 decoding/train_WR.py` 
3. **Run Decoder**: `python3 decoding/run_decoder.py --subject S1 --experiment perceived_movie --task laluna`
4. **Evaluate Results**: `python3 decoding/evaluate_predictions.py --subject S1 --experiment perceived_movie --task laluna --metrics WER`

## Viewing Results

Use the provided utility scripts:
- `python3 view_results.py` - Display evaluation metrics
- `python3 compare_predictions.py` - Compare predictions vs reference

## Configuration

Edit `decoding/config.py` to adjust:
- Number of voxels (`VOXELS`)
- Bootstrap iterations (`NBOOTS`) 
- Chunk length (`CHUNKLEN`)
- Device settings (`GPT_DEVICE`, `DEVICE`)

## Tested Environment

- Python 3.13
- macOS (ARM64)
- CPU-only execution (CUDA optional)

## Notes

- The pipeline works with reduced parameters for testing
- Full performance requires complete datasets and more computational resources
- Results may vary based on available training data and parameter settings