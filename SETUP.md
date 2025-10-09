# Semantic Decoding Setup Instructions

This repository reproduces the semantic decoding pipeline from "Semantic reconstruction of continuous language from non-invasive brain recordings".

## Quick Setup

### 1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows PowerShell
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download data using DataLad

#### Install DataLad (if not already installed)
```bash
# Via conda (recommended for Windows)
conda install -c conda-forge datalad

# OR via pip
pip install datalad
```

#### Clone the datasets
```bash
# Training dataset (ds003020)
datalad clone https://github.com/OpenNeuroDatasets/ds003020.git data_train_raw

# Test dataset (ds004510)  
datalad clone https://github.com/OpenNeuroDatasets/ds004510.git data_test_raw
```

#### Get the data files
```bash
# Get training data
cd data_train_raw
datalad get derivative/preprocessed_data/
datalad get derivative/TextGrids/
datalad get derivative/respdict.json
datalad get derivative/sess_to_story.json

# Get test data
cd ../data_test_raw
datalad get derivative/preprocessed_data/
datalad get derivative/TextGrids/

cd ..
```

#### Copy data to expected structure
```bash
cd decoding

# Create directories
mkdir -p data_train/train_stimulus data_train/train_response/{S1,S2,S3}
mkdir -p data_test/test_stimulus/{imagined_speech,perceived_movie,perceived_multispeaker,perceived_speech}
mkdir -p data_test/test_response/{S1,S2,S3}/{imagined_speech,perceived_movie,perceived_multispeaker,perceived_speech}

# Copy training data
cp ../data_train_raw/derivative/TextGrids/*.TextGrid data_train/train_stimulus/
cp ../data_train_raw/derivative/preprocessed_data/S1/*.hf5 data_train/train_response/S1/
cp ../data_train_raw/derivative/preprocessed_data/S2/*.hf5 data_train/train_response/S2/
cp ../data_train_raw/derivative/preprocessed_data/S3/*.hf5 data_train/train_response/S3/
cp ../data_train_raw/derivative/sess_to_story.json data_train/
cp ../data_train_raw/derivative/respdict.json data_train/

# Copy test data (mapping UTS01->S1, UTS02->S2, UTS03->S3)
cp ../data_test_raw/derivative/TextGrids/imagined_speech/*.TextGrid data_test/test_stimulus/imagined_speech/
cp ../data_test_raw/derivative/TextGrids/perceived_movie/*.TextGrid data_test/test_stimulus/perceived_movie/
cp ../data_test_raw/derivative/TextGrids/perceived_multispeaker/*.TextGrid data_test/test_stimulus/perceived_multispeaker/
cp ../data_test_raw/derivative/TextGrids/perceived_speech/*.TextGrid data_test/test_stimulus/perceived_speech/

cp ../data_test_raw/derivative/preprocessed_data/UTS01/imagined_speech/*.hf5 data_test/test_response/S1/imagined_speech/
cp ../data_test_raw/derivative/preprocessed_data/UTS01/perceived_movie/*.hf5 data_test/test_response/S1/perceived_movie/
cp ../data_test_raw/derivative/preprocessed_data/UTS01/perceived_multispeaker/*.hf5 data_test/test_response/S1/perceived_multispeaker/
cp ../data_test_raw/derivative/preprocessed_data/UTS01/perceived_speech/*.hf5 data_test/test_response/S1/perceived_speech/

# Repeat for S2 and S3 (UTS02->S2, UTS03->S3)...
```

### 4. Download Language Model Data
Download `data_lm.tar.gz` from the Box link in the original README and extract to `decoding/data_lm/`

### 5. Windows PowerShell Commands
For Windows users, use these PowerShell equivalents:

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "data_train\train_stimulus"
New-Item -ItemType Directory -Force -Path "data_train\train_response\S1"
New-Item -ItemType Directory -Force -Path "data_train\train_response\S2" 
New-Item -ItemType Directory -Force -Path "data_train\train_response\S3"
New-Item -ItemType Directory -Force -Path "data_test\test_stimulus\imagined_speech"
New-Item -ItemType Directory -Force -Path "data_test\test_stimulus\perceived_movie"
New-Item -ItemType Directory -Force -Path "data_test\test_stimulus\perceived_multispeaker"
New-Item -ItemType Directory -Force -Path "data_test\test_stimulus\perceived_speech"
New-Item -ItemType Directory -Force -Path "data_test\test_response\S1\imagined_speech"
New-Item -ItemType Directory -Force -Path "data_test\test_response\S1\perceived_movie"
New-Item -ItemType Directory -Force -Path "data_test\test_response\S1\perceived_multispeaker"
New-Item -ItemType Directory -Force -Path "data_test\test_response\S1\perceived_speech"

# Copy training data
Copy-Item "..\data_train_raw\derivative\TextGrids\*.TextGrid" "data_train\train_stimulus\"
Copy-Item "..\data_train_raw\derivative\preprocessed_data\S1\*.hf5" "data_train\train_response\S1\"
Copy-Item "..\data_train_raw\derivative\preprocessed_data\S2\*.hf5" "data_train\train_response\S2\"
Copy-Item "..\data_train_raw\derivative\preprocessed_data\S3\*.hf5" "data_train\train_response\S3\"
Copy-Item "..\data_train_raw\derivative\sess_to_story.json" "data_train\"
Copy-Item "..\data_train_raw\derivative\respdict.json" "data_train\"

# Copy test data
Copy-Item "..\data_test_raw\derivative\TextGrids\imagined_speech\*.TextGrid" "data_test\test_stimulus\imagined_speech\"
Copy-Item "..\data_test_raw\derivative\TextGrids\perceived_movie\*.TextGrid" "data_test\test_stimulus\perceived_movie\"
Copy-Item "..\data_test_raw\derivative\TextGrids\perceived_multispeaker\*.TextGrid" "data_test\test_stimulus\perceived_multispeaker\"
Copy-Item "..\data_test_raw\derivative\TextGrids\perceived_speech\*.TextGrid" "data_test\test_stimulus\perceived_speech\"

Copy-Item "..\data_test_raw\derivative\preprocessed_data\UTS01\imagined_speech\*.hf5" "data_test\test_response\S1\imagined_speech\"
Copy-Item "..\data_test_raw\derivative\preprocessed_data\UTS01\perceived_movie\*.hf5" "data_test\test_response\S1\perceived_movie\"
Copy-Item "..\data_test_raw\derivative\preprocessed_data\UTS01\perceived_multispeaker\*.hf5" "data_test\test_response\S1\perceived_multispeaker\"
Copy-Item "..\data_test_raw\derivative\preprocessed_data\UTS01\perceived_speech\*.hf5" "data_test\test_response\S1\perceived_speech\"
```

## Pipeline Steps

1. **Train Encoding Models**: `python3 decoding/train_EM.py --subject S1 --gpt perceived`
2. **Train Word Rate Models**: `python3 decoding/train_WR.py --subject S1` 
3. **Run Decoder**: `python3 decoding/run_decoder.py --subject S1 --experiment perceived_movie --task laluna`
4. **Evaluate Results**: `python3 decoding/evaluate_predictions.py --subject S1 --experiment perceived_movie --task laluna --metrics WER`

## Viewing Results

Use the provided utility scripts:
- `python3 view_results.py` - Display evaluation metrics
- `python3 compare_predictions.py` - Compare predictions vs reference

## Configuration

Edit `decoding/config.py` to adjust:
- Number of voxels (`EM_N_VOXELS`)
- Bootstrap iterations (`EM_N_BOOTS`) 
- Chunk count (`EM_N_CHUNKS`)
- Device settings (`DEVICE`)

For GPU acceleration:
```python
DEVICE = "cuda"  # For GPU
EM_N_VOXELS = 10000  # Full voxel count
EM_N_BOOTS = 10  # More bootstrap samples
EM_N_CHUNKS = 100  # More chunks
```

## Key Dependencies

**Critical for evaluation**: 
- `jiwer==2.2.0` (newer versions have compatibility issues)
- `h5py==3.14.0` (for reading HDF5 brain data files)
- `torch==2.8.0` (for neural networks and GPT features)

## Tested Environment

- Python 3.9-3.13
- macOS (ARM64) and Windows
- CPU and CUDA GPU support

## Notes

- For Windows, use PowerShell instead of CMD
- The pipeline works with reduced parameters for testing
- Full performance requires complete datasets and GPU acceleration
- Results may vary based on available training data and parameter settings

## OpenNeuro Datasets

- **Training**: [ds003020](https://github.com/OpenNeuroDatasets/ds003020) - Perceived speech fMRI data
- **Test**: [ds004510](https://github.com/OpenNeuroDatasets/ds004510) - Multi-condition speech decoding data

## Troubleshooting

### Common Issues

1. **Tables import error**: Fixed with h5py fallback in `utils_ridge/util.py`
2. **jiwer compatibility**: Use `jiwer==2.2.0` specifically
3. **GPU memory**: Reduce `EM_N_VOXELS` if running out of memory
4. **DataLad issues**: Ensure git-annex is installed for DataLad functionality

### Verification

Run `python3 check_setup.py` to verify all dependencies are working correctly.