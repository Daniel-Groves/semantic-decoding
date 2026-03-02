import os
import numpy as np

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECODING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LM_DIR = os.path.join(DECODING_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(DECODING_DIR, "data_train")
DATA_TEST_DIR = os.path.join(DECODING_DIR, "data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = os.path.join(REPO_DIR, "results")
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 4, 10)
NBOOTS = 50
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 36
GPT_WORDS = 5

GPT_MODEL_NAME = "gpt2-xl"

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.025  # Reduced for GPT2-XL (more confident LM needs lower threshold)
EXTENSIONS = 5

# decoding model parameters (brain → embedding)

DM_TOP_K = 50           # Words to propose from brain signal
DM_LOG_FLOOR = -10.0    # LM log-prob floor for brain-proposed words

# evaluation parameters

WINDOW = 20

# devices


# can change to "cuda" if GPU available
GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"