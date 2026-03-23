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
# These defaults are for GPT-1 (fine-tuned OpenAI GPT, 12 layers, 768-dim).
# For GPT-2 XL: set GPT_LAYER = 36, LM_RATIO = 0.025,
# ALPHAS = np.logspace(1, 4, 10), and use GPT2XL.py instead of GPT.py.

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 50
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# MCTS decoder parameters

MCTS_WIDTH = 20
MCTS_SIMULATIONS = 60
MCTS_DEPTH = 3
MCTS_CPUCT = 2.5
MCTS_GAMMA = 0.7
MCTS_VALUE_BLEND = 0.0

# evaluation parameters

WINDOW = 20

# devices


# can change to "cuda" if GPU available
GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"