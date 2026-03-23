import os
import numpy as np
import json
from sklearn.decomposition import PCA
import h5py
import torch

import config
from utils_stim import get_story_wordseqs
from utils_ridge.stimulus_utils import load_simulated_trfiles, TRFile


def fit_pca(subject, stories, n_components=256, save_path=None, voxel_mask=None):
    # fit PCA on training fMRI data and optionally save
    # returns pca (fitted sklearn PCA object)
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
    all_resp = []
    for story in stories:
        resp_path = os.path.join(subject_dir, f"{story}.hf5")
        with h5py.File(resp_path, "r") as hf:
            resp = np.nan_to_num(hf["data"][:]).astype(np.float32)
        if voxel_mask is not None:
            resp = resp[:, voxel_mask]
        all_resp.append(resp)
    all_resp = np.vstack(all_resp)
    print(f"PCA: fitting on {all_resp.shape[0]} TRs x {all_resp.shape[1]} voxels")

    pca = PCA(n_components=n_components)
    pca.fit(all_resp)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components explain {explained:.1%} variance")

    if save_path is not None:
        np.savez(save_path,
                 components=pca.components_,
                 mean=pca.mean_,
                 explained_variance=pca.explained_variance_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        print(f"PCA saved to {save_path}")

    return pca


def load_pca(path):
    # Load saved PCA model
    data = np.load(path)
    return {
        'components': data['components'],      # (n_components, n_voxels)
        'mean': data['mean'],                  # (n_voxels,)
        'explained_variance_ratio': data['explained_variance_ratio'],
    }


def apply_pca(resp, pca_data):
    # Apply PCA projection to fMRI data.
    centered = resp - pca_data['mean']
    return centered @ pca_data['components'].T


def get_brain_context_for_story(subject, story, pca_data, brain_window=20,
                                voxel_mask=None):
    # Extract brain context window for each word for a single story

    # Load fMRI response
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
    resp_path = os.path.join(subject_dir, f"{story}.hf5")
    with h5py.File(resp_path, "r") as hf:
        resp = np.nan_to_num(hf["data"][:]).astype(np.float32)
    if voxel_mask is not None:
        resp = resp[:, voxel_mask]

    # PCA project
    resp_pca = apply_pca(resp, pca_data)  # (n_TRs, pca_dim)
    n_trs = resp_pca.shape[0]

    # Get word sequences with timing info
    wordseqs = get_story_wordseqs([story])
    ds = wordseqs[story]
    word_times = ds.data_times
    tr_times = ds.tr_times
    words = ds.data

    # Map each word to its corresponding TRs
    brain_contexts = []
    valid_indices = []
    valid_words = []

    for wi, (word, wtime) in enumerate(zip(words, word_times)):
        # Find the TR closest to word time + max hemodynamic delay
        end_tr = np.argmin(np.abs(tr_times - wtime)) + max(config.STIM_DELAYS)
        start_tr = end_tr - brain_window + 1

        # Check all TRs are valid, words too near start of story will get skipped
        if start_tr >= 0 and end_tr < n_trs:
            # get array with brain_window TRs of PCA-projected activity
            context = resp_pca[start_tr:end_tr + 1]
            brain_contexts.append(context)
            # record index and word
            valid_indices.append(wi)
            valid_words.append(word)

    return brain_contexts, valid_words, valid_indices


def build_training_data(subject, stories, pca_data, context_words=5, brain_window=20,
                        voxel_mask=None):
    # Build all (text context, brain context, label) training data
    # Durign training the model gets first context_words (5) words as GPT input and brain window
    # as cross-attention context and tries to predict (context_words+1)th (6th) word
    all_contexts = []
    all_brain = []
    all_words = []

    for si, story in enumerate(stories):
        if si % 10 == 0:
            print(f"  Building data: {si+1}/{len(stories)} stories")

        # get brain window for each valid word pos in the story
        brain_contexts, words, valid_indices = get_brain_context_for_story(
            subject, story, pca_data, brain_window=brain_window,
            voxel_mask=voxel_mask
        )

        # Get full word list for context
        wordseqs = get_story_wordseqs([story])
        full_words = wordseqs[story].data

        for bc, word, wi in zip(brain_contexts, words, valid_indices):
            # Skip if not enough preceding context
            if wi < context_words:
                continue

            # Context = preceding words + current word
            context = list(full_words[wi - context_words:wi + 1])
            all_contexts.append(context)
            all_brain.append(bc)
            all_words.append(word)

    print(f"  Total samples: {len(all_contexts)}")
    return {
        'contexts': all_contexts,
        'brain_contexts': all_brain,
        'words': all_words,
    }


def build_contrastive_data(subject, stories, pca_data, features,
                           brain_window=20, hop_length=10, voxel_mask=None):
    # build windowed (brain, text_embedding) pairs for contrastive pre-training
    # e.g. pairs of window of brain activity and window of text (represented by average GPT-1 layer
    # 9 embedding of words in window)
    max_delay = max(config.STIM_DELAYS)
    all_brain = []
    all_text = []

    for si, story in enumerate(stories):
        if si % 10 == 0:
            print(f"  Contrastive data: {si+1}/{len(stories)} stories")

        # Load fMRI + PCA
        subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
        resp_path = os.path.join(subject_dir, f"{story}.hf5")
        with h5py.File(resp_path, "r") as hf:
            resp = np.nan_to_num(hf["data"][:]).astype(np.float32)
        if voxel_mask is not None:
            resp = resp[:, voxel_mask]
        resp_pca = apply_pca(resp, pca_data)
        n_trs = resp_pca.shape[0]

        # Get word sequences + text embeddings
        wordseqs = get_story_wordseqs([story])
        ds = wordseqs[story]
        word_times = ds.data_times
        tr_times = ds.tr_times
        words = ds.data

        # Get GPT-1 layer-9 embeddings for all words
        word_embs = features.make_stim(words)  # (n_words, 768)

        # Create windows
        for win_start in range(0, n_trs - brain_window - max_delay + 1, hop_length):
            # shift window by hemodynamic delay
            brain_start = win_start + max_delay
            brain_end = brain_start + brain_window

            if brain_end > n_trs:
                break

            # convert stimulus window to time in secconds to find which words are in window
            t_start = tr_times[win_start]
            t_end = tr_times[win_start + brain_window - 1]

            # create mask over all words (to indicate which are in time window)
            word_mask = (word_times >= t_start) & (word_times < t_end)
            # skip if less than 2 words in window
            if word_mask.sum() < 2:
                continue

            # Average text embeddings for words in this window
            text_target = word_embs[word_mask].mean(axis=0)  # (768,)

            # Brain context
            brain_ctx = resp_pca[brain_start:brain_end]  # (brain_window, pca_dim)

            all_brain.append(brain_ctx)
            all_text.append(text_target)

    print(f"  Total contrastive pairs: {len(all_brain)}")
    return all_brain, all_text


class BrainContextProvider:
    # maps sample_index to brain_tokens tensor at decode time

    def __init__(self, resp_pca, word_times, tr_times, brain_window=20):
        self.resp_pca = resp_pca
        self.word_times = word_times
        self.tr_times = tr_times
        self.brain_window = brain_window
        self.n_trs = resp_pca.shape[0]

    def get_context(self, sample_index):
        # returns (1, brain_window, pca_dim) tensor or None if out of bounds
        
        if sample_index >= len(self.word_times):
            return None

        # get prediceted time for word index
        wtime = self.word_times[sample_index]
        end_tr = np.argmin(np.abs(self.tr_times - wtime)) + max(config.STIM_DELAYS)
        start_tr = end_tr - self.brain_window + 1

        if start_tr < 0 or end_tr >= self.n_trs:
            return None

        # extract window
        context = self.resp_pca[start_tr:end_tr + 1]  # (brain_window, pca_dim)
        return torch.tensor(context, dtype=torch.float32).unsqueeze(0)
