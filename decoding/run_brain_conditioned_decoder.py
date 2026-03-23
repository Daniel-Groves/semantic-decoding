import os
import numpy as np
import json
import argparse
import h5py
import torch

import config
from BrainConditionedGPT import BrainConditionedGPT
from brain_data_utils import load_pca, apply_pca, BrainContextProvider
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel, get_nucleus, context_filter, INIT, STOPWORDS
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times


class BrainConditionedLM(LanguageModel):
    """LanguageModel that threads brain context through to the GPT model."""

    def __init__(self, model, vocab, brain_provider, nuc_mass=1.0, nuc_ratio=0.0):
        super().__init__(model, vocab, nuc_mass, nuc_ratio)
        self.brain_provider = brain_provider
        self._current_brain_context = None

    def set_sample_index(self, sample_index):
        # Set current brain context from sample index
        self._current_brain_context = self.brain_provider.get_context(sample_index)

    def ps(self, contexts):
        # Get probability distributions conditioned on brain context
        context_arr = self.model.get_context_array(contexts)
        # Expand brain context to match batch size
        brain_ctx = self._current_brain_context
        if brain_ctx is not None:
            brain_ctx = brain_ctx.expand(len(contexts), -1, -1)
        probs = self.model.get_probs(context_arr, brain_context=brain_ctx)
        return probs[:, len(contexts[0]) - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Cross-attention checkpoint filename")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for model dir and output file")
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--use_em_voxels", action="store_true")
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--encoder_ff_mult", type=int, default=2)
    parser.add_argument("--brain_window", type=int, default=20)
    parser.add_argument("--cross_attn_layers", nargs="+", type=int, default=[11])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--encoding_model_path", type=str, default=None,
                        help="Override encoding model path")
    parser.add_argument("--no_brain_lm", action="store_true",
                        help="Disable brain conditioning in LM (ablation)")
    args = parser.parse_args()

    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"

    if args.experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"

    print("Brain-Conditioned Decoder")
    print(f"Subject: {args.subject}, Task: {args.task}")
    print(f"Cross-attn layers: {args.cross_attn_layers}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Brain LM: {not args.no_brain_lm}")

    # Load test responses
    response_path = os.path.join(config.DATA_TEST_DIR, "test_response",
                                 args.subject, args.experiment, args.task + ".hf5")
    print(f"Loading brain responses from: {response_path}")
    with h5py.File(response_path, "r") as hf:
        resp = np.nan_to_num(hf["data"][:])
    print(f"Response shape: {resp.shape}")

    if args.limit is not None:
        resp = resp[:args.limit]
        print(f"Limited to {args.limit} time points")

    # Load PCA and project test fMRI (keep full resp for word rate + encoding models)
    if args.use_em_voxels:
        em_path = os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz")
        em_voxels = np.load(em_path)['voxels']
        resp_for_pca = resp[:, em_voxels]
        pca_path = os.path.join(config.MODEL_DIR, args.subject, f"pca_{args.pca_dim}_emvox.npz")
        print(f"Using EM voxels ({len(em_voxels)}), PCA from {pca_path}")
    else:
        resp_for_pca = resp
        pca_path = os.path.join(config.MODEL_DIR, args.subject, f"pca_{args.pca_dim}.npz")
        print(f"Loading PCA from {pca_path}")
    pca_data = load_pca(pca_path)
    resp_pca = apply_pca(resp_for_pca, pca_data)
    print(f"PCA-projected response: {resp_pca.shape}")

    # Load vanilla GPT for encoding model features (must match encoding model's GPT)
    print("Loading GPT models")
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    from GPT import GPT as VanillaGPT
    vanilla_gpt = VanillaGPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"),
                              vocab=gpt_vocab, device=config.GPT_DEVICE)

    # Load brain-conditioned GPT
    print("Loading Brain-Conditioned GPT")
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
        bc_vocab = json.load(f)
    gpt = BrainConditionedGPT(
        path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
        vocab=bc_vocab, device=config.GPT_DEVICE, pca_dim=args.pca_dim,
        cross_attn_layers=tuple(args.cross_attn_layers),
        encoder_layers=args.encoder_layers, encoder_ff_mult=args.encoder_ff_mult
    )

    # Load trained cross-attention weights
    model_dir = os.path.join(config.MODEL_DIR, args.subject,
                             "brain_conditioned" + args.suffix)
    checkpoint_path = os.path.join(model_dir, args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    gpt.load_trainable(checkpoint_path)
    gpt.eval_mode()

    # Print gate values
    for layer_idx in gpt.cross_attn_layers:
        m = gpt.cross_attn_modules[str(layer_idx)]
        gc = torch.tanh(m.gate_cross).item()
        gf = torch.tanh(m.gate_ff).item()
        print(f"Layer {layer_idx}: gate_cross={gc:.4f}, gate_ff={gf:.4f}")

    # Predict word times
    print("Predicting word timing")
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location,
                              f"word_rate_model_{word_rate_voxels}.npz"), allow_pickle=True)
    word_rate = predict_word_rate(resp, word_rate_model["weights"],
                                 word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech":
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=-10)
    else:
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    print(f"Predicted {len(word_times)} word times")

    # Create brain context provider to map each word pos to brain window
    brain_provider = BrainContextProvider(resp_pca, word_times, tr_times,
                                             brain_window=args.brain_window)

    # Encoding model features, uses vanilla GPT
    features = LMFeatures(model=vanilla_gpt, layer=config.GPT_LAYER,
                          context_words=config.GPT_WORDS)

    # LM proposals with brain-conditioned GPT
    if args.no_brain_lm:
        lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)
    else:
        lm = BrainConditionedLM(gpt, decoder_vocab, brain_provider,
                                nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    # Load encoding model
    em_path = args.encoding_model_path or os.path.join(load_location,
                                                       f"encoding_model_{gpt_checkpoint}.npz")
    print(f"Loading encoding model from: {em_path}")
    encoding_model = np.load(em_path)
    em = EncodingModel(resp, encoding_model["weights"], encoding_model["voxels"],
                       encoding_model["noise_model"], device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    assert args.task not in encoding_model["stories"]
    print(f"Encoding model loaded ({len(encoding_model['voxels'])} voxels)")

    # Decode
    print(f"Decoding {len(word_times)} words...")
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)

    for sample_index in range(len(word_times)):
        if sample_index % 50 == 0:
            print(f"  Progress: {sample_index+1}/{len(word_times)} "
                  f"({100*(sample_index+1)/len(word_times):.1f}%)")

        # Update brain context for this word position
        if hasattr(lm, 'set_sample_index'):
            lm.set_sample_index(sample_index)

        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent=hyp, extension=x)
                                for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)

    print("Decoding complete")

    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        decoder.word_times += 10

    # Save with suffix
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_name = args.task + ("_crossattn" + args.suffix if args.suffix else "_crossattn")
    decoder.save(os.path.join(save_location, save_name))
    print(f"Results saved to: {save_location}/{save_name}")
