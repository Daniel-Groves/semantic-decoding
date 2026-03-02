import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures, MixedLMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--limit", type = int, default = None, help = "Limit to first N time points for testing")
    parser.add_argument("--layers", nargs = "+", type = int, default = None,
        help = "GPT layers to use (multiple = concatenated mixed features)")
    parser.add_argument("--lm_model", type = str, default = None,
        help = "Separate LM model for beam search (e.g. openai-gpt). If not set, uses same model as encoding.")
    parser.add_argument("--lm_ratio", type = float, default = None,
        help = "Override LM_RATIO for language model nucleus sampling")
    args = parser.parse_args()
    
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"
    
    print(f"Decoding experiment: {args.experiment}")
    print(f"Task: {args.task}")
    print(f"Subject: {args.subject}")
    print(f"Using GPT checkpoint: {gpt_checkpoint}")
    print(f"Using word rate model: {word_rate_voxels}")

    # load responses
    response_path = os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5")
    print(f"Loading brain responses from: {response_path}")
    hf = h5py.File(response_path, "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    print(f"Response data shape: {resp.shape}")
    
    # Limit data for testing if specified
    if args.limit is not None:
        original_shape = resp.shape
        resp = resp[:args.limit]
        print(f"LIMITED to first {args.limit} time points for testing (was {original_shape[0]} points)")
        print(f"New response data shape: {resp.shape}")

    # load gpt
    print("Loading GPT models...")

    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)

    # Load encoding model GPT
    gpt = GPT(model_name=config.GPT_MODEL_NAME, device=config.GPT_DEVICE)
    if args.layers and len(args.layers) > 1:
        features = MixedLMFeatures(model = gpt, layers = args.layers, context_words = config.GPT_WORDS)
        print(f"Encoding features: mixed layers {args.layers} ({config.GPT_MODEL_NAME})")
    else:
        layer = args.layers[0] if args.layers else config.GPT_LAYER
        features = LMFeatures(model = gpt, layer = layer, context_words = config.GPT_WORDS)
        print(f"Encoding features: layer {layer} ({config.GPT_MODEL_NAME})")

    # Load language model (optionally separate from encoding model)
    lm_ratio = args.lm_ratio if args.lm_ratio is not None else config.LM_RATIO
    if args.lm_model:
        lm_gpt = GPT(model_name=args.lm_model, device=config.GPT_DEVICE)
        lm = LanguageModel(lm_gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = lm_ratio)
        print(f"Language model: {args.lm_model} (LM_RATIO={lm_ratio})")
    else:
        lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = lm_ratio)
        print(f"Language model: {config.GPT_MODEL_NAME} (LM_RATIO={lm_ratio})")
    print(f"LM vocab matched: {len(lm.ids)} tokens")

    # load models
    print("Loading trained models...")
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    print(f"Encoding model loaded ({len(encoding_model['voxels'])} voxels)")
    
    # predict word times
    print("Predicting word timing from brain activity...")
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    print(f"Predicted {len(word_times)} word times")

    # decode responses
    print("Starting decoding process...")
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
    
    print(f"Processing {len(word_times)} time points...")
    for sample_index in range(len(word_times)):
        if sample_index % 50 == 0:  # Progress every 50 samples
            print(f"  Progress: {sample_index+1}/{len(word_times)} ({100*(sample_index+1)/len(word_times):.1f}%)")
        
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
    
    print("Decoding completed!")
        
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, args.task))
    print(f"Results saved to: {save_location}/{args.task}")
    print("Decoding complete! Check the results file for the reconstructed text.")