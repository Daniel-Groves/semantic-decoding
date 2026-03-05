import os
import numpy as np
import json
import argparse
import h5py
import time

import config
from GPT import GPT
from MCTSDecoder import MCTSDecoder
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N time points for testing")
    # MCTS parameters
    parser.add_argument("--beam_width", type=int, default=config.MCTS_WIDTH)
    parser.add_argument("--simulations", type=int, default=config.MCTS_SIMULATIONS)
    parser.add_argument("--depth", type=int, default=config.MCTS_DEPTH)
    parser.add_argument("--c_puct", type=float, default=config.MCTS_CPUCT)
    parser.add_argument("--gamma", type=float, default=config.MCTS_GAMMA)
    parser.add_argument("--value_blend", type=float, default=config.MCTS_VALUE_BLEND)
    # Model paths
    parser.add_argument("--encoding_model", type=str, default=None,
                        help="Path to encoding model .npz (overrides default)")
    parser.add_argument("--word_rate_model", type=str, default=None,
                        help="Path to word rate model .npz (overrides default)")
    # Output suffix for distinguishing runs
    parser.add_argument("--suffix", type=str, default="mcts",
                        help="Suffix for output filename (default: mcts)")
    args = parser.parse_args()

    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"

    print("=" * 60)
    print("MCTS DECODER")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Experiment: {args.experiment}")
    print(f"Task: {args.task}")
    print(f"GPT checkpoint: {gpt_checkpoint}")
    print(f"Word rate model: {word_rate_voxels}")
    print(f"MCTS params: width={args.beam_width}, sims={args.simulations}, "
          f"depth={args.depth}, c_puct={args.c_puct}, gamma={args.gamma}")
    print("=" * 60)

    # load responses
    response_path = os.path.join(config.DATA_TEST_DIR, "test_response",
                                 args.subject, args.experiment, args.task + ".hf5")
    print(f"Loading brain responses from: {response_path}")
    hf = h5py.File(response_path, "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    print(f"Response data shape: {resp.shape}")

    # Limit data for testing if specified
    if args.limit is not None:
        original_shape = resp.shape
        resp = resp[:args.limit]
        print(f"LIMITED to first {args.limit} time points (was {original_shape[0]})")
        print(f"New response data shape: {resp.shape}")

    # load gpt
    print("Loading GPT models...")
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"),
              vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)
    print("GPT and language models loaded")

    # load models
    print("Loading trained models...")
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    if args.word_rate_model:
        wr_path = args.word_rate_model
    else:
        wr_path = os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels)
    if args.encoding_model:
        em_path = args.encoding_model
    else:
        em_path = os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint)
    print(f"Encoding model: {em_path}")
    print(f"Word rate model: {wr_path}")
    word_rate_model = np.load(wr_path, allow_pickle=True)
    encoding_model = np.load(em_path)
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model,
                       device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    print(f"Encoding model loaded ({len(encoding_model['voxels'])} voxels, "
          f"feature_dim={encoding_model['word_stats'][0].shape[0]})")

    # predict word times
    print("Predicting word timing from brain activity...")
    word_rate = predict_word_rate(resp, word_rate_model["weights"],
                                 word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech":
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=-10)
    else:
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    print(f"Predicted {len(word_times)} word times")

    # initialize MCTS decoder
    print("Initializing MCTS decoder...")
    decoder = MCTSDecoder(
        word_times=word_times,
        beam_width=args.beam_width,
        simulations=args.simulations,
        max_depth=args.depth,
        c_puct=args.c_puct,
        gamma=args.gamma,
        value_blend=args.value_blend,
    )
    sm = StimulusModel(lanczos_mat, encoding_model["tr_stats"],
                       encoding_model["word_stats"][0], device=config.SM_DEVICE)

    # decode
    n_words = len(word_times)
    print(f"Decoding {n_words} word positions with MCTS...")
    t_start = time.time()

    for sample_index in range(n_words):
        t_word_start = time.time()
        decoder.step(sample_index, lm, features, sm, em, lanczos_mat)
        t_word = time.time() - t_word_start

        if sample_index % 10 == 0 or sample_index == n_words - 1:
            elapsed = time.time() - t_start
            words_per_sec = (sample_index + 1) / elapsed if elapsed > 0 else 0
            eta = (n_words - sample_index - 1) / words_per_sec if words_per_sec > 0 else 0
            current_text = " ".join(decoder.beam[0].words[-10:]) if decoder.beam[0].words else ""
            print(f"  [{sample_index+1}/{n_words}] {t_word:.1f}s/word | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                  f"beam={len(decoder.beam)} | ...{current_text}")

    total_time = time.time() - t_start
    print(f"\nDecoding completed in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Average {total_time/n_words:.2f}s per word")

    # save results
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        decoder.word_times += 10

    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_name = f"{args.task}_{args.suffix}"
    decoder.save(os.path.join(save_location, save_name))
    print(f"Results saved to: {save_location}/{save_name}.npz")

    # Print decoded text
    decoded_words = decoder.beam[0].words
    print(f"\nDecoded text ({len(decoded_words)} words):")
    print(" ".join(decoded_words))
    print("\nDone!")
