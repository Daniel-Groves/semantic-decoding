import os
import numpy as np
import json
import argparse
import h5py
import time

import config
from GPT import GPT
from MaskPredictDecoder import MaskPredictDecoder
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline .npz result file")
    # Mask-predict parameters
    parser.add_argument("--n_iterations", type=int, default=10)
    parser.add_argument("--mask_fraction", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for BERT MLM score")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for brain likelihood score")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of BERT candidates per position")
    parser.add_argument("--no_brain", action="store_true",
                        help="Ablation: BERT-only, no brain scoring")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    # Model paths
    parser.add_argument("--encoding_model", type=str, default=None)
    parser.add_argument("--word_rate_model", type=str, default=None)
    # Output
    parser.add_argument("--suffix", type=str, default="maskpred")
    args = parser.parse_args()

    # determine GPT checkpoint
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"

    print("=" * 60)
    print("MASK-PREDICT DECODER")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Experiment: {args.experiment}")
    print(f"Task: {args.task}")
    print(f"Baseline: {args.baseline}")
    print(f"Params: iterations={args.n_iterations}, mask_frac={args.mask_fraction}, "
          f"alpha={args.alpha}, beta={args.beta}, top_k={args.top_k}")
    print(f"BERT model: {args.bert_model}")
    print(f"Brain scoring: {'OFF' if args.no_brain else 'ON'}")
    print("=" * 60)

    # Load baseline result
    print("Loading baseline result...")
    baseline_data = np.load(args.baseline, allow_pickle=True)
    initial_words = list(baseline_data["words"])
    word_times = baseline_data["times"]
    print(f"Baseline: {len(initial_words)} words")

    # Load brain responses
    response_path = os.path.join(config.DATA_TEST_DIR, "test_response",
                                 args.subject, args.experiment, args.task + ".hf5")
    print(f"Loading brain responses from: {response_path}")
    hf = h5py.File(response_path, "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    print(f"Response data shape: {resp.shape}")

    # Load GPT-1 (for brain scoring via encoding model)
    print("Loading GPT-1 model...")
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"),
              vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    print("GPT-1 loaded")

    # Load encoding model
    print("Loading encoding model...")
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    em_path = args.encoding_model or os.path.join(
        load_location, "encoding_model_%s.npz" % gpt_checkpoint)
    print(f"Encoding model: {em_path}")
    encoding_model = np.load(em_path)
    em = EncodingModel(resp, encoding_model["weights"], encoding_model["voxels"],
                       encoding_model["noise_model"], device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)

    # Build stimulus model and lanczos matrix
    tr_times = np.arange(resp.shape[0]) * 2  # TRs at 2s intervals
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    sm = StimulusModel(lanczos_mat, encoding_model["tr_stats"],
                       encoding_model["word_stats"][0], device=config.SM_DEVICE)

    # Load BERT
    print(f"Loading BERT ({args.bert_model})...")
    from transformers import BertForMaskedLM, BertTokenizerFast
    bert_tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model).eval().to(config.GPT_DEVICE)
    print("BERT loaded")

    # Build decoder vocab ↔ BERT vocab intersection
    # Only single-token BERT words that are in the decoder vocabulary
    decoder_vocab_set = set(decoder_vocab)
    bert_vocab_ids = {}
    for word in decoder_vocab:
        tokens = bert_tokenizer.tokenize(word)
        if len(tokens) == 1 and not tokens[0].startswith("##"):
            token_id = bert_tokenizer.convert_tokens_to_ids(tokens[0])
            bert_vocab_ids[token_id] = word
    print(f"Decoder vocab: {len(decoder_vocab)}, "
          f"BERT-compatible: {len(bert_vocab_ids)} "
          f"({len(bert_vocab_ids)/len(decoder_vocab)*100:.1f}%)")

    # Create decoder
    print("Initializing mask-predict decoder...")
    decoder = MaskPredictDecoder(
        initial_words=initial_words,
        word_times=word_times,
        lanczos_mat=lanczos_mat,
        features=features,
        sm=sm,
        em=em,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        bert_vocab_ids=bert_vocab_ids,
        n_iterations=args.n_iterations,
        mask_fraction=args.mask_fraction,
        alpha=args.alpha,
        beta=args.beta,
        top_k=args.top_k,
        use_brain=not args.no_brain,
        device=config.GPT_DEVICE,
    )

    # Run refinement
    print(f"\nRefining {len(initial_words)} words...")
    t_start = time.time()
    refined_words = decoder.refine(verbose=True)
    total_time = time.time() - t_start
    print(f"\nRefinement completed in {total_time:.1f}s ({total_time/60:.1f}min)")

    # Count changes
    n_changed = sum(1 for a, b in zip(initial_words, refined_words) if a != b)
    print(f"Words changed: {n_changed}/{len(initial_words)} "
          f"({n_changed/len(initial_words)*100:.1f}%)")

    # Save results
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        word_times = word_times + 10

    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_name = f"{args.task}_{args.suffix}"
    np.savez(os.path.join(save_location, save_name),
             words=np.array(refined_words), times=np.array(word_times))
    print(f"Results saved to: {save_location}/{save_name}.npz")

    # Print refined text
    print(f"\nRefined text ({len(refined_words)} words):")
    print(" ".join(refined_words))
    print("\nDone!")
