# Takes an already-decoded sequence (e.g. from beam search) and iteratively improves
# by masking low-confidence words and replacing them using BERT + brain likelihood

import numpy as np
import torch
from StimulusModel import affected_trs

# Window size in words for BERT processing (keeps under 512 tokens)
BERT_WINDOW = 200
BERT_STRIDE = 100  # overlap between windows


class MaskPredictDecoder:
    """Takes an already-decoded sequence (e.g. from beam search) and iteratively improves
    by masking low-confidence words and replacing them using BERT + brain likelihood
    """

    def __init__(self, initial_words, word_times, lanczos_mat,
                 features, sm, em,
                 bert_model, bert_tokenizer, bert_vocab_ids,
                 n_iterations=10, mask_fraction=0.15,
                 alpha=1.0, beta=1.0, top_k=30,
                 use_brain=True, device="cuda"):
        self.current_words = list(initial_words)
        self.word_times = word_times
        self.lanczos_mat = lanczos_mat
        self.features = features
        self.sm = sm
        self.em = em
        self.bert = bert_model
        self.tokenizer = bert_tokenizer
        self.bert_vocab_ids = bert_vocab_ids
        self.n_iterations = n_iterations
        self.mask_fraction = mask_fraction
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.use_brain = use_brain
        self.device = device

        # Calculate vocab ID list once for fast indexing
        self._vocab_id_list = list(self.bert_vocab_ids.keys())
        self._vocab_id_tensor = torch.tensor(self._vocab_id_list, dtype=torch.long)

        # Keep track of words that are single BERT tokens, as otherwise we can't easily mask
        self._single_token_words = set(self.bert_vocab_ids.values())

        # Precompute GPT-1 embeddings for the initial sequence
        self.current_embs = self.features.make_stim(self.current_words)

    def refine(self, verbose=True):
        # iterative refinement loop. 
        # returns refined word list
        n_words = len(self.current_words)

        for iteration in range(self.n_iterations):
            # mask fewer percentage of words as iterations go on
            frac = self.mask_fraction * (1 - iteration / self.n_iterations)
            n_mask = max(1, int(frac * n_words))

            # calculate BERT confidence for each word position and pick the least-confident positions
            # BERT is bidirectional so it is also taking into account the words after the mask
            # unlike GPT which is generating candidates based on what has been generated so far
            confidence = self._compute_confidence()

            mask_indices = np.argsort(confidence)[:n_mask]
            mask_indices = np.sort(mask_indices)

            # for each masked position replace the word with the best replacement word (if they differ)
            n_changed = 0
            for pos in mask_indices:
                old_word = self.current_words[pos]
                new_word = self._replace_position(pos)
                if new_word != old_word:
                    self.current_words[pos] = new_word
                    n_changed += 1

            # recompute GPT-1 embeddings for whole sequence if changes are made
            if n_changed > 0:
                self.current_embs = self.features.make_stim(self.current_words)

            if verbose:
                print(f"  Iteration {iteration+1}/{self.n_iterations}: "
                      f"masked={n_mask}, changed={n_changed}")

            if n_changed == 0:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}")
                break

        return self.current_words

    def _get_window_for_pos(self, pos):
        # Get a window of words centered on pos that fits in BERT's 512 tokens
        # Returns (window_words, offset) where offset is the index of pos within window_word
        
        n_words = len(self.current_words)
        half = BERT_WINDOW // 2
        start = max(0, pos - half)
        end = min(n_words, start + BERT_WINDOW)
        start = max(0, end - BERT_WINDOW)  # adjust if near end
        window_words = self.current_words[start:end]
        offset = pos - start
        return window_words, start, offset

    def _tokenize_window(self, window_words):
        # Tokenize a window of words and return (input_ids, word_to_tokens mapping)
        text = " ".join(window_words)
        tokens = self.tokenizer(text, return_tensors="pt",
                                return_offsets_mapping=True,
                                truncation=True, max_length=512)
        input_ids = tokens["input_ids"][0]
        offsets = tokens["offset_mapping"][0]
        word_to_tokens = self._align_words_to_tokens(window_words, text, offsets)
        return input_ids, word_to_tokens

    def _compute_confidence(self):
        """Compute BERT pseudo-log-likelihood for each position using sliding windows."""
        n_words = len(self.current_words)
        confidence = np.full(n_words, -np.inf)

        # Sliding windows
        for win_start in range(0, n_words, BERT_STRIDE):
            win_end = min(win_start + BERT_WINDOW, n_words)
            window_words = self.current_words[win_start:win_end]
            input_ids, word_to_tokens = self._tokenize_window(window_words)

            with torch.no_grad():
                for wi, token_span in enumerate(word_to_tokens):
                    global_wi = win_start + wi
                    # Only score positions roughly in the center of the window to avoid
                    # edge effects, except at the actual boundaries
                    # as positions will appear in multiple overlapping windows 
                    if win_start > 0 and wi < BERT_STRIDE // 2:
                        continue
                    if win_end < n_words and wi >= len(window_words) - BERT_STRIDE // 2:
                        continue
                    if confidence[global_wi] > -np.inf:
                        continue  # already scored

                    # Skip words not in single-token BERT vocab (e.g. contractions)
                    # give neutral confidence so they won't be masked
                    word = self.current_words[global_wi]
                    if word not in self._single_token_words:
                        confidence[global_wi] = 0.0
                        continue

                    if token_span is None:
                        confidence[global_wi] = 0.0
                        continue

                    start_tok, end_tok = token_span
                    masked_ids = input_ids.clone().unsqueeze(0).to(self.device)

                    # replace this word's tokens with [MASK]
                    for ti in range(start_tok, end_tok):
                        masked_ids[0, ti] = self.tokenizer.mask_token_id

                    # run BERT on masked sequence
                    logits = self.bert(masked_ids).logits[0]

                    # for each masked token position see what log-prob BERT gives original token to work out confidence
                    log_prob = 0.0
                    for ti in range(start_tok, end_tok):
                        orig_id = input_ids[ti].item()
                        lp = torch.log_softmax(logits[ti], dim=-1)[orig_id].item()
                        log_prob += lp
                    confidence[global_wi] = log_prob

            if win_end >= n_words:
                break

        # Any remaining -inf positions get score 0 (safety net)
        # confidence[confidence == -np.inf] = 0.0

        return confidence

    def _replace_position(self, pos):
        # Find best replacement word at position pos

        # Get BERTS best candidates
        candidates, bert_scores = self._get_bert_candidates(pos)

        if not self.use_brain or len(candidates) == 1:
            return candidates[np.argmax(bert_scores)]

        brain_scores = self._score_brain(pos, candidates)

        bert_z = self._zscore(bert_scores)
        brain_z = self._zscore(brain_scores)
        # combine brain and bert score to pick the best replacement
        combined = self.alpha * bert_z + self.beta * brain_z

        return candidates[np.argmax(combined)]

    def _get_bert_candidates(self, pos):
        # Get top-k BERT MLM candidates at position pos, filtered to decoder vocab

        # window sequence around pos, tokenize and mask target word's tokens
        window_words, win_start, offset = self._get_window_for_pos(pos)
        input_ids, word_to_tokens = self._tokenize_window(window_words)
        token_span = word_to_tokens[offset]

        if token_span is None:
            return [], np.array([])

        # replace target word with [MASK]
        start_tok, end_tok = token_span

        masked_ids = input_ids.clone().unsqueeze(0).to(self.device)
        for ti in range(start_tok, end_tok):
            masked_ids[0, ti] = self.tokenizer.mask_token_id

        # run BERT on masked sequence and get predictions at every position
        with torch.no_grad():
            logits = self.bert(masked_ids).logits[0]

        log_probs = torch.log_softmax(logits[start_tok], dim=-1)

        # extract log probs for the words in our vocab
        vocab_scores = log_probs[self._vocab_id_tensor].cpu().numpy()

        # return top-k scores along with log-probs
        top_indices = np.argsort(-vocab_scores)[:self.top_k]
        candidates = [self.bert_vocab_ids[self._vocab_id_list[i]] for i in top_indices]
        scores = vocab_scores[top_indices]

        return candidates, scores

    def _score_brain(self, pos, candidates):
        # Score candidate words using GPT-1 encoding model

        # build context
        context_start = max(0, pos - self.features.context_words)
        extend_words_list = []
        for cand in candidates:
            modified = list(self.current_words[context_start:pos]) + [cand]
            extend_words_list.append(modified)

        # get gpt-1 embedding for candidate in context
        cand_embs = list(self.features.extend(extend_words_list))

        # Use (pos, pos) — we're only changing one word
        try:
            trs = affected_trs(pos, pos, self.lanczos_mat)
        except IndexError:
            # Position has no affected TRs (e.g. beyond lanczos range)
            return np.zeros(len(candidates))

        history_embs = list(self.current_embs[:pos])

        # use EM to predict brain response and compare against actual fMRI
        stim = self.sm.make_variants(pos, history_embs, cand_embs, trs)
        brain_scores = self.em.prs(stim, trs)

        return brain_scores

    def _align_words_to_tokens(self, words, text, offsets):
        # Map each word index to its (start_token, end_token) span in BERT tokenization

        # find where each word starts in the string
        word_char_spans = []
        char_pos = 0
        for word in words:
            start = text.find(word, char_pos)
            if start == -1:
                word_char_spans.append(None)
                continue

            # record character span
            end = start + len(word)
            word_char_spans.append((start, end))
            char_pos = end

        result = []
        for wspan in word_char_spans:
            if wspan is None:
                result.append(None)
                continue

            # works out the token spans for each word (as some words span multiple tokens)
            w_start, w_end = wspan
            tok_start, tok_end = None, None

            for ti, (ts, te) in enumerate(offsets):
                ts, te = ts.item(), te.item()
                if ts == 0 and te == 0:
                    continue
                if ts < w_end and te > w_start:
                    if tok_start is None:
                        tok_start = ti
                    tok_end = ti + 1

            result.append((tok_start, tok_end) if tok_start is not None else None)

        return result

    @staticmethod
    def _zscore(scores):
        # Z-score normalize an array
        # returns zeros if std is 0
        std = scores.std()
        if std < 1e-8:
            return np.zeros_like(scores)
        return (scores - scores.mean()) / std
