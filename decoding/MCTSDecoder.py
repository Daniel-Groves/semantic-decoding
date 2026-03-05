import math
import numpy as np
from Decoder import Hypothesis
from LanguageModel import get_nucleus, context_filter, INIT
from StimulusModel import affected_trs


class MCTSNode:
    """A node in the MCTS search tree.

    Each node represents a word choice at a given depth in the lookahead.
    The root node (depth 0) corresponds to the current sample_index.
    """

    def __init__(self, word=None, logprob=0.0, emb=None, prior=0.0, parent=None):
        self.word = word
        self.logprob = logprob
        self.emb = emb
        self.prior = prior  # LM probability (PUCT prior)
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, c_puct):
        """PUCT selection score: Q + exploration bonus."""
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct):
        """Select child with highest PUCT score."""
        return max(self.children, key=lambda c: c.puct_score(c_puct))

    def backpropagate(self, value):
        """Update visit count and total value up to the root."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def most_visited_child(self):
        """Return the child with the most visits (robust selection)."""
        return max(self.children, key=lambda c: c.visit_count)

    def word_trace(self):
        """Trace words from root to this node (excluding root)."""
        words = []
        n = self
        while n.parent is not None:
            words.append(n.word)
            n = n.parent
        words.reverse()
        return words

    def emb_trace(self):
        """Trace embeddings from root to this node (excluding root)."""
        embs = []
        n = self
        while n.parent is not None:
            embs.append(n.emb)
            n = n.parent
        embs.reverse()
        return embs


class MCTSDecoder:
    """MCTS-based decoder that replaces beam search with tree search + lookahead.

    Maintains a beam of root hypotheses. For each root, runs MCTS simulations
    to evaluate the best next word using lookahead via the encoding model.
    """

    def __init__(self, word_times, beam_width, simulations, max_depth,
                 c_puct, gamma, value_blend=0.0, extensions=5):
        self.word_times = word_times
        self.beam_width = beam_width
        self.simulations = simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.gamma = gamma
        self.value_blend = value_blend
        self.extensions = extensions
        self.beam = [Hypothesis()]
        # LM distribution + embedding cache keyed by context tuple
        self._lm_cache = {}

    def first_difference(self):
        """Get first index where hypotheses on the beam differ."""
        words_arr = np.array([h.words for h in self.beam])
        if words_arr.shape[0] == 1:
            return words_arr.shape[1]
        for index in range(words_arr.shape[1]):
            if len(set(words_arr[:, index])) > 1:
                return index
        return 0

    def time_window(self, sample_index, seconds, floor=0):
        """Number of prior words within [seconds] of the currently sampled time point."""
        window = [t for t in self.word_times if t < self.word_times[sample_index]
                  and t > self.word_times[sample_index] - seconds]
        return max(len(window), floor)

    def _get_lm_candidates(self, context_words, lm, features):
        """Get nucleus-filtered LM candidates for a context. Results are cached.

        Returns (words, logprobs, embs) where embs are GPT hidden states.
        """
        context_key = tuple(context_words)

        if context_key in self._lm_cache:
            return self._lm_cache[context_key]

        # Get LM probabilities
        probs = lm.ps([context_words])
        probs = probs[0]  # single context

        # Nucleus filtering
        nuc_ids = get_nucleus(probs, nuc_mass=lm.nuc_mass, nuc_ratio=lm.nuc_ratio)
        nuc_words = [lm.model.vocab[i] for i in nuc_ids if i in lm.ids]

        # If all nucleus tokens were excluded by lm.ids, fall back to top
        # tokens from the full vocab that ARE in lm.ids
        if len(nuc_words) == 0:
            sorted_ids = np.argsort(-probs)
            for i in sorted_ids:
                if i in lm.ids:
                    nuc_words.append(lm.model.vocab[i])
                    if len(nuc_words) >= 5:
                        break

        filtered = context_filter(nuc_words, context_words)
        # Fall back to unfiltered nucleus if context_filter removes everything
        nuc_words = filtered if len(filtered) > 0 else nuc_words

        if len(nuc_words) == 0:
            self._lm_cache[context_key] = ([], [], [], [])
            return [], [], [], []

        nuc_logprobs = np.log([probs[lm.model.word2id[w]] for w in nuc_words])
        # Normalized probabilities for PUCT prior
        nuc_probs_raw = np.array([probs[lm.model.word2id[w]] for w in nuc_words])
        nuc_priors = nuc_probs_raw / nuc_probs_raw.sum()

        # Get embeddings for each candidate
        extend_words_list = [context_words + [w] for w in nuc_words]
        embs = list(features.extend(extend_words_list))

        result = (nuc_words, nuc_logprobs, embs, nuc_priors)
        self._lm_cache[context_key] = result
        return result

    def _get_init_candidates(self, lm):
        """Get initial word candidates (first word position)."""
        nuc_words = [w for w in INIT if lm.model.word2id[w] in lm.ids]
        nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
        nuc_priors = np.ones(len(nuc_words)) / len(nuc_words)
        return nuc_words, nuc_logprobs, nuc_priors

    def _expand_node(self, node, hyp, depth, sample_index, lm, features, context_words_count):
        """Expand a leaf node by adding all nucleus-filtered children."""
        # Build context from hypothesis + rollout words
        rollout_words = node.word_trace()
        all_words = hyp.words + rollout_words

        if len(all_words) == 0:
            # First word position — compute embeddings same as beam search
            words, logprobs, priors = self._get_init_candidates(lm)
            extend_words = [[w] for w in words]
            embs = list(features.extend(extend_words))
            for w, lp, emb, pr in zip(words, logprobs, embs, priors):
                child = MCTSNode(word=w, logprob=lp, emb=emb, prior=pr, parent=node)
                node.children.append(child)
        else:
            context = all_words[-context_words_count:]
            words, logprobs, embs, priors = self._get_lm_candidates(context, lm, features)

            for w, lp, emb, pr in zip(words, logprobs, embs, priors):
                child = MCTSNode(word=w, logprob=lp, emb=emb, prior=pr, parent=node)
                node.children.append(child)

        node.expanded = True

    def _greedy_rollout(self, node, hyp, sample_index, depth, lm, features,
                        context_words_count):
        """Do a greedy LM rollout from a node to max_depth, collecting embeddings.

        Returns list of (word, emb) pairs for each rollout step.
        """
        rollout = []
        current_words = hyp.words + node.word_trace()

        for d in range(depth, self.max_depth):
            if len(current_words) == 0:
                break
            context = current_words[-context_words_count:]
            words, logprobs, embs, priors = self._get_lm_candidates(context, lm, features)

            if len(words) == 0:
                break

            # Greedy: pick highest logprob
            best_idx = np.argmax(logprobs)
            best_word = words[best_idx]
            best_emb = embs[best_idx]
            rollout.append((best_word, best_emb))
            current_words = current_words + [best_word]

        return rollout

    def _evaluate_sequence(self, hyp, node, rollout, sample_index,
                           sm, em, lanczos_mat, features, context_words_count):
        """Evaluate a word sequence (node path + rollout) using the encoding model.

        Scores each position with gamma-discounted brain likelihood.
        """
        # Collect all words/embs from node trace + rollout
        node_words = node.word_trace()
        node_embs = node.emb_trace()

        all_words = node_words + [w for w, e in rollout]
        all_embs = node_embs + [e for w, e in rollout]

        if len(all_embs) == 0:
            return 0.0

        total_value = 0.0
        n_positions = len(all_words)

        for i in range(n_positions):
            future_index = sample_index + i
            if future_index >= len(self.word_times):
                break

            # Build history: hypothesis embs + lookahead embs up to position i
            history_embs = hyp.embs + all_embs[:i]
            var_emb = [all_embs[i]]

            # Verify history length matches future_index
            if len(history_embs) != future_index:
                break

            # Get affected TRs for this future position
            start_diff = self.first_difference()
            trs = affected_trs(start_diff, future_index, lanczos_mat)

            # Build stimulus and score
            stim = sm.make_variants(future_index, history_embs, var_emb, trs)
            likelihood = em.prs(stim, trs)[0]

            # Gamma-discounted value
            discount = self.gamma ** i
            total_value += discount * likelihood

        return total_value

    def _run_mcts(self, hyp, sample_index, lm, features, sm, em, lanczos_mat,
                  context_words_count):
        """Run MCTS simulations for a single hypothesis.

        Returns list of (Hypothesis, mcts_value) for each candidate next word.
        """
        root = MCTSNode()
        root.visit_count = 0

        for sim in range(self.simulations):
            # === Selection: traverse tree using PUCT ===
            node = root
            depth = 0
            while node.expanded and len(node.children) > 0 and depth < self.max_depth:
                node = node.best_child(self.c_puct)
                depth += 1

            # === Expansion: expand leaf if not at max depth ===
            if not node.expanded and depth < self.max_depth:
                self._expand_node(node, hyp, depth, sample_index, lm, features,
                                  context_words_count)
                # Pick an untried child (first one since we just expanded)
                if len(node.children) > 0:
                    node = node.children[0]
                    depth += 1

            # === Simulation: greedy rollout to max_depth ===
            rollout = self._greedy_rollout(node, hyp, sample_index, depth,
                                           lm, features, context_words_count)

            # === Evaluation: brain likelihood with gamma discount ===
            value = self._evaluate_sequence(hyp, node, rollout, sample_index,
                                            sm, em, lanczos_mat, features,
                                            context_words_count)

            # === Backpropagation ===
            node.backpropagate(value)

        # Collect results: for each child of root, create an extension hypothesis
        results = []
        if not root.expanded or len(root.children) == 0:
            return results

        for child in root.children:
            if child.visit_count == 0:
                continue
            ext_hyp = Hypothesis(parent=hyp, extension=(child.word, child.logprob, child.emb))
            results.append((ext_hyp, child.q_value, child.visit_count))

        return results

    def step(self, sample_index, lm, features, sm, em, lanczos_mat):
        """Run one decoding step: MCTS for each hypothesis, then prune beam.

        This replaces the beam search loop in run_decoder.py.
        """
        context_words_count = self.time_window(sample_index, 8, floor=5)

        # Clear per-step cache
        self._lm_cache = {}

        all_extensions = []

        for hyp in self.beam:
            results = self._run_mcts(hyp, sample_index, lm, features, sm, em,
                                     lanczos_mat, context_words_count)
            all_extensions.extend(results)

        if len(all_extensions) == 0:
            # No extensions found — fall back to greedy LM extension to keep
            # hypothesis lengths in sync with sample_index
            print(f"  WARNING: no MCTS extensions at sample_index={sample_index}, "
                  f"falling back to greedy LM")
            new_beam = []
            for hyp in self.beam:
                context = hyp.words[-context_words_count:] if hyp.words else []
                if len(context) == 0:
                    continue
                words, logprobs, embs, priors = self._get_lm_candidates(
                    context, lm, features)
                if len(words) > 0:
                    best = int(np.argmax(logprobs))
                    ext = Hypothesis(parent=hyp,
                                     extension=(words[best], logprobs[best], embs[best]))
                    new_beam.append(ext)
            if new_beam:
                self.beam = new_beam[:self.beam_width]
            return

        # Sort by MCTS Q-value and keep top beam_width
        all_extensions.sort(key=lambda x: -x[1])
        self.beam = [ext[0] for ext in all_extensions[:self.beam_width]]

    def save(self, path):
        """Save decoder results in same format as Decoder."""
        np.savez(path, words=np.array(self.beam[0].words),
                 times=np.array(self.word_times))
