"""
Fed-ICL Replication — Core Algorithm (with ordering)

Implements Fed-ICL (Wang et al., ICML 2025) with support for example ordering experiments.
"""

import numpy as np
from collections import Counter
from config import NUM_SHOTS, SELECTION_STRATEGY, ORDER_STRATEGY, SEED, FED_VARIANT

np.random.seed(SEED)


""" Embedding-based similarity (paper-faithful selection).
 Wang et al. perform kNN retrieval in sentence-embedding space with paraphrase-MiniLM-L6-v2 (Appendix C.1, Algorithm 2), applied both to the round-1 local-dataset prefilter and to context selection at lines 5 and 7 of Algorithm 1. 
 Implementing per-query kNN selection here makes the separate round-1 prefilter redundant for outputs: a query's top-n neighbours in the full pool are, by construction, contained in the union-of-top-n filtered pool, so the selected context is identical. 
 The prefilter in the paper is a cost optimisation, not a correctness step. """

_EMBED_MODEL = None
_EMB_CACHE = {}


def _get_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "SELECTION_STRATEGY='similarity_embedding' requires the "
                "sentence-transformers package (paper's retriever, "
                "paraphrase-MiniLM-L6-v2). Install with: "
                "pip install sentence-transformers"
            ) from e
        _EMBED_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _EMBED_MODEL


def _embed(texts: list) -> np.ndarray:
    """Embed texts with a module-level cache; rows align with input order.

    Embeddings are L2-normalised at encode time, so a dot product between
    two rows equals cosine similarity. The cache means each unique text
    (pool example, server query, eval query) is embedded exactly once per
    process, keeping the retrieval cost negligible next to LLM calls.
    """
    missing = [t for t in texts if t not in _EMB_CACHE]
    if missing:
        model = _get_embedder()
        vecs = model.encode(missing, normalize_embeddings=True,
                            show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _EMB_CACHE[t] = v
    return np.stack([_EMB_CACHE[t] for t in texts])

class FedICLClient:
    def __init__(self, client_id: int, local_data: list, model: str = None):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.relabelled_data = None

    def select_examples(self, query_text: str, pool: list, n: int) -> list:
        if len(pool) <= n:
            return list(pool)
        
        if SELECTION_STRATEGY == "similarity_embedding":
            """ Paper-faithful kNN (Wang et al., Algorithm 2): cosine
             similarity in paraphrase-MiniLM-L6-v2 space. Vectors are
             normalised, so dot product == cosine similarity."""
            pool_vecs = _embed([text for text, _ in pool])
            query_vec = _embed([query_text])[0]
            scores = pool_vecs @ query_vec
            top_indices = np.argsort(scores)[-n:]
            selected = [pool[i] for i in top_indices]
            # Shuffle so order_examples is the sole controller of order;
            # "original" ordering stays a true control.
            np.random.shuffle(selected)
            return selected

        if SELECTION_STRATEGY == "similarity":
            """ Lexical word-overlap. Documented deviation kept as an ablation
             arm; the paper's method is "similarity_embedding" above. """

            query_words = set(query_text.lower().split())
            scores = []
            for text, label in pool:
                example_words = set(text.lower().split())
                scores.append(len(query_words & example_words))
            top_indices = np.argsort(scores)[-n:]
            selected = [pool[i] for i in top_indices]
            # CRITICAL: shuffle so order_examples is the sole controller of order.
            np.random.shuffle(selected)
            return selected

        # Default: random selection.
        indices = np.random.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    def order_examples(self, examples: list, query_text: str) -> list:
        ex = list(examples)  # do not mutate caller's list
        if ORDER_STRATEGY == "original":
            return ex
        elif ORDER_STRATEGY == "similarity_ascending":
            query_words = set(query_text.lower().split())
            return sorted(ex, key=lambda x: len(set(x[0].lower().split()) & query_words))
        elif ORDER_STRATEGY == "similarity_descending":
            query_words = set(query_text.lower().split())
            return sorted(ex, key=lambda x: len(set(x[0].lower().split()) & query_words), reverse=True)
        elif ORDER_STRATEGY == "label_grouped":
            return sorted(ex, key=lambda x: x[1])
        elif ORDER_STRATEGY == "label_alternating":
            from collections import defaultdict
            by_label = defaultdict(list)
            for e in ex:
                by_label[e[1]].append(e)
            result = []
            while any(by_label.values()):
                for label in sorted(by_label.keys()):
                    if by_label[label]:
                        result.append(by_label[label].pop(0))
            return result
        elif ORDER_STRATEGY == "random_shuffle":
            shuffled = list(ex)
            np.random.shuffle(shuffled)
            return shuffled
        return ex

    def relabel_local_data(self, global_context: list):
        from llm import predict_with_icl
        self.relabelled_data = []
        for text, original_label in self.local_data:
            examples = self.select_examples(text, global_context, NUM_SHOTS)
            examples = self.order_examples(examples, text)
            predicted_label = predict_with_icl(examples, text, model=self.model)
            # Unparseable output (None): drop the example for this round rather than writing a wrong label into the few-shot pool.
            if predicted_label is not None:
                self.relabelled_data.append((text, predicted_label))

    def predict_server_queries(self, server_queries: list, global_context: list) -> list:
        from llm import predict_with_icl
        if self.relabelled_data:
            if FED_VARIANT == "fed_icl_free":
                # Fed-ICL-Free (Wang et al., App. C.3): condition on the
                # relabelled pool D_k^i only; ground-truth labels unused
                # at this step.
                example_pool = self.relabelled_data
            else:
                # Full Fed-ICL: condition on both D^i and D_k^i.
                example_pool = self.local_data + self.relabelled_data
        else:
            example_pool = self.local_data

        predictions = []
        for query_text, _ in server_queries:
            examples = self.select_examples(query_text, example_pool, NUM_SHOTS)
            examples = self.order_examples(examples, query_text)
            predicted_label = predict_with_icl(examples, query_text, model=self.model)
            predictions.append((query_text, predicted_label))

        return predictions


class FedICLServer:
    def __init__(self, server_queries: list):
        self.queries = server_queries
        self.true_labels = {text: label for text, label in server_queries}

        from data import LABEL_SPACE
        self.global_context = [
            (text, np.random.choice(LABEL_SPACE))
            for text, _ in server_queries
        ]
        # Tie-breaking for majority vote (Eq. 5 argmax is not unique on
        # ties; the paper does not specify a rule). Seeded RNG, independent
        # stream from partition/init randomness. Tie counts are recorded so
        # the tie rate can be reported alongside accuracy.
        self._tie_rng = np.random.default_rng(SEED + 1000)
        self.tie_count = 0
        self.ties_by_round = []

    def get_global_context(self) -> list:
        return self.global_context.copy()

    def aggregate_predictions(self, all_client_predictions: list):
        new_context = []
        round_ties = 0
        for q_idx in range(len(self.queries)):
            query_text = self.queries[q_idx][0]
            votes = []
            for client_preds in all_client_predictions:
                if q_idx < len(client_preds):
                    _, label = client_preds[q_idx]
                    if label is not None:   # exclude unparseable outputs
                        votes.append(label)

            if votes:
                vote_counts = Counter(votes)
                top_count = max(vote_counts.values())
                tied = sorted(l for l, c in vote_counts.items()
                              if c == top_count)
                if len(tied) == 1:
                    aggregated_label = tied[0]
                else:
                    # Uniform random among the argmax set, seeded for reproducibility. 
                    # Counter.most_common(1) broke ties by insertion order, which systematically favoured client 0 on 1-1-1 splits. tied is sorted first so the draw is deterministic given the RNG state.
                    aggregated_label = str(self._tie_rng.choice(tied))
                    round_ties += 1
            else:
                # All votes unparseable: retain the previous round's label.
                aggregated_label = self.global_context[q_idx][1]

            new_context.append((query_text, aggregated_label))

        self.global_context = new_context
        self.tie_count += round_ties
        self.ties_by_round.append(round_ties)

    def evaluate_context(self) -> dict:
        correct = 0
        total = len(self.global_context)
        per_class = {}

        for text, predicted_label in self.global_context:
            true_label = self.true_labels[text]
            if true_label not in per_class:
                per_class[true_label] = {"correct": 0, "total": 0}
            per_class[true_label]["total"] += 1
            if predicted_label == true_label:
                correct += 1
                per_class[true_label]["correct"] += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_class": per_class,
        }
