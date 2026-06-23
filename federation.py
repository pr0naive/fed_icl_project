"""
Fed-ICL Replication — Core Algorithm (with ordering)

Implements Fed-ICL (Wang et al., ICML 2025) with support for example ordering experiments.
"""

import numpy as np
from collections import Counter
from config import NUM_SHOTS, SELECTION_STRATEGY, ORDER_STRATEGY, SEED

np.random.seed(SEED)


class FedICLClient:
    def __init__(self, client_id: int, local_data: list, model: str = None):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.relabelled_data = None

    def select_examples(self, query_text: str, pool: list, n: int) -> list:
        if len(pool) <= n:
            return list(pool)

        if SELECTION_STRATEGY == "similarity":
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
            self.relabelled_data.append((text, predicted_label))

    def predict_server_queries(self, server_queries: list, global_context: list) -> list:
        from llm import predict_with_icl
        if self.relabelled_data:
            example_pool = self.relabelled_data
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

    def get_global_context(self) -> list:
        return self.global_context.copy()

    def aggregate_predictions(self, all_client_predictions: list):
        new_context = []
        for q_idx in range(len(self.queries)):
            query_text = self.queries[q_idx][0]
            votes = []
            for client_preds in all_client_predictions:
                if q_idx < len(client_preds):
                    _, label = client_preds[q_idx]
                    votes.append(label)

            if votes:
                vote_counts = Counter(votes)
                aggregated_label = vote_counts.most_common(1)[0][0]
            else:
                aggregated_label = self.global_context[q_idx][1]

            new_context.append((query_text, aggregated_label))

        self.global_context = new_context

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
