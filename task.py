"""
task.py

The single place that knows the shape of a pool item. Everywhere else in the
pipeline reads an item through these four accessors instead of destructuring a
tuple, so the federation loop, aggregation, and evaluation stay identical across
datasets.

AG News / DBpedia item : (text: str, label: str)   label in the global LABEL_SPACE
MMLU item              : MMLUExample                answer carried as an index 0..3

For the AG News branch every accessor is an identity transform over what the
existing code already did inline. That is deliberate: routing the old path
through this module must not change its behaviour, which is what lets the
seed-42 regression run certify the refactor.
"""

from config import DATASET

if DATASET == "mmlu":
    import dataclasses
    from mmlu_data import embedding_text, LETTER_TO_INDEX, MMLUExample

    def embed_text(item) -> str:
        """Text handed to the kNN embedder: the question stem only."""
        return embedding_text(item)

    def true_label(item) -> str:
        """Ground-truth label as a letter."""
        return item.answer_letter

    def with_label(item, label):
        """Return a copy carrying `label` as its (pseudo) answer.

        `label` is a letter or None. On None the item is returned unchanged;
        callers are expected to drop None-labelled items rather than store them,
        so this is a defensive no-op, not a silent default.
        """
        if label is None:
            return item
        return dataclasses.replace(item, answer_index=LETTER_TO_INDEX[label])

    def item_id(item):
        """Stable, hashable key for server-side maps (true labels, context)."""
        return (item.subject, item.question)

else:
    def embed_text(item) -> str:
        return item[0]

    def true_label(item) -> str:
        return item[1]

    def with_label(item, label):
        return (item[0], label)

    def item_id(item):
        return item[0]