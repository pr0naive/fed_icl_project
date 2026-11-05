"""Offline stand-ins: synthetic MMLU items and a fake LLM, for testing the pipeline without a GPU."""

import random
import re
import zlib

from mmlu_data import MMLUExample


def fake_items(n_subjects=57, per_subject=40):
    rng = random.Random(0)
    items = []
    for s in range(n_subjects):
        for j in range(per_subject):
            items.append(MMLUExample(
                question=f"Subject {s} question {j}: what is item {rng.random():.6f}?",
                choices=[f"opt{k}" for k in range(4)],
                answer_index=rng.randrange(4),
                subject=f"subject_{s:02d}",
            ))
    return items


class FakeBackend:
    """Answers correctly with base rate 0.55, plus 0.06 per demonstration whose shown label is right."""

    def __init__(self, items):
        self.truth = {it.question: it.answer_letter for it in items}
        self.stats = {"calls": 0, "unparseable": 0, "errors": 0}
        self.constrain_mode = "fake"

    def _q(self, content):
        return re.search(r"Question: (.*?)\n", content).group(1)

    def classify(self, messages, seed):
        self.stats["calls"] += 1
        q = self._q(messages[-1]["content"])
        good = 0
        for u, a in zip(messages[1:-1:2], messages[2:-1:2]):
            good += self.truth[self._q(u["content"])] == a["content"]
        p = min(0.95, 0.55 + 0.06 * good)
        rng = random.Random(seed ^ zlib.crc32(q.encode()))
        true = self.truth[q]
        letter = true if rng.random() < p else rng.choice([l for l in "ABCD" if l != true])
        return {"letter": letter, "raw": letter, "top_logprobs": {letter: -0.1}}

    def classify_many(self, message_lists, seeds):
        return [self.classify(m, s) for m, s in zip(message_lists, seeds)]
