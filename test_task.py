"""
Tests for the task adapter.

The adapter reads config.DATASET at import, so a single process can only test
one branch. This file guards the AG News identity invariant (the property that
protects the canonical results). Run the MMLU branch with:
    FED_ICL_DATASET=mmlu python -m pytest test_task.py -q -k mmlu
"""

import os
import task


AGNEWS = os.environ.get("FED_ICL_DATASET", "agnews") != "mmlu"


def test_agnews_accessors_are_identity():
    if not AGNEWS:
        return
    it = ("Central bank raised rates.", "business")
    assert task.embed_text(it) == it[0]
    assert task.true_label(it) == it[1]
    assert task.with_label(it, "world") == (it[0], "world")
    assert task.item_id(it) == it[0]


def test_mmlu_accessors():
    if AGNEWS:
        return
    from mmlu_data import MMLUExample
    ex = MMLUExample("What is 2+2?", ["3", "4", "5", "6"], 1, "elementary_mathematics")
    assert task.embed_text(ex) == "What is 2+2?"          # stem only, not options
    assert task.true_label(ex) == "B"
    relabelled = task.with_label(ex, "D")
    assert relabelled.answer_index == 3
    assert relabelled.question == ex.question             # frozen copy, not mutation
    assert task.with_label(ex, None) is ex                # None is a no-op
    assert task.item_id(ex) == ("elementary_mathematics", "What is 2+2?")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} tests ran (branch: {'agnews' if AGNEWS else 'mmlu'})")