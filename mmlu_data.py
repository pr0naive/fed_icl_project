"""
mmlu_data.py

Data representation and serialization for the MMLU extension of the Fed-ICL
pipeline. Defines the normalized example structure, the single source of truth
for the letter/index mapping, and the functions that turn an example into
demonstration text (answer shown) or query text (answer withheld).

Notation map (Fed-ICL paper -> code):
    x_i (demonstration input)  : MMLUExample.question (+ options)
    y_i (demonstration label)  : MMLUExample.answer_index, displayed as a letter
    serialized demonstration   : serialize_demonstration(ex, include_answer=True)
    query (label withheld)     : serialize_demonstration(ex, include_answer=False)
    kNN embedding input        : embedding_text(ex)
    assembled ICL prompt       : build_prompt(demonstrations, query)

Design note: there is no silent default label anywhere. Anything that cannot be
mapped to a valid option is rejected (loader) or returns None (parser, in the
sibling module), mirroring the fix that replaced the "world" fallback in the
AG News pipeline.
"""

import re
from dataclasses import dataclass
from typing import Optional, Sequence

# Single source of truth for the option label space.
# MMLU stores the answer as an integer 0..3; ICL shows and generates a letter.
LETTERS = ("A", "B", "C", "D")
INDEX_TO_LETTER = {i: letter for i, letter in enumerate(LETTERS)}
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}
NUM_OPTIONS = len(LETTERS)


@dataclass(frozen=True)
class MMLUExample:
    question: str
    choices: Sequence[str]   # exactly NUM_OPTIONS entries; order is meaningful
    answer_index: int        # 0..3, ground truth
    subject: str             # e.g. "high_school_biology", used for partitioning

    def __post_init__(self):
        if len(self.choices) != NUM_OPTIONS:
            raise ValueError(
                f"expected {NUM_OPTIONS} choices, got {len(self.choices)} "
                f"(subject={self.subject})"
            )
        if self.answer_index not in INDEX_TO_LETTER:
            raise ValueError(
                f"answer_index {self.answer_index} out of range "
                f"0..{NUM_OPTIONS - 1} (subject={self.subject})"
            )

    @property
    def answer_letter(self) -> str:
        return INDEX_TO_LETTER[self.answer_index]


def _format_options(choices: Sequence[str]) -> str:
    return "\n".join(
        f"{INDEX_TO_LETTER[i]}. {choice}" for i, choice in enumerate(choices)
    )


def serialize_demonstration(ex: MMLUExample, include_answer: bool) -> str:
    """
    Render one example as a standard MMLU multiple-choice block.

    include_answer=True  : a demonstration (question, options, answer letter)
    include_answer=False : a query (answer left open for the model to complete)
    """
    block = f"Question: {ex.question}\n{_format_options(ex.choices)}\nAnswer:"
    if include_answer:
        return f"{block} {ex.answer_letter}"
    return block


def embedding_text(ex: MMLUExample, include_options: bool = False) -> str:
    """
    Text handed to the kNN embedder (paraphrase-MiniLM-L6-v2).

    Default keys retrieval on the question stem only. include_options=True
    appends the options, which can help when the stem is terse (many MMLU stems
    are just "Which of the following ...") but can also dilute the signal. This
    is a knob to sweep, not a fixed choice.
    """
    if include_options:
        return f"{ex.question}\n{_format_options(ex.choices)}"
    return ex.question


_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Respond with only the single letter (A, B, C, or D) of the correct option, "
    "and nothing else.\n\n"
)


def build_prompt(demonstrations: Sequence[MMLUExample], query: MMLUExample) -> str:
    """
    Assemble the full ICL prompt: an instruction, ordered demonstrations, then
    the query.

    The instruction constrains weak models (e.g. phi3) that otherwise reason
    aloud and never emit a bare letter within the token budget. Ordering of
    `demonstrations` is the independent variable in the ordering sweep, so this
    preserves the given order exactly and performs no sorting of its own.
    """
    parts = [serialize_demonstration(d, include_answer=True) for d in demonstrations]
    parts.append(serialize_demonstration(query, include_answer=False))
    return _INSTRUCTION + "\n\n".join(parts)


# Explicit "the answer is X" style statements, highest precision.
_ANSWER_PATTERN = re.compile(
    r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=|-)?\s*\(?\s*([ABCD])(?![A-Z])"
)
# Leading letter followed by a strong delimiter: "B.", "C)", "(D):".
# Requires the delimiter, so prose like "A cat sat" does not match.
_LEADING_PATTERN = re.compile(r"^\(?\s*([ABCD])\s*[\).:\-]")


def parse_letter(raw_response: str) -> Optional[str]:
    """
    Extract the chosen option letter (A-D) from a raw model generation.

    Precision over recall by design: anything ambiguous returns None rather
    than a guess, because a wrong pseudo-label poisons the refined context, the
    same failure the None sentinel fixed in parse_label. None means: counted
    incorrect in evaluation, excluded from relabelled pools, excluded from
    server votes. Callers must handle None, never substitute a default letter.
    """
    if not raw_response:
        return None
    up = raw_response.upper()

    m = _ANSWER_PATTERN.search(up)
    if m:
        return m.group(1)

    stripped = up.strip(" \t\n.()[]{}:;'\"")
    if stripped in LETTER_TO_INDEX:      # whole response is a bare letter
        return stripped

    m = _LEADING_PATTERN.match(up)
    if m:
        return m.group(1)

    return None
    # Known extension, not built here: when parse rate on a weak model is poor,
    # a further step can match the generation against the query's option TEXT
    # (e.g. model emits "Mitochondrion" instead of "B"). That couples the parser
    # to the item, so it is deferred until the None rate justifies it.


def example_from_hf_row(row) -> MMLUExample:
    """
    Adapter from a Hugging Face `cais/mmlu` row to MMLUExample.

    VERIFY ON YOUR MACHINE. The field names below are recalled, not checked
    against a live load in this environment. The expected schema is:
        row["question"] : str
        row["choices"]  : list[str] of length 4
        row["answer"]   : int in 0..3
        row["subject"]  : str
    If the loaded schema differs (some older `hendrycks_test` mirrors used other
    keys), correct this one function and the rest of the module is unaffected.
    The load itself is typically:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all")   # or a single subject config
    Note the "dev" split holds exactly 5 exemplars per subject, which is the
    natural seed pool for demonstrations.
    """
    return MMLUExample(
        question=row["question"],
        choices=list(row["choices"]),
        answer_index=int(row["answer"]),
        subject=row["subject"],
    )