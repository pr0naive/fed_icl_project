"""
Chat-format few-shot prompts for MMLU.

Why chat turns: the old prompt put every demonstration and the query in one user
message, so the instruct model read the demonstrations as questions to answer
and replied "Here are my answers: D B D". 25-32% of outputs were unparseable.
Here each demonstration is a past turn: the user asks it, the assistant answers
with a single letter. The final user turn is the query.
"""

from mmlu_data import INDEX_TO_LETTER

SYSTEM = (
    "Answer the following multiple-choice question. Respond with only the "
    "single letter (A, B, C, or D) of the correct option, and nothing else."
)


def question_block(ex) -> str:
    options = "\n".join(f"{INDEX_TO_LETTER[i]}. {c}" for i, c in enumerate(ex.choices))
    return f"Question: {ex.question}\n{options}\nAnswer:"


def build_messages(demos, query) -> list:
    """demos carry their (true or pseudo) answer in answer_index; query's answer is never shown."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for d in demos:
        msgs.append({"role": "user", "content": question_block(d)})
        msgs.append({"role": "assistant", "content": d.answer_letter})
    msgs.append({"role": "user", "content": question_block(query)})
    return msgs
