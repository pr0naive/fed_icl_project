"""
Tests for mmlu_data. Pure logic only: no network, no model, no HF load.
Run: python -m pytest test_mmlu_data.py -q   (or just: python test_mmlu_data.py)
"""

import mmlu_data as m


def make_example():
    return m.MMLUExample(
        question="What is the powerhouse of the cell?",
        choices=["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus"],
        answer_index=1,
        subject="high_school_biology",
    )


def test_letter_mapping_is_consistent():
    # The two maps must be exact inverses, and NUM_OPTIONS must agree.
    assert m.NUM_OPTIONS == 4
    for i, letter in m.INDEX_TO_LETTER.items():
        assert m.LETTER_TO_INDEX[letter] == i
    assert set(m.LETTER_TO_INDEX) == {"A", "B", "C", "D"}


def test_answer_letter_property():
    assert make_example().answer_letter == "B"


def test_demonstration_shows_answer():
    ex = make_example()
    demo = m.serialize_demonstration(ex, include_answer=True)
    assert demo.splitlines() == [
        "Question: What is the powerhouse of the cell?",
        "A. Nucleus",
        "B. Mitochondrion",
        "C. Ribosome",
        "D. Golgi apparatus",
        "Answer: B",
    ]


def test_query_withholds_answer():
    ex = make_example()
    query = m.serialize_demonstration(ex, include_answer=False)
    assert query.endswith("Answer:")
    assert " B" not in query.split("Answer:")[1]  # nothing after the colon


def test_embedding_text_default_is_stem_only():
    ex = make_example()
    assert m.embedding_text(ex) == "What is the powerhouse of the cell?"


def test_embedding_text_with_options_appends_them():
    ex = make_example()
    txt = m.embedding_text(ex, include_options=True)
    assert txt.startswith("What is the powerhouse of the cell?")
    assert "B. Mitochondrion" in txt


def test_build_prompt_preserves_demonstration_order():
    ex = make_example()
    other = m.MMLUExample(
        question="Which gas do plants absorb?",
        choices=["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"],
        answer_index=2,
        subject="high_school_biology",
    )
    prompt = m.build_prompt([ex, other], query=ex)
    # First demonstration block must appear before the second.
    assert prompt.index("powerhouse") < prompt.index("plants absorb")
    # Query block (answer withheld) sits last and ends open.
    assert prompt.rstrip().endswith("Answer:")


def test_reversed_order_changes_the_prompt():
    # Guards the ordering sweep: order must actually affect the serialized text.
    ex = make_example()
    other = m.MMLUExample(
        question="Which gas do plants absorb?",
        choices=["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"],
        answer_index=2,
        subject="high_school_biology",
    )
    assert m.build_prompt([ex, other], ex) != m.build_prompt([other, ex], ex)


def test_wrong_number_of_choices_rejected():
    try:
        m.MMLUExample("q", ["A", "B", "C"], 0, "subj")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "expected 4 choices" in str(e)


def test_out_of_range_answer_rejected():
    try:
        m.MMLUExample("q", ["A", "B", "C", "D"], 4, "subj")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "out of range" in str(e)


def test_hf_adapter_builds_example():
    row = {
        "question": "2 + 2 = ?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "subject": "elementary_mathematics",
    }
    ex = m.example_from_hf_row(row)
    assert ex.answer_letter == "B"
    assert ex.subject == "elementary_mathematics"


def test_parse_letter_plain_and_wrapped():
    for raw, want in [("B", "B"), ("b", "B"), ("B.", "B"), ("(C)", "C"), ("D)", "D")]:
        assert m.parse_letter(raw) == want, raw


def test_parse_letter_answer_statements():
    assert m.parse_letter("The answer is B.") == "B"
    assert m.parse_letter("Answer: C") == "C"
    assert m.parse_letter("I think the correct choice is D") == "D"
    assert m.parse_letter("answer: a") == "A"


def test_parse_letter_leading_letter_with_delimiter():
    assert m.parse_letter("B. Mitochondrion") == "B"


def test_parse_letter_returns_none_on_ambiguous():
    # Precision over recall: no guessing.
    assert m.parse_letter("A cat sat on the mat") is None   # article trap
    assert m.parse_letter("Mitochondrion") is None          # option text, no letter
    assert m.parse_letter("") is None
    assert m.parse_letter("maybe C or D") is None            # two candidates, no anchor


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} tests passed")