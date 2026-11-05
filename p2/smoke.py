"""
Ten-second check that the live vLLM server behaves before a real run:
model name, constrained-answer mode, one easy question, one long 10-shot prompt.

    python -m p2.smoke
"""

from mmlu_data import MMLUExample
from p2.backend import VLLMBackend
from p2.config import Config
from p2.prompts import build_messages


def main():
    cfg = Config()
    be = VLLMBackend(cfg)
    print("Models on server:", be.health())

    q = MMLUExample("What is 2 + 2?", ["3", "4", "5", "6"], 1, "elementary_mathematics")
    out = be.classify(build_messages([], q), seed=1)
    print("Easy question ->", out["letter"], "| raw:", repr(out["raw"]), "| constrain mode:", be.constrain_mode)
    print("Top log-probs:", out["top_logprobs"])

    long_text = "This is a long scenario sentence for testing the context window. " * 60
    demos = [MMLUExample(long_text + f" Case {i}?", ["w", "x", "y", "z"], i % 4, "professional_law")
             for i in range(10)]
    out = be.classify(build_messages(demos, q), seed=2)
    print("Long 10-shot prompt ->", out["letter"], "| stats:", be.stats)

    ok = out["letter"] in {"A", "B", "C", "D"} and be.stats["errors"] == 0
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED: paste this output to Claude")


if __name__ == "__main__":
    main()
