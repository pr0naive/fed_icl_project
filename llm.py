"""
Fed-ICL Replication — LLM Module (4-class)
=================================================
Handles all interactions with Ollama.
Updated for 4-class news topic classification.
"""

import requests
import re
import time
from config import MODEL_NAME, OLLAMA_HOST, TEMPERATURE, MAX_TOKENS, MMLU_MAX_TOKENS, DATASET
from data import LABEL_SPACE
from mmlu_data import build_prompt as _mmlu_build_prompt, parse_letter as _mmlu_parse_letter

_PARSE_STATS = {"total": 0, "fallback": 0, "examples": []}

def build_icl_prompt(examples: list, query_text) -> str:
    if DATASET == "mmlu":
        # examples are MMLUExample demonstrations; query_text is the query
        # MMLUExample (name kept for signature stability). build_prompt renders
        # each with its own four options and leaves the query answer open.
        return _mmlu_build_prompt(examples, query_text)
    labels = ", ".join(f'"{l}"' for l in LABEL_SPACE)
    noun  = "text" if DATASET == "dbpedia" else "news headline"
    field = "Text" if DATASET == "dbpedia" else "Headline"
    prompt = (
        f"Classify the following {noun} into exactly one of these categories: {labels}. "
        "Reply with only the category label, nothing else.\n\n"
    )
    for text, label in examples:
        prompt += f'{field}: "{text}"\nCategory: {label}\n\n'
    prompt += f'{field}: "{query_text}"\nCategory:'
    return prompt

def query_ollama(prompt: str, model: str = None, max_retries: int = 2) -> str:
    model = model or MODEL_NAME
    options = {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
    if DATASET == "mmlu":
        # Weak models reason before committing; give room to reach the letter,
        # and stop if they roll into a fresh "Question:" block (pattern echo),
        # since nothing after that is an answer to the current query.
        options["num_predict"] = MMLU_MAX_TOKENS
        options["stop"] = ["\nQuestion:"]
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"  [ERROR] Cannot connect to Ollama at {OLLAMA_HOST}.")
            raise
        except Exception as e:
            if attempt < max_retries:
                print(f"  [WARN] Ollama attempt {attempt+1} failed: {e}; retrying in 3s...")
                time.sleep(3)
            else:
                print(f"  [ERROR] Ollama query failed after {max_retries+1} attempts: {e}")
    return ""

def parse_label(raw_response: str) -> str:
    _PARSE_STATS["total"] += 1

    if DATASET == "mmlu":
        result = _mmlu_parse_letter(raw_response)
    else:
        text = raw_response.lower().strip().strip(".,!\"'")
        tokens = [t for t in re.split(r"[\s.,!?:;]+", text) if t]
        result = None
        # Strictest: the very first token IS a label.
        if tokens and tokens[0] in LABEL_SPACE:
            result = tokens[0]
        else:
            # Less strict: any standalone token equals a label.
            for tok in tokens:
                if tok in LABEL_SPACE:
                    result = tok
                    break

    # Fallback: None (sentinel). Returning a default label silently injected
    # wrong labels into relabelled_data. Callers must treat None as: incorrect
    # in evaluation, excluded from relabelled pools, excluded from server votes.
    if result is None:
        _PARSE_STATS["fallback"] += 1
        if len(_PARSE_STATS["examples"]) < 30:
            _PARSE_STATS["examples"].append(raw_response[:120])
    return result

def get_parse_stats():
    total = _PARSE_STATS["total"]
    fb    = _PARSE_STATS["fallback"]
    return {
        "total_calls": total,
        "fallback_count": fb,
        "fallback_rate": (fb / total) if total else 0,
        "sample_unparseable": list(_PARSE_STATS["examples"]),
    }

def predict_with_icl(examples: list, query_text: str, model: str = None) -> str:
    prompt = build_icl_prompt(examples, query_text)
    raw = query_ollama(prompt, model=model)
    return parse_label(raw)


def check_ollama_ready() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(MODEL_NAME in m for m in models)
        if not available:
            print(f"  [WARNING] Model '{MODEL_NAME}' not found in Ollama.")
            print(f"  Available models: {models}")
            print(f"  Run: ollama pull {MODEL_NAME}")
            return False
        print(f"  Ollama ready. Using model: {MODEL_NAME}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Ollama not running at {OLLAMA_HOST}")
        print(f"  Start it with: ollama serve")
        return False


if __name__ == "__main__":
    if check_ollama_ready():
        examples = [
            ("The central bank raised interest rates for the third time.", "business"),
            ("A world record was broken in the hundred metres.", "sports"),
            ("Researchers discovered a new species of deep-sea fish.", "science"),
        ]
        result = predict_with_icl(examples, "The president met with foreign diplomats to discuss trade.")
        print(f"Prediction: {result}")