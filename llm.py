"""
Fed-ICL Replication — LLM Module (4-class)
=================================================
Handles all interactions with Ollama.
Updated for 4-class news topic classification.
"""

import requests
import re
import time
from config import MODEL_NAME, OLLAMA_HOST, TEMPERATURE, MAX_TOKENS
from data import LABEL_SPACE


def build_icl_prompt(examples: list, query_text: str) -> str:
    prompt = (
        "Classify the following news headline into exactly one of these categories: "
        "\"world\", \"sports\", \"business\", or \"science\". "
        "Reply with only the category label, nothing else.\n\n"
    )

    for text, label in examples:
        prompt += f"Headline: \"{text}\"\nCategory: {label}\n\n"

    prompt += f"Headline: \"{query_text}\"\nCategory:"
    return prompt

_PARSE_STATS = {"total": 0, "fallback": 0, "examples": []}

def build_icl_prompt(examples: list, query_text: str) -> str:
    prompt = (
        "Classify the following news headline into exactly one of these categories: "
        "\"world\", \"sports\", \"business\", or \"science\". "
        "Reply with only the category label, nothing else.\n\n"
    )
    for text, label in examples:
        prompt += f"Headline: \"{text}\"\nCategory: {label}\n\n"
    prompt += f"Headline: \"{query_text}\"\nCategory:"
    return prompt

def query_ollama(prompt: str, model: str = None, max_retries: int = 2) -> str:
    model = model or MODEL_NAME
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                    }
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
    text = raw_response.lower().strip().strip(".,!\"'")
    tokens = [t for t in re.split(r"[\s.,!?:;]+", text) if t]

    # Strictest: the very first token IS a label.
    if tokens and tokens[0] in LABEL_SPACE:
        return tokens[0]

    # Less strict: any standalone token equals a label.
    for tok in tokens:
        if tok in LABEL_SPACE:
            return tok

    # Fallback: log and return LABEL_SPACE[0]. Behaviour unchanged downstream,
    # but we now know the fallback rate per run.
    _PARSE_STATS["fallback"] += 1
    if len(_PARSE_STATS["examples"]) < 30:
        _PARSE_STATS["examples"].append(raw_response[:120])
    return LABEL_SPACE[0]

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
