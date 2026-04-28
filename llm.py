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


def query_ollama(prompt: str, model: str = None) -> str:
    model = model or MODEL_NAME
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
        print(f"          Make sure Ollama is running: ollama serve")
        raise
    except Exception as e:
        print(f"  [ERROR] Ollama query failed: {e}")
        return ""


def parse_label(raw_response: str) -> str:
    text = raw_response.lower().strip().strip(".,!\"'")

    for label in LABEL_SPACE:
        if label in text:
            return label

    first_word = re.split(r"[\s.,!]", text)[0]
    for label in LABEL_SPACE:
        if first_word == label:
            return label

    # Fallback: return most common label
    return LABEL_SPACE[0]


def predict_with_icl(examples: list, query_text: str, model: str = None) -> str:
    prompt = build_icl_prompt(examples, query_text)
    raw = query_ollama(prompt, model=model)
    label = parse_label(raw)
    return label


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
