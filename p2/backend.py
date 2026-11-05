"""
Inference backend: a vLLM OpenAI-compatible server, queried concurrently.

Each call returns the chosen letter, the raw text and the top log-probabilities
of the first generated token (kept for later privacy and uncertainty analyses).
Requests carry their own seed, so a run at temperature 0.1 is reproducible.
"""

import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests

from mmlu_data import parse_letter

LETTERS = ["A", "B", "C", "D"]


def request_seed(*parts) -> int:
    """Stable per-request seed from (run seed, arm, round, client, index)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


class VLLMBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        # vLLM versions name constrained decoding differently. Try the newer
        # field first, fall back to the older one, then to unconstrained.
        self.constrain_mode = "structured_outputs" if cfg.constrain else "none"
        self.stats = {"calls": 0, "unparseable": 0, "errors": 0}

    def _payload(self, messages, seed):
        p = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "seed": seed,
            "logprobs": True,
            "top_logprobs": self.cfg.top_logprobs,
        }
        if self.constrain_mode == "structured_outputs":
            p["structured_outputs"] = {"choice": LETTERS}
        elif self.constrain_mode == "guided_choice":
            p["guided_choice"] = LETTERS
        return p

    def _post(self, messages, seed):
        for attempt in range(4):
            try:
                r = requests.post(self.url, json=self._payload(messages, seed), timeout=180)
                if r.status_code == 400:
                    body = r.text.lower()
                    field_problem = any(w in body for w in ("structured_outputs", "guided", "extra inputs", "extra_forbidden"))
                    if field_problem and self.constrain_mode != "none":
                        # Unsupported constraint field on this vLLM version: step down once.
                        self.constrain_mode = {"structured_outputs": "guided_choice",
                                               "guided_choice": "none"}[self.constrain_mode]
                        continue
                    # Any other 400 (e.g. prompt longer than the context window):
                    # record it and give up on this request, never retry blindly.
                    self.stats["errors"] += 1
                    if "context" in body or "maximum" in body or "too long" in body:
                        self.stats["too_long"] = self.stats.get("too_long", 0) + 1
                    return None
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError:
                if attempt == 3:
                    raise
                time.sleep(5)
            except requests.exceptions.RequestException:
                if attempt == 3:
                    self.stats["errors"] += 1
                    return None
                time.sleep(2)
        return None

    def classify(self, messages, seed):
        data = self._post(messages, seed)
        self.stats["calls"] += 1
        raw, top = "", {}
        if data:
            choice = data["choices"][0]
            raw = (choice.get("message") or {}).get("content") or ""
            lp = (choice.get("logprobs") or {}).get("content") or []
            if lp:
                top = {t["token"]: t["logprob"] for t in lp[0].get("top_logprobs", [])}
        letter = parse_letter(raw)
        if letter is None:
            self.stats["unparseable"] += 1
        return {"letter": letter, "raw": raw[:40], "top_logprobs": top}

    def classify_many(self, message_lists, seeds):
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(self.classify, message_lists, seeds))

    def health(self):
        r = requests.get(self.cfg.base_url.rstrip("/") + "/models", timeout=10)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
