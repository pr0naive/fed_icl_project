"""
Fed-ICL protocol and the P1 comparison arms, with full per-round logging.

Arms (all scored on the same 114 server queries):
  zeroshot     one model, no demonstrations
  gt           Fed-ICL-GT: each client answers with its C nearest TRUE-labelled
               local examples, one round, majority vote
  ub           Fed-ICL-UB: all client data merged into one client, true labels,
               one round
  fedicl       Fed-ICL (Algorithm 1): iterative relabelling, conditions on D_i and D_i^k
  fedicl_free  Fed-ICL-Free: conditions on D_i^k only (client labels never used)
  solo_free    OUR CONTROL: Fed-ICL-Free run by ONE client holding all client
               questions. No federation, no labels, same iterative loop. If it
               matches Fed-ICL, the gain is not coming from federation.

Everything a client sends the server is logged per round (needed later for the
privacy and poisoning pilots), plus pseudo-label accuracy of each client's
relabelled pool (gap W8: the paper never reports it).
"""

import dataclasses
from collections import Counter

import numpy as np

from mmlu_data import LETTER_TO_INDEX
from p2.backend import request_seed
from p2.prompts import build_messages

LETTERS = ["A", "B", "C", "D"]


def relabel(item, letter):
    return dataclasses.replace(item, answer_index=LETTER_TO_INDEX[letter])


class Runner:
    def __init__(self, cfg, backend, retriever):
        self.cfg, self.be, self.rt = cfg, backend, retriever
        self.trace = []   # one record per LLM call: who, when, what was answered

    # ---- batched prediction ------------------------------------------------
    def _predict(self, jobs, arm, rnd, client, phase):
        """jobs: list of (demos, query). Returns letters (None if unparseable)."""
        msgs = [build_messages(d, q) for d, q in jobs]
        seeds = [request_seed(self.cfg.seed, arm, rnd, client, phase, i) for i in range(len(jobs))]
        outs = self.be.classify_many(msgs, seeds)
        for (demos, q), o in zip(jobs, outs):
            self.trace.append({
                "arm": arm, "round": rnd, "client": client, "phase": phase,
                "subject": q.subject, "q": q.question[:80],
                "true": q.answer_letter, "pred": o["letter"],
                "n_demos": len(demos), "top_logprobs": o["top_logprobs"],
            })
        return [o["letter"] for o in outs]

    def _vote(self, per_client, previous, rng):
        out, ties = [], 0
        for qi in range(len(previous)):
            votes = [p[qi] for p in per_client if p[qi] is not None]
            if not votes:
                out.append(previous[qi])
                continue
            counts = Counter(votes)
            top = max(counts.values())
            tied = sorted(l for l, c in counts.items() if c == top)
            if len(tied) > 1:
                ties += 1
                out.append(str(rng.choice(tied)))
            else:
                out.append(tied[0])
        return out, ties

    @staticmethod
    def _acc(labels, queries):
        return sum(l == q.answer_letter for l, q in zip(labels, queries)) / len(queries)

    # ---- arms ----------------------------------------------------------------
    def zeroshot(self, queries):
        preds = self._predict([([], q) for q in queries], "zeroshot", 0, -1, "query")
        return {"final_labels": preds, "accuracy": self._acc(preds, queries)}

    def gt(self, queries, clients, arm="gt"):
        S = self.cfg.num_shots
        if self.cfg.use_filter:
            clients = [self.rt.filter_pool(c, queries, S) for c in clients]
        per_client = [
            self._predict([(self.rt.knn(q, c, S), q) for q in queries], arm, 1, i, "query")
            for i, c in enumerate(clients)
        ]
        rng = np.random.default_rng(self.cfg.seed + 1000)
        labels, ties = self._vote(per_client, [None] * len(queries), rng)
        return {"final_labels": labels, "accuracy": self._acc(labels, queries),
                "client_preds": per_client, "ties": ties,
                "client_acc": [self._acc(p, queries) for p in per_client]}

    def ub(self, queries, clients):
        merged = [it for c in clients for it in c]
        return self.gt(queries, [merged], arm="ub")

    def loop(self, queries, clients, variant, arm):
        cfg, S = self.cfg, self.cfg.num_shots
        rng = np.random.default_rng(cfg.seed + 1000)          # tie-breaks
        init_rng = np.random.default_rng(cfg.seed + 2000)     # random initial labels
        if cfg.use_filter:
            clients = [self.rt.filter_pool(c, queries, S) for c in clients]

        if cfg.init == "zeroshot":
            labels = self.zeroshot(queries)["final_labels"]
            labels = [l if l is not None else str(init_rng.choice(LETTERS)) for l in labels]
        else:
            labels = [str(init_rng.choice(LETTERS)) for _ in queries]

        rounds = [{"round": 0, "accuracy": self._acc(labels, queries), "labels": labels}]
        for k in range(1, cfg.num_rounds + 1):
            context = [relabel(q, l) for q, l in zip(queries, labels)]
            per_client, pseudo_acc, pool_sizes = [], [], []
            for i, local in enumerate(clients):
                # Step 1 (line 5): relabel local questions using the server context.
                rl = self._predict([(self.rt.knn(x, context, S), x) for x in local],
                                   arm, k, i, "relabel")
                relabelled = [relabel(x, l) for x, l in zip(local, rl) if l is not None]
                pseudo_acc.append(
                    sum(l == x.answer_letter for x, l in zip(local, rl)) / max(len(local), 1))
                pool_sizes.append(len(relabelled))
                # Step 2 (line 7): answer the server queries.
                jobs = []
                for q in queries:
                    if variant == "free":
                        demos = self.rt.knn(q, relabelled, S)
                    elif cfg.prompt_pool == "both":
                        demos = self.rt.knn(q, local, S) + self.rt.knn(q, relabelled, S)
                    else:
                        demos = self.rt.knn(q, local + relabelled, S)
                    jobs.append((demos, q))
                per_client.append(self._predict(jobs, arm, k, i, "query"))
            labels, ties = self._vote(per_client, labels, rng)
            rounds.append({
                "round": k,
                "accuracy": self._acc(labels, queries),
                "labels": labels,
                "client_preds": per_client,               # what each client sent the server
                "client_acc": [self._acc(p, queries) for p in per_client],
                "pseudo_label_acc": pseudo_acc,           # W8
                "relabelled_pool_sizes": pool_sizes,
                "ties": ties,
            })
            print(f"  [{arm}] round {k}: {rounds[-1]['accuracy']:.3f} "
                  f"(clients {[round(a, 3) for a in rounds[-1]['client_acc']]}, ties {ties})",
                  flush=True)
        return {"final_labels": labels, "accuracy": rounds[-1]["accuracy"], "rounds": rounds,
                "filtered_pool_sizes": [len(c) for c in clients]}

    def heldout(self, final_labels, queries, eval_set, arm):
        """OUR addition: score eval items using the final server context as demonstrations."""
        if not eval_set:
            return None
        context = [relabel(q, l) for q, l in zip(queries, final_labels) if l is not None]
        S = self.cfg.num_shots
        preds = self._predict([(self.rt.knn(x, context, S), x) for x in eval_set],
                              arm, 99, -1, "heldout")
        return self._acc(preds, eval_set)

    def run_arm(self, arm, queries, clients, eval_set):
        if arm == "zeroshot":
            out = self.zeroshot(queries)
        elif arm == "gt":
            out = self.gt(queries, clients)
        elif arm == "ub":
            out = self.ub(queries, clients)
        elif arm == "fedicl":
            out = self.loop(queries, clients, "full", arm)
        elif arm == "fedicl_free":
            out = self.loop(queries, clients, "free", arm)
        elif arm == "solo_free":
            out = self.loop(queries, [[it for c in clients for it in c]], "free", arm)
        else:
            raise ValueError(f"unknown arm {arm}")
        out["correct"] = int(round(out["accuracy"] * len(queries)))
        out["per_query_correct"] = [l == q.answer_letter for l, q in zip(out["final_labels"], queries)]
        if arm != "zeroshot":
            out["heldout_accuracy"] = self.heldout(out["final_labels"], queries, eval_set, arm)
        return out
