"""
MMLU data for Wang et al.'s setting.

Server queries: 2 per subject (57 subjects -> 114), drawn with a fixed QUERY_SEED
so every arm and every seed is scored on the same questions (needed for paired
tests). Client data: the remaining test questions, partitioned across clients by
a Dirichlet over subjects with concentration alpha (Hsu et al., 2019), as in the
paper. The partition depends on SEED, so seeds vary the clients, not the queries.
"""

from collections import defaultdict

import numpy as np


def load_mmlu_test():
    from datasets import load_dataset
    from mmlu_data import example_from_hf_row
    ds = load_dataset("cais/mmlu", "all", split="test")
    return [example_from_hf_row(r) for r in ds]


def stratified(items, per_subject, seed, exclude=frozenset()):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, it in enumerate(items):
        if i not in exclude:
            by[it.subject].append(i)
    out = []
    for s in sorted(by):
        k = min(per_subject, len(by[s]))
        out.extend(int(i) for i in rng.choice(by[s], size=k, replace=False))
    return out


def stratified_total(items, total, seed, exclude=frozenset()):
    """About `total` indices spread evenly over subjects (for the optional eval set)."""
    subjects = sorted({it.subject for it in items})
    per, rem = divmod(total, len(subjects))
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, it in enumerate(items):
        if i not in exclude:
            by[it.subject].append(i)
    out = []
    for j, s in enumerate(subjects):
        k = min(per + (1 if j < rem else 0), len(by[s]))
        out.extend(int(i) for i in rng.choice(by[s], size=k, replace=False))
    return out


def dirichlet_by_subject(items, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for it in items:
        by[it.subject].append(it)
    clients = [[] for _ in range(num_clients)]
    for s in sorted(by):
        group = list(by[s])
        rng.shuffle(group)
        props = rng.dirichlet([alpha] * num_clients)
        if not np.all(np.isfinite(props)):
            # alpha -> 0 underflow: the limit is the whole subject on one client.
            props = np.zeros(num_clients)
            props[rng.integers(num_clients)] = 1.0
        bounds = [0] + (np.cumsum(props) * len(group)).round().astype(int).tolist()
        bounds[-1] = len(group)
        for k in range(num_clients):
            clients[k].extend(group[bounds[k]:bounds[k + 1]])
    for c in clients:
        rng.shuffle(c)
    return clients


def mean_pairwise_jsd(clients, subjects):
    idx = {s: i for i, s in enumerate(subjects)}
    dists = []
    for c in clients:
        v = np.zeros(len(subjects))
        for it in c:
            v[idx[it.subject]] += 1
        dists.append(v / v.sum() if v.sum() else v)

    def kl(a, b):
        m = a > 0
        return float(np.sum(a[m] * np.log2(a[m] / b[m])))

    vals = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            m = 0.5 * (dists[i] + dists[j])
            vals.append(0.5 * kl(dists[i], m) + 0.5 * kl(dists[j], m))
    return float(np.mean(vals)) if vals else 0.0


def prepare(cfg, items=None):
    items = items if items is not None else load_mmlu_test()
    qseed = cfg.query_seed + cfg.seed if cfg.query_mode == "per_seed" else cfg.query_seed
    q_idx = stratified(items, cfg.queries_per_subject, qseed)
    used = set(q_idx)
    e_idx = []
    if cfg.eval_size > 0:
        e_idx = stratified_total(items, cfg.eval_size, cfg.eval_seed, exclude=frozenset(used))
        used |= set(e_idx)
    pool = [it for i, it in enumerate(items) if i not in used]
    clients = dirichlet_by_subject(pool, cfg.num_clients, cfg.alpha, cfg.seed)
    subjects = sorted({it.subject for it in items})
    info = {
        "n_items": len(items),
        "n_queries": len(q_idx),
        "query_seed_used": qseed,
        "n_eval": len(e_idx),
        "n_pool": len(pool),
        "client_sizes": [len(c) for c in clients],
        "client_subjects": [len({it.subject for it in c}) for c in clients],
        "heterogeneity_jsd": mean_pairwise_jsd(clients, subjects),
    }
    return [items[i] for i in q_idx], clients, [items[i] for i in e_idx], info
