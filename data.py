"""
Fed-ICL Replication — Data Module (v2 - Harder Task)
=====================================================
"""

import numpy as np
from datasets import load_dataset
from numpy.random import seed
from config import (SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES, EVAL_SIZE, CLIENT_POOL_SIZE, EVAL_SEED,
                    DATA_REGIME, SPLIT_SEED, TEST_FRACTION, QUERY_SEED)

np.random.seed(SEED)

LABEL_SPACE = ["world", "sports", "business", "science"]
_AG_NEWS_LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "science"}




"""Draw a uniform random sample from the full canonical AG News training
split (120,000 examples), which is the sampling frame for client pools and
server queries. Evaluation is drawn separately from the canonical test
split (7,600 examples); see _load_ag_news_test."""
def _load_ag_news(num_examples, seed):
    ds = load_dataset("fancyzhx/ag_news", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=num_examples, replace=False)
    return [(ds[int(i)]["text"], _AG_NEWS_LABEL_MAP[ds[int(i)]["label"]])
for i in indices]
                    
RAW_DATA = None if DATA_REGIME == "split8020" else _load_ag_news(
    num_examples=NUM_SERVER_QUERIES + CLIENT_POOL_SIZE, seed=SEED)

def _load_ag_news_test(num_examples, seed=EVAL_SEED):
    """Deterministic, class-balanced eval sample from the AG News TEST split.

    Uses EVAL_SEED (fixed) rather than the partition SEED, so the eval set is
    IDENTICAL across partition seeds. Class-balanced sampling makes per-class
    accuracy comparable across runs and removes sampling skew as a confound.
    """
    ds = load_dataset("fancyzhx/ag_news", split="test")
    rng = np.random.default_rng(seed)
    labels = np.array(ds["label"])
    num_classes = len(LABEL_SPACE)
    per_class = num_examples // num_classes
    remainder = num_examples - per_class * num_classes

    indices = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        take = per_class + (1 if c < remainder else 0)
        chosen = rng.choice(cls_idx, size=take, replace=False)
        indices.extend(int(i) for i in chosen)
    rng.shuffle(indices)
    return [(ds[i]["text"], _AG_NEWS_LABEL_MAP[ds[i]["label"]]) for i in indices]

def get_label_id(label: str) -> int:
    return LABEL_SPACE.index(label)

def partition_data_dirichlet(data: list, num_clients: int, alpha: float):
    labels = np.array([get_label_id(d[1]) for d in data])
    num_classes = len(LABEL_SPACE)
    client_data = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(class_indices)).astype(int)

        if len(class_indices) >= num_clients:
            for k in range(num_clients):
                if counts[k] == 0:
                    counts[k] = 1

        # Change 2b: distribute residual round-robin from a random client,
        # not all on client 0 (which biased the local-only baseline upward).
        deficit = len(class_indices) - int(counts.sum())
        if deficit > 0:
            start = np.random.randint(num_clients)
            for offset in range(deficit):
                counts[(start + offset) % num_clients] += 1
        elif deficit < 0:
            excess = -deficit
            while excess > 0:
                biggest = int(np.argmax(counts))
                if counts[biggest] > 1:
                    counts[biggest] -= 1
                    excess -= 1
                else:
                    break

        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            for idx in class_indices[start:end]:
                client_data[k].append(data[idx])
            start = end

    for k in range(num_clients):
        np.random.shuffle(client_data[k])
    return client_data

def _materialize(ds, idxs):
    """Fast index -> (text, label) via ds.select (order preserved)."""
    sub = ds.select([int(i) for i in idxs])
    return list(zip(sub["text"], [_AG_NEWS_LABEL_MAP[l] for l in sub["label"]]))


def _stratified_subset(labels, candidate_idx, n, num_classes, rng):
    """Class-balanced draw of n indices restricted to candidate_idx."""
    cand = np.asarray(candidate_idx)
    cand_labels = labels[cand]
    per, rem = divmod(n, num_classes)
    chosen = []
    for c in range(num_classes):
        pool_c = cand[cand_labels == c]
        k = per + (1 if c < rem else 0)
        chosen.extend(rng.choice(pool_c, size=k, replace=False).tolist())
    rng.shuffle(chosen)
    return [int(i) for i in chosen]


def _prepare_split8020():
    """80/20 split of the 120k train split. 80% -> queries + Dirichlet pools; 20% -> eval."""
    ds = load_dataset("fancyzhx/ag_news", split="train")
    labels = np.array(ds["label"])
    n = len(ds)
    num_classes = len(LABEL_SPACE)

    perm = np.random.default_rng(SPLIT_SEED).permutation(n)     # learn/test disjoint by construction
    cut = int((1.0 - TEST_FRACTION) * n)
    learn_idx, test_idx = perm[:cut], perm[cut:]

    query_idx = _stratified_subset(labels, learn_idx, NUM_SERVER_QUERIES, num_classes,
                                   np.random.default_rng(QUERY_SEED))
    pool_idx = np.setdiff1d(learn_idx, np.array(query_idx))     # pool = learn minus queries

    server_queries = _materialize(ds, query_idx)
    client_pool    = _materialize(ds, pool_idx)

    eval_idx = _stratified_subset(labels, test_idx, EVAL_SIZE, num_classes,
                                  np.random.default_rng(EVAL_SEED))
    eval_set = _materialize(ds, eval_idx)

    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    print("=" * 60); print("DATA DISTRIBUTION SUMMARY  (regime=split8020)"); print("=" * 60)
    print(f"  120k train -> learn {len(learn_idx)} / test {len(test_idx)}  (test_frac={TEST_FRACTION})")
    print(f"  Server queries: {len(server_queries)} (stratified, QUERY_SEED={QUERY_SEED})")
    print(f"  Client pool:    {len(client_pool)} (learn minus queries)")
    print(f"  Eval:           {len(eval_set)} (stratified from the 20%, EVAL_SEED={EVAL_SEED})")
    print(f"  Dirichlet alpha:{DIRICHLET_ALPHA}   partition SEED={SEED}")
    for k, cd in enumerate(client_datasets):
        counts = {l: sum(1 for _, lab in cd if lab == l) for l in LABEL_SPACE}
        print(f"  Client {k}: {len(cd)} examples (" + ", ".join(f"{l}={counts[l]}" for l in LABEL_SPACE) + ")")
    print("=" * 60)
    return server_queries, client_datasets, eval_set


def prepare_experiment():
    if DATA_REGIME == "split8020":
        return _prepare_split8020()
    data = RAW_DATA.copy()
    # ... existing canonical body unchanged ...
    np.random.shuffle(data)

# eval_set: AG News TEST split, fixed EVAL_SEED, class-balanced.
    eval_set = _load_ag_news_test(EVAL_SIZE, EVAL_SEED)
    server_queries = data[:NUM_SERVER_QUERIES]
    client_pool = data[NUM_SERVER_QUERIES:]

    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    print("=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Total train:    {len(RAW_DATA)}")
    print(f"  Sampling frame:    train split n=120000, test split n=7600 (canonical)")
    print(f"  Drawn this run:    pool={len(client_pool)}, queries={len(server_queries)}, eval={len(eval_set)}")
    print(f"  Label space:       {LABEL_SPACE}")
    print(f"  Evaluation set (test):    {len(eval_set)} drawn from ag_news split='test')")
    print(f"  Server queries:    {len(server_queries)}")
    print(f"  Client pool:       {len(client_pool)}")
    print(f"  (configured pool:  {CLIENT_POOL_SIZE})")
    print(f"  Dirichlet alpha:   {DIRICHLET_ALPHA}")
    print()

    for k, cd in enumerate(client_datasets):
        counts = {l: sum(1 for _, lab in cd if lab == l) for l in LABEL_SPACE}
        total = len(cd)
        dist = ", ".join(f"{l}={counts[l]}" for l in LABEL_SPACE)
        print(f"  Client {k}: {total} examples ({dist})")

    print("=" * 60)
    return server_queries, client_datasets, eval_set


if __name__ == "__main__":
    sq, cd, ev = prepare_experiment()
    print(f"\nSample server query: {sq[0]}")
    print(f"Sample eval item:   {ev[0]}")
