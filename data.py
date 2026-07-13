"""
Fed-ICL Replication — Data Module (v2 - Harder Task)
=====================================================
"""

import numpy as np
from datasets import load_dataset
from numpy.random import seed
from config import (SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES, EVAL_SIZE, CLIENT_POOL_SIZE, EVAL_SEED)

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
                    
RAW_DATA = _load_ag_news(num_examples=NUM_SERVER_QUERIES + CLIENT_POOL_SIZE, seed=SEED)

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


def prepare_experiment():
    data = RAW_DATA.copy()
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
