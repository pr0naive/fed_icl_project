"""
Fed-ICL Replication — Data Module (v2 - Harder Task)
=====================================================
4-class news topic classification (AG News style).
This is much harder than binary sentiment — llama3 won't get 100% easily,
so you'll see real differences between baselines and Fed-ICL.
"""

import numpy as np
from datasets import load_dataset
from numpy.random import seed
from config import SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES, EVAL_SIZE

np.random.seed(SEED)

LABEL_SPACE = ["world", "sports", "business", "science"]
_AG_NEWS_LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "science"}




"""Load AG News from HuggingFace and subsample deterministically.
AG News has 120k training examples — far more than ICL needs.
We take a random subset for comparable scale to the synthetic data."""
def _load_ag_news(num_examples, seed):
    ds = load_dataset("ag_news", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=num_examples, replace=False)
    return [(ds[int(i)]["text"], _AG_NEWS_LABEL_MAP[ds[int(i)]["label"]])
for i in indices]
                    
RAW_DATA = _load_ag_news(num_examples=200, seed=SEED)


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
        # Ensure at least 1 example per client per class when possible
        for k in range(num_clients):
            if counts[k] == 0 and len(class_indices) > num_clients:
                counts[k] = 1
        # Adjust first client to match total
        counts[0] += len(class_indices) - counts.sum()
        counts[0] = max(counts[0], 0)

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

    eval_set = data[:EVAL_SIZE]
    server_queries = data[EVAL_SIZE:EVAL_SIZE + NUM_SERVER_QUERIES]
    client_pool = data[EVAL_SIZE + NUM_SERVER_QUERIES:]

    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    print("=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Total examples:    {len(RAW_DATA)}")
    print(f"  Label space:       {LABEL_SPACE}")
    print(f"  Evaluation set:    {len(eval_set)}")
    print(f"  Server queries:    {len(server_queries)}")
    print(f"  Client pool:       {len(client_pool)}")
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
