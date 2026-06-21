"""
Fed-ICL Replication, Data Module
Supports AG News (default) or DBpedia, switchable via FED_ICL_DATASET env var.
"""

import numpy as np
from datasets import load_dataset
from config import (
    SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES,
    EVAL_SIZE, CLIENT_POOL_SIZE, DATASET,
)

np.random.seed(SEED)

if DATASET == "agnews":
    LABEL_SPACE = ["world", "sports", "business", "science"]
    _LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "science"}
    _HF_PATH = "fancyzhx/ag_news"
    _TEXT_FIELD = "text"
elif DATASET == "dbpedia":
    LABEL_SPACE = [
        "company", "educational_institution", "artist", "athlete",
        "office_holder", "transportation", "building", "natural_place",
        "village", "animal", "plant", "album", "film", "written_work",
    ]
    _LABEL_MAP = {i: name for i, name in enumerate(LABEL_SPACE)}
    _HF_PATH = "dbpedia_14"
    _TEXT_FIELD = "content"
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")


def _load_split(split_name, num_examples, seed):
    ds = load_dataset(_HF_PATH, split=split_name)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=num_examples, replace=False)
    return [(ds[int(i)][_TEXT_FIELD], _LABEL_MAP[ds[int(i)]["label"]])
            for i in indices]


RAW_DATA = _load_split("train", num_examples=NUM_SERVER_QUERIES + CLIENT_POOL_SIZE, seed=SEED)


def get_held_out(num_examples=EVAL_SIZE, seed=SEED):
    return _load_split("test", num_examples=num_examples, seed=seed + 1)


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

        # Distribute residual round-robin from a random client,
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

    eval_set = get_held_out()
    server_queries = data[:NUM_SERVER_QUERIES]
    client_pool = data[NUM_SERVER_QUERIES:]

    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    print("=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Dataset:           {DATASET} ({_HF_PATH})")
    print(f"  Total train:       {len(RAW_DATA)}")
    print(f"  Label space:       {LABEL_SPACE}")
    print(f"  Evaluation set:    {len(eval_set)} drawn from {_HF_PATH} split='test'")
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
    print(f"Sample eval item:    {ev[0]}")