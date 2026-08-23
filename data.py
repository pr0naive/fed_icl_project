"""
Fed-ICL Replication — Data Module (v2 - Harder Task)
=====================================================
"""

import numpy as np
from collections import defaultdict
from datasets import load_dataset
from numpy.random import seed
from config import (SEED, DIRICHLET_ALPHA, NUM_CLIENTS, NUM_SERVER_QUERIES, EVAL_SIZE, CLIENT_POOL_SIZE, EVAL_SEED,
                    DATA_REGIME, SPLIT_SEED, TEST_FRACTION, QUERY_SEED, DATASET, POOL_CAP, FILTER_LOCAL_DATA)

np.random.seed(SEED)

if DATASET == "dbpedia":
    LABEL_SPACE = ["company", "school", "artist", "athlete", "politician", "transport",
                   "building", "nature", "village", "animal", "plant", "album", "film", "book"]
    _AG_NEWS_LABEL_MAP = {i: LABEL_SPACE[i] for i in range(len(LABEL_SPACE))}
    _HF_NAME = "fancyzhx/dbpedia_14"
    _TEXT_FIELDS = ("title", "content")      # joined into one text
elif DATASET == "mmlu":
    # Task label space is the four option letters. The 57 MMLU subjects are NOT
    # the label space; they are the partition axis (see _prepare_mmlu), kept
    # separate from prediction, parsing, and aggregation.
    LABEL_SPACE = ["A", "B", "C", "D"]
    _HF_NAME = "cais/mmlu"
    _AG_NEWS_LABEL_MAP = None
    _TEXT_FIELDS = None
else:
    LABEL_SPACE = ["world", "sports", "business", "science"]
    _AG_NEWS_LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "science"}
    _HF_NAME = "fancyzhx/ag_news"
    _TEXT_FIELDS = ("text",)




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
                    
RAW_DATA = None if (DATA_REGIME in ("split8020", "canonical_full") or DATASET in ("dbpedia", "mmlu")) else _load_ag_news(
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
        if not np.all(np.isfinite(proportions)):
            # Inert at the alphas used for AG News; guards the alpha -> 0 limit
            # where dirichlet underflows to NaN. One-hot = fully skewed split.
            proportions = np.zeros(num_clients)
            proportions[np.random.randint(num_clients)] = 1.0
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
    """Index -> (text, label). Reads rows directly, so it is robust across
    datasets versions and to single- or multi-field text."""
    sub = ds.select([int(i) for i in idxs])
    out = []
    for row in sub:
        text = " ".join(str(row[f]) for f in _TEXT_FIELDS)
        out.append((text, _AG_NEWS_LABEL_MAP[row["label"]]))
    return out


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


def _prepare_canonical_full():
    """Meeting regime: the FULL 120k train is Dirichlet-partitioned across clients;
       server queries and held-out eval are both drawn from the 7.6k TEST split, disjoint."""
    train_ds = load_dataset(_HF_NAME, split="train")
    test_ds  = load_dataset(_HF_NAME, split="test")
    test_labels = np.array(test_ds["label"])
    num_classes = len(LABEL_SPACE)

    client_pool = _materialize(train_ds, range(len(train_ds)))
    if POOL_CAP > 0 and len(client_pool) > POOL_CAP:                     # scale-matched cross-dataset runs
        pl = np.array([get_label_id(lab) for _, lab in client_pool])
        keep = _stratified_subset(pl, np.arange(len(client_pool)), POOL_CAP, num_classes,
                                  np.random.default_rng(SEED))
        client_pool = [client_pool[i] for i in keep]
    client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    query_idx = _stratified_subset(test_labels, np.arange(len(test_ds)),
                                   NUM_SERVER_QUERIES, num_classes, np.random.default_rng(QUERY_SEED))
    remaining = np.setdiff1d(np.arange(len(test_ds)), np.array(query_idx))
    eval_idx  = _stratified_subset(test_labels, remaining,
                                   EVAL_SIZE, num_classes, np.random.default_rng(EVAL_SEED))
    server_queries = _materialize(test_ds, query_idx)
    eval_set       = _materialize(test_ds, eval_idx)

    print("=" * 60); print("DATA DISTRIBUTION SUMMARY  (regime=canonical_full)"); print("=" * 60)
    print(f"  Dataset: {DATASET} ({_HF_NAME}), {len(LABEL_SPACE)} classes")
    print(f"  Client pools: {len(client_pool)} of {len(train_ds)} train partitioned across {NUM_CLIENTS} clients")
    print(f"  Server queries: {len(server_queries)} from TEST (QUERY_SEED={QUERY_SEED})")
    print(f"  Held-out eval:  {len(eval_set)} from TEST (EVAL_SEED={EVAL_SEED}), disjoint from queries")
    print(f"  Dirichlet alpha: {DIRICHLET_ALPHA}   partition SEED={SEED}")
    for k, cd in enumerate(client_datasets):
        counts = {l: sum(1 for _, lab in cd if lab == l) for l in LABEL_SPACE}
        print(f"  Client {k}: {len(cd)} (" + ", ".join(f"{l}={counts[l]}" for l in LABEL_SPACE) + ")")
    print("=" * 60)
    return server_queries, client_datasets, eval_set


def partition_by_subject_dirichlet(data: list, num_clients: int, alpha: float):
    """Dirichlet partition over MMLU subjects: Wang et al.'s heterogeneity
    construction for MMLU (Hsu et al., 2019). Mirrors partition_data_dirichlet
    exactly, including the round-robin residual and the minimum-one rule, but
    uses the subject as the categorical axis instead of the class label. AG News
    and MMLU therefore share one partition algorithm on different keys.
    """
    subjects = sorted({ex.subject for ex in data})
    subj_id = {s: i for i, s in enumerate(subjects)}
    labels = np.array([subj_id[ex.subject] for ex in data])
    num_classes = len(subjects)
    client_data = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        if not np.all(np.isfinite(proportions)):
            # At very small alpha (the paper uses 0.001) the gamma draws underflow
            # and dirichlet returns NaN. The alpha -> 0 limit is a fully skewed
            # split, so assign this class entirely to one random client.
            proportions = np.zeros(num_clients)
            proportions[np.random.randint(num_clients)] = 1.0
        counts = (proportions * len(class_indices)).astype(int)

        if len(class_indices) >= num_clients:
            for k in range(num_clients):
                if counts[k] == 0:
                    counts[k] = 1

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


def _subject_stratified_indices(examples: list, n: int, seed_value: int):
    """n indices into `examples`, drawn as evenly as possible across subjects.
    Uses its own seeded RNG so query and eval draws stay decoupled from the
    partition SEED, matching the AG News eval-decoupling discipline."""
    rng = np.random.default_rng(seed_value)
    by_subject = defaultdict(list)
    for i, ex in enumerate(examples):
        by_subject[ex.subject].append(i)
    subjects = sorted(by_subject)
    per, rem = divmod(n, len(subjects))
    chosen = []
    for j, s in enumerate(subjects):
        k = per + (1 if j < rem else 0)
        pool = by_subject[s]
        k = min(k, len(pool))
        chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    rng.shuffle(chosen)
    return [int(i) for i in chosen]


def mean_pairwise_jsd(client_datasets: list, key_fn, categories: list) -> float:
    """Realized heterogeneity: mean pairwise Jensen-Shannon divergence (base 2)
    between clients' category distributions. Bounded in [0, 1] regardless of the
    number of categories, so it is comparable across AG News (4 labels) and MMLU
    (57 subjects). 0 means identical client distributions (IID); 1 means
    disjoint supports. This is the honest cross-dataset heterogeneity axis, since
    the Dirichlet alpha value is not comparable across different category counts.
    """
    cat_id = {c: i for i, c in enumerate(categories)}
    dists = []
    for cd in client_datasets:
        v = np.zeros(len(categories))
        for ex in cd:
            v[cat_id[key_fn(ex)]] += 1
        total = v.sum()
        dists.append(v / total if total > 0 else v)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    def _jsd(p, q):
        m = 0.5 * (p + q)
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    pairs = [_jsd(dists[i], dists[j])
             for i in range(len(dists)) for j in range(i + 1, len(dists))]
    return float(np.mean(pairs)) if pairs else 0.0


def _prepare_mmlu():
    """MMLU regime. Dirichlet-over-subjects client pool (paper's heterogeneity
    construction), with server queries and a decoupled held-out eval carved
    disjointly from the test split, mirroring canonical_full discipline."""
    from mmlu_data import example_from_hf_row
    ds = load_dataset(_HF_NAME, "all", split="test")
    items = [example_from_hf_row(row) for row in ds]
    subjects = sorted({ex.subject for ex in items})

    # Disjoint carve: queries first, then eval from the remainder, then pool.
    query_idx = _subject_stratified_indices(items, NUM_SERVER_QUERIES, QUERY_SEED)
    qset = set(query_idx)
    rem_idx = [i for i in range(len(items)) if i not in qset]
    rem_items = [items[i] for i in rem_idx]
    eval_rel = _subject_stratified_indices(rem_items, EVAL_SIZE, EVAL_SEED)
    eval_idx = [rem_idx[i] for i in eval_rel]
    eset = set(eval_idx)
    pool_idx = [i for i in rem_idx if i not in eset]

    server_queries = [items[i] for i in query_idx]
    eval_set = [items[i] for i in eval_idx]
    client_pool = [items[i] for i in pool_idx]
    client_datasets = partition_by_subject_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)

    het = mean_pairwise_jsd(client_datasets, lambda ex: ex.subject, subjects)

    print("=" * 60); print("DATA DISTRIBUTION SUMMARY  (dataset=mmlu)"); print("=" * 60)
    print(f"  Dataset: mmlu ({_HF_NAME}), {len(subjects)} subjects, labels {LABEL_SPACE}")
    print(f"  Client pool: {len(client_pool)} of {len(items)} test items, Dirichlet-over-subjects across {NUM_CLIENTS} clients")
    print(f"  Server queries: {len(server_queries)} (subject-stratified, QUERY_SEED={QUERY_SEED})")
    print(f"  Held-out eval:  {len(eval_set)} (subject-stratified, EVAL_SEED={EVAL_SEED}), disjoint from queries and pool")
    print(f"  Dirichlet alpha: {DIRICHLET_ALPHA}   partition SEED={SEED}")
    print(f"  Realized heterogeneity (mean pairwise JS divergence over subjects, base 2): {het:.3f}  [0=IID, 1=disjoint]")
    for k, cd in enumerate(client_datasets):
        top = defaultdict(int)
        for ex in cd:
            top[ex.subject] += 1
        n_subj_k = len(top)
        print(f"  Client {k}: {len(cd)} items across {n_subj_k} subjects")
    print("=" * 60)
    return server_queries, client_datasets, eval_set


def prepare_experiment():
    if DATASET == "mmlu":
        if not FILTER_LOCAL_DATA:
            raise SystemExit(
                "MMLU requires FED_ICL_FILTER=1 (paper's Algorithm 2 kNN filtering). "
                "Without it, relabelling the full subject-partitioned pool is far too many LLM calls."
            )
        return _prepare_mmlu()
    if DATASET == "dbpedia" and DATA_REGIME != "canonical_full":
        raise SystemExit("DBpedia is only wired for FED_ICL_REGIME=canonical_full.")
    if DATA_REGIME == "canonical_full":
        return _prepare_canonical_full()
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