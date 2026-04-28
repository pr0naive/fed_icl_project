# Fed-ICL Replication

**Replication of Wang et al. (ICML 2025) — Federated In-Context Learning**
At smaller scale, using Ollama on a MacBook Air M4.

## Quick Start

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Make sure you have llama3 pulled
ollama pull llama3

# 3. Install dependency
pip install requests

# 4. Run the experiment
cd fed_icl_replication
python main.py
```

The experiment takes roughly **30–60 minutes** depending on your machine
(3 clients × 6 rounds × ~70 LLM calls per client per round).

## What This Does

This replicates the core Fed-ICL algorithm on a **sentiment classification task**
(positive/negative) with 150 built-in examples:

1. **Partitions data** across 3 simulated clients using Dirichlet distribution
   (controls how non-IID each client's data is)
2. **Runs baselines**: zero-shot and local-only (no federation)
3. **Runs Fed-ICL** for 6 rounds:
   - Server sends global context C to all clients
   - Each client relabels its local data using C
   - Each client predicts answers for server queries using local data
   - Server aggregates via majority vote
   - Repeat
4. **Evaluates** on a held-out test set
5. **Saves results** to `results.json`

## Project Structure

```
fed_icl_replication/
├── config.py        # All tuneable parameters (change these for experiments)
├── data.py          # Built-in dataset + Dirichlet partitioning
├── llm.py           # Ollama interaction + prompt building
├── federation.py    # Client and Server classes (core algorithm)
├── main.py          # Entry point — runs everything
└── README.md        # This file
```

## Key Parameters (config.py)

| Parameter            | Default   | What it controls                          |
|----------------------|-----------|-------------------------------------------|
| `MODEL_NAME`         | `llama3`  | Which Ollama model to use                 |
| `NUM_CLIENTS`        | `3`       | K: number of simulated clients            |
| `NUM_ROUNDS`         | `6`       | T: number of federation rounds            |
| `DIRICHLET_ALPHA`    | `0.5`     | Data heterogeneity (lower = more non-IID) |
| `NUM_SHOTS`          | `3`       | Number of ICL examples per prompt         |
| `SELECTION_STRATEGY` | `random`  | How clients pick examples                 |

## What to Try First

After the default run completes:

1. **Change `DIRICHLET_ALPHA`** to `0.05` (extreme non-IID) and `10.0` (nearly IID).
   Compare how federation gain changes.

2. **Change `NUM_SHOTS`** from `3` to `1` and `5`.
   See how example count affects federation benefit.

3. **Change `SELECTION_STRATEGY`** to `"similarity"` and compare with `"random"`.
   This is a starting point for your dissertation experiments.

## Extending for Your Dissertation

### Adding new selection strategies
Edit `FedICLClient.select_examples()` in `federation.py`.
To add SBERT-based selection (KATE), install `sentence-transformers`:

```python
# In select_examples(), add:
elif SELECTION_STRATEGY == "sbert":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(query_text)
    pool_embs = model.encode([t for t, _ in pool])
    sims = np.dot(pool_embs, query_emb)
    top_indices = np.argsort(sims)[-n:]
    return [pool[i] for i in top_indices]
```

### Testing example ordering
After selecting examples, try different orderings before building the prompt:

```python
# In federation.py, after select_examples():
if ORDER_STRATEGY == "similarity_asc":
    examples.sort(key=lambda x: similarity(x[0], query_text))
elif ORDER_STRATEGY == "similarity_desc":
    examples.sort(key=lambda x: similarity(x[0], query_text), reverse=True)
elif ORDER_STRATEGY == "random":
    np.random.shuffle(examples)
```

### Heterogeneous LLMs
Assign different models to different clients:

```python
clients = [
    FedICLClient(0, client_datasets[0], model="llama3"),
    FedICLClient(1, client_datasets[1], model="mistral"),
    FedICLClient(2, client_datasets[2], model="phi"),
]
```

## Expected Output

```
Round 0 (random init): ~50%    (random labels = chance)
Round 1:               ~55-65% (first round of federation)
Round 2:               ~60-70% (improving)
...
Round 6:               ~70-80% (convergence)

Zero-shot baseline:    ~65-75%
Local-only baseline:   ~55-70% (depends on data partition)

Federation gain:       +15-25% (from random init to convergence)
```

Results will vary by model, data partition, and random seed.
