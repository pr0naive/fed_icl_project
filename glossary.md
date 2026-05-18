# Glossary

Definitions for every term and code variable used in this project. Organised by concept area so the file can be skimmed by topic rather than alphabetically. Variable names in `CODE_FONT` map to identifiers in `config.py` or in the modules named.

## 1. In-context learning

**In-context learning (ICL).** A paradigm in which a language model performs a task by conditioning on a small number of labelled demonstrations placed in the prompt, with no updates to model parameters. The model's response to a test query depends on which demonstrations are shown and in what order, but not on any learned weights.

**Demonstration (in-context example).** A single (input, label) pair shown in the prompt before the test query. In this project, the input is an AG News headline and the label is one of `world`, `sports`, `business`, `science`.

**Shot count (`NUM_SHOTS`, paper notation K).** The number of demonstrations included in the prompt. Zero-shot means no demonstrations. This project tests K ∈ {1, 3, 5} during the shot-count sweep; the default for headline runs is K=3.

**Prompt template.** The fixed string format used to wrap demonstrations and the test query into the input string sent to the model. Held constant across all conditions so that effects can be attributed to selection and ordering rather than prompt wording. Defined in `build_icl_prompt()` in `llm.py`.

**Test query.** The input instance on which the model is asked to make a prediction, appended to the prompt after the K demonstrations.

## 2. Federated learning

**Federated learning (FL).** A protocol in which multiple clients hold their own data and collaborate by exchanging messages without exposing raw data. FedAvg (McMahan et al., 2017) is the canonical example, in which clients exchange model weight updates.

**Client.** An entity holding a private local pool of labelled examples. This project simulates `NUM_CLIENTS = 3` clients on a single machine.

**`NUM_CLIENTS` (paper notation K, but written as N in this project to avoid clashing with shot count).** Number of clients in the federation. Default 3.

**Local pool.** The set of labelled examples one client holds, allocated from the global training set by the Dirichlet partition. Sizes vary across clients depending on the Dirichlet draw.

**Federation round (`NUM_ROUNDS`, paper notation T).** One cycle of local computation followed by inter-client information exchange. T=6 is the default in this project.

**Aggregation.** The operation that combines messages from clients into a shared state for the next round. In FedAvg this is over weights; in Fed-ICL it is over predictions on shared server queries, via majority vote. Implemented in `aggregate_predictions()` in `federation.py`.

## 3. Fed-ICL protocol

**Fed-ICL (Wang et al., 2025).** A federated protocol for collaborative in-context learning. Each client holds a local pool of demonstrations and a shared base language model; clients exchange predictions across rounds rather than model weights.

**Fed-ICL-Free.** The variant of the protocol that does not require a labelled validation set on the server side. Used in this project because it matches the constraints of local replication. The default variant in `federation.py`.

**Fed-ICL-Standard.** The variant requiring a labelled validation set on the server side for example scoring. Not used here.

**Server query.** A labelled example held by the server, used as a shared test prompt that every client predicts each round. The server uses the resulting per-client predictions to update the shared global context. `NUM_SERVER_QUERIES = 100` by default.

**Global context.** The server-side set of (server query, current best label) pairs that all clients see at the start of each round. Initialised randomly at round 0; updated by majority-vote aggregation after each round.

**Local relabelling.** The step in which each client relabels its own local examples using the current global context as in-context demonstrations. Produces a `relabelled_data` set distinct from the client's original local data. Implemented in `relabel_local_data()` in `federation.py`.

## 4. Data heterogeneity

**Data heterogeneity.** The degree to which client data distributions differ from each other. Higher heterogeneity means each client sees a less representative slice of the global distribution.

**Non-IID.** Non-independent and identically distributed. Client data allocation is non-IID when the per-client label distributions differ from the global distribution.

**Dirichlet partition.** A method for allocating labelled examples to clients in which each class's allocation across clients is drawn from a Dirichlet distribution. Standard in federated learning evaluation (Hsu et al., 2019). Implemented in `partition_data_dirichlet()` in `data.py`.

**Dirichlet concentration parameter (`DIRICHLET_ALPHA`, paper notation α).** Controls the partition. Small α (0.05) produces highly skewed per-client label distributions, where some clients may hold only one or two classes. Large α (10.0) approaches uniform allocation across clients. This project sweeps α ∈ {0.05, 0.5, 10.0}.

**Residual distribution.** When the Dirichlet draw is converted to integer counts via floor rounding, some examples are unallocated. This project distributes the residual round-robin starting from a randomly chosen client. Earlier code dumped all residual on client 0, which biased the local-only baseline.

## 5. Demonstration selection

**Demonstration selection (`SELECTION_STRATEGY`).** The rule by which K examples are chosen from a client's local pool for inclusion in a prompt. Distinct from ordering, which is what happens after selection. Implemented in `select_examples()` in `federation.py`.

**Random selection.** K examples drawn uniformly at random from the local pool. The baseline.

**Similarity selection.** K examples chosen by lexical-overlap similarity to the test query (set intersection on lowercased tokens). A lightweight proxy for KATE (Liu et al., 2022), which uses sentence embeddings. SBERT-based similarity is on the roadmap.

## 6. Ordering strategies

**Ordering strategy (`ORDER_STRATEGY`).** The rule by which the K selected demonstrations are arranged in the prompt. The primary experimental variable of this dissertation. Implemented in `order_examples()` in `federation.py`.

**Original.** Examples in the order returned by the selection step. The control. After the fix to `select_examples`, this is now genuinely unordered under both random and similarity selection.

**Similarity-ascending (`similarity_ascending`).** Examples sorted by lexical-overlap similarity to the test query in ascending order, so the most similar example appears last (closest to the query).

**Similarity-descending (`similarity_descending`).** The reverse, so the most similar example appears first.

**Label-grouped (`label_grouped`).** Examples sorted by label, so all examples of one class appear consecutively before the next class begins.

**Label-alternating (`label_alternating`).** Round-robin across labels, so consecutive positions cycle through the label space until examples run out.

**Random-shuffle (`random_shuffle`).** A fresh random order each time, used to quantify ordering-induced variance.

## 7. Baselines

**Zero-shot baseline.** Model accuracy when given no demonstrations, only the test query. If Fed-ICL does not exceed zero-shot, demonstrations are not contributing.

**Local-only baseline.** Accuracy of Client 0 using only its own local pool of K demonstrations, with the same select-and-order pipeline as Fed-ICL, and no exchange of information across clients. The decisive test of whether federation justifies its complexity overhead. Implemented in `run_baseline_local_only()` in `main.py`.

**Held-out evaluation.** Accuracy on examples drawn from the AG News test split (not the same subsample as `server_queries` and `client_pool`), evaluated using the final Fed-ICL global context with the same select-and-order pipeline as the in-pool evaluation. Reports whether the federation result generalises off the server queries.

## 8. Evaluation metrics

**Evaluation pool (`EVAL_SIZE`, paper notation n).** The number of test instances on which all conditions are scored. Default n=100. Held constant across orderings, alpha values, and models so comparisons are valid.

**Round accuracy.** Accuracy of the current global context on the `server_queries` set, computed after each federation round. Plotted as the convergence trajectory.

**Final accuracy.** Round-T accuracy. The primary outcome variable for ordering and heterogeneity comparisons.

**Held-out accuracy.** Round-T accuracy on the test-split evaluation pool.

**Parse fallback rate.** Fraction of model calls in a run whose output could not be parsed into any of the four AG News labels and fell back to returning `LABEL_SPACE[0]` (which is `world`). Logged by `parse_label()` in `llm.py` and reported in every results JSON. A high rate would mean ordering effects could be confounded with parser failures; the project flags any cell exceeding 10%.

## 9. Reproducibility

**Seed (`SEED`).** Integer used to initialise all random operations within a run: the data subsample, the Dirichlet partition, the selection shuffle, and the ordering shuffle. Three seeds are used per configuration to estimate run-to-run variance.

**Paired comparison.** A comparison in which two conditions are evaluated on the same seeds and the same evaluation pool, so the difference attributes to the condition rather than to sampling noise.

**Environment variable overrides (`FED_ICL_*`).** All experimental parameters in `config.py` can be overridden from the shell. Examples: `FED_ICL_MODEL`, `FED_ICL_ALPHA`, `FED_ICL_K`, `FED_ICL_ORDER`, `FED_ICL_SEED`. Means a sweep is a shell loop, not a series of source edits.

**Auto-named output file.** Each run writes to `results_{model}_alpha{α}_K{K}_T{T}_seed{S}_order-{order}.json` so two runs with different parameters cannot overwrite each other.

## 10. Code identifiers (quick reference)

| Identifier              | File          | Meaning                                                |
|-------------------------|---------------|--------------------------------------------------------|
| `MODEL_NAME`            | `config.py`   | Ollama model identifier (llama3, mistral, phi).        |
| `NUM_CLIENTS`           | `config.py`   | Number of simulated clients (N in glossary).           |
| `NUM_ROUNDS`            | `config.py`   | Federation rounds T.                                   |
| `NUM_SHOTS`             | `config.py`   | Demonstrations per prompt K.                           |
| `NUM_SERVER_QUERIES`    | `config.py`   | Number of shared server queries.                       |
| `EVAL_SIZE`             | `config.py`   | Held-out evaluation pool size n.                       |
| `DIRICHLET_ALPHA`       | `config.py`   | Concentration parameter α for the partition.           |
| `SELECTION_STRATEGY`    | `config.py`   | `random` or `similarity`.                              |
| `ORDER_STRATEGY`        | `config.py`   | One of six values (see Section 6).                     |
| `SEED`                  | `config.py`   | Random seed.                                           |
| `LABEL_SPACE`           | `data.py`     | `["world", "sports", "business", "science"]`.          |
| `RAW_DATA`              | `data.py`     | 350-row subsample of AG News train split.              |
| `partition_data_dirichlet` | `data.py`  | Implements the Dirichlet allocation.                   |
| `FedICLClient`          | `federation.py` | Holds a local pool, implements relabel and predict.  |
| `FedICLServer`          | `federation.py` | Holds the global context, aggregates predictions.    |
| `select_examples`       | `federation.py` | Picks K examples (random or similarity).             |
| `order_examples`        | `federation.py` | Arranges the K examples per `ORDER_STRATEGY`.        |
| `aggregate_predictions` | `federation.py` | Majority-vote over per-client predictions.           |
| `parse_label`           | `llm.py`      | Token-level matching; logs fallbacks.                  |
| `query_ollama`          | `llm.py`      | HTTP call to Ollama with retry on transient failure.   |
| `_PARSE_STATS`          | `llm.py`      | Module-level dict tracking parse fallback rate.        |
| `run_baseline_zero_shot`| `main.py`     | Zero-shot baseline runner.                             |
| `run_baseline_local_only`| `main.py`    | Local-only baseline using Client 0.                    |
| `run_fed_icl`           | `main.py`     | Main Fed-ICL training and evaluation loop.             |