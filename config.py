"""
Fed-ICL Replication Configuration (v3 -  env-overrideable)
========================================================
All tuneable experiment parameters in one place.
Adjust these before running experiments.
"""
import os
# LLM Settings
MODEL_NAME = os.environ.get("FED_ICL_MODEL", "mistral")           # Ollama model name (must be pulled first)
OLLAMA_HOST = os.environ.get("FED_ICL_HOST", "http://localhost:11434")
TEMPERATURE = 0.0               # 0 = deterministic outputs (reproducible)
MAX_TOKENS = 10                 # Short - we only need a category label

# Federation variant (Wang et al., Appendix C.3)
FED_VARIANT = os.environ.get("FED_ICL_VARIANT", "fed_icl")
# Options:
#   "fed_icl"       Step 2 conditions on D^i AND D_k^i (local + relabelled).
#   "fed_icl_free"  Step 2 conditions on D_k^i only (the label-free variant);
#                   falls back to local data if the relabelled pool is empty.

# Federation Settings
NUM_CLIENTS = int(os.environ.get("FED_ICL_CLIENTS", 3))                 # K: number of simulated clients
NUM_ROUNDS = int(os.environ.get("FED_ICL_ROUNDS", 6))                  # T: number of federation rounds
DIRICHLET_ALPHA = float(os.environ.get("FED_ICL_ALPHA", 0.5))           # Controls data heterogeneity:
                                                                        #   0.05 = extreme non-IID
                                                                        #   0.5  = moderate non-IID (default)
                                                                        #   10.0 = nearly uniform (IID)

# ICL Settings
NUM_SHOTS = int(os.environ.get("FED_ICL_K", 3))                   # Number of in-context examples per prompt
SELECTION_STRATEGY = os.environ.get("FED_ICL_SEL", "similarity_embedding")
# Options:
#   "similarity_embedding"  kNN in paraphrase-MiniLM-L6-v2 embedding space.
#                           Paper-faithful (Wang et al., Appendix C.1) and the DEFAULT. Requires sentence-transformers.
#   "similarity"            lexical word-overlap. Documented deviation, kept as an ablation arm, NOT the paper's method.
#   "random"                no filtering. Corresponds to the paper's "without filtering" ablation (their Figure 8).

# Ordering Settings (for dissertation experiments)
ORDER_STRATEGY = os.environ.get("FED_ICL_ORDER", "original")     # Options: "original", "similarity_ascending", "similarity_descending", "label_grouped", "label_alternating", "random_shuffle"
                                                               

# Server Query Settings 
NUM_SERVER_QUERIES = int(os.environ.get("FED_ICL_Q", 100))         # Queries forming the global context C
CLIENT_POOL_SIZE = int(os.environ.get("FED_ICL_POOL_SIZE", 250))
# Evaluation 
# Evaluation
EVAL_SIZE = int(os.environ.get("FED_ICL_EVAL", 1000))            # Held-out test examples (never seen during federation)
EVAL_SEED = int(os.environ.get("FED_ICL_EVAL_SEED", 12345))      # FIXED, decoupled from partition SEED.
                                                                 # The eval set must be identical across partition seeds, otherwise partition variance and evaluation-sampling noise are conflated.

# Random Seed 
SEED = int(os.environ.get("FED_ICL_SEED", 42))


# --- One-seed 80/20 run (Dr. Jin) + paper-faithful filtering ------------------
# "canonical"  train split = pool/query frame, test split = eval (DEFAULT)
# "split8020"  80% of the 120k train -> clients, 20% -> eval; queries off the 80%
DATA_REGIME   = os.environ.get("FED_ICL_REGIME", "canonical")
SPLIT_SEED    = int(os.environ.get("FED_ICL_SPLIT_SEED", 2024))   # fixes the 80/20 split
TEST_FRACTION = float(os.environ.get("FED_ICL_TEST_FRAC", 0.2))

# Paper's one-time local dataset filtering (Wang et al., App. C.1, Algorithm 2).
# OFF by default. REQUIRED for split8020, else relabelling is ~576k LLM calls.
FILTER_LOCAL_DATA  = os.environ.get("FED_ICL_FILTER", "0") == "1"

STRATIFIED_QUERIES = os.environ.get("FED_ICL_STRAT_QUERIES", "0") == "1"
QUERY_SEED         = int(os.environ.get("FED_ICL_QUERY_SEED", 12345))