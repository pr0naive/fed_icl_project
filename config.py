"""
Fed-ICL Replication — Configuration (v2 - 4-class task)
========================================================
All tuneable experiment parameters in one place.
Adjust these before running experiments.
"""

# LLM Settings
MODEL_NAME = "mistral"           # Ollama model name (must be pulled first)
OLLAMA_HOST = "http://localhost:11434"
TEMPERATURE = 0.0               # 0 = deterministic outputs (reproducible)
MAX_TOKENS = 10                 # Short — we only need a category label

# Federation Settings
NUM_CLIENTS = 3                 # K: number of simulated clients
NUM_ROUNDS = 6                  # T: number of federation rounds
DIRICHLET_ALPHA = 0.5           # Controls data heterogeneity:
                                #   0.05 = extreme non-IID
                                #   0.5  = moderate non-IID (default)
                                #   10.0 = nearly uniform (IID)

# ICL Settings
NUM_SHOTS = 3                   # Number of in-context examples per prompt
SELECTION_STRATEGY = "random"   # Options: "random", "similarity"

# ── Ordering Settings (for dissertation experiments)
ORDER_STRATEGY = "original"     # Options: "original", "similarity_ascending",
                                #   "similarity_descending", "label_grouped",
                                #   "label_alternating", "random_shuffle"

# Server Query Settings 
NUM_SERVER_QUERIES = 100         # Queries forming the global context C

# Evaluation 
EVAL_SIZE = 100                  # Held-out test examples (never seen during federation)

# Random Seed 
SEED = 42
