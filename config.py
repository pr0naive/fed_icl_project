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

# Federation Settings
NUM_CLIENTS = int(os.environ.get("FED_ICL_CLIENTS", 3))                 # K: number of simulated clients
NUM_ROUNDS = int(os.environ.get("FED_ICL_ROUNDS", 6))                  # T: number of federation rounds
DIRICHLET_ALPHA = float(os.environ.get("FED_ICL_ALPHA", 0.5))           # Controls data heterogeneity:
                                                                        #   0.05 = extreme non-IID
                                                                        #   0.5  = moderate non-IID (default)
                                                                        #   10.0 = nearly uniform (IID)

# ICL Settings
NUM_SHOTS = int(os.environ.get("FED_ICL_K", 3))                   # Number of in-context examples per prompt
SELECTION_STRATEGY = os.environ.get("FED_ICL_SEL", "random")   # Options: "random", "similarity"

# Ordering Settings (for dissertation experiments)
ORDER_STRATEGY = os.environ.get("FED_ICL_ORDER", "original")     # Options: "original", "similarity_ascending", "similarity_descending", "label_grouped", "label_alternating", "random_shuffle"
                                                               

# Server Query Settings 
NUM_SERVER_QUERIES = int(os.environ.get("FED_ICL_Q", 100))         # Queries forming the global context C
CLIENT_POOL_SIZE = int(os.environ.get("FED_ICL_POOL_SIZE", 250))
# Evaluation 
EVAL_SIZE = int(os.environ.get("FED_ICL_EVAL", 100))                 # Held-out test examples (never seen during federation)

# Random Seed 
SEED = int(os.environ.get("FED_ICL_SEED", 42))
