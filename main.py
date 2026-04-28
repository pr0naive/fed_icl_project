"""
Fed-ICL Replication — Main Runner (Run this file to execute the full Fed-ICL experiment.)

Usage:
    python main.py

Prerequisites:
    1. Ollama must be running:     ollama serve
    2. Model must be pulled:       ollama pull llama3
    3. Install requests:           pip install requests

The experiment will:
    1. Partition data across clients using Dirichlet distribution
    2. Run Fed-ICL for T rounds
    3. Print accuracy per round
    4. Run baselines for comparison
    5. Save results to results.json
"""

import time
import json
import sys
import numpy as np
from config import (
    NUM_CLIENTS, NUM_ROUNDS, DIRICHLET_ALPHA, NUM_SHOTS,
    SELECTION_STRATEGY, MODEL_NAME, SEED,
)
from data import prepare_experiment, LABEL_SPACE
from federation import FedICLClient, FedICLServer
from llm import check_ollama_ready, predict_with_icl

np.random.seed(SEED)


def run_baseline_zero_shot(eval_set: list) -> float:
    """Baseline: zero-shot (no examples) on evaluation set."""
    print("\n" + "=" * 60)
    print("BASELINE: Zero-Shot (no examples)")
    print("=" * 60)

    correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        pred = predict_with_icl([], text)
        if pred == true_label:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_set)} "
                  f"(accuracy so far: {correct/(i+1):.1%})")

    accuracy = correct / len(eval_set)
    print(f"  Zero-shot accuracy: {accuracy:.1%} ({correct}/{len(eval_set)})")
    return accuracy


def run_baseline_local_only(client_datasets: list, eval_set: list) -> float:
    """Baseline: each client uses only its own local data (no federation)."""
    print("\n" + "=" * 60)
    print("BASELINE: Local-Only (no federation)")
    print("=" * 60)

    # Use client 0's local data as examples (simulates a single client)
    local_data = client_datasets[0]
    correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        # Random selection from local data
        n = min(NUM_SHOTS, len(local_data))
        indices = np.random.choice(len(local_data), size=n, replace=False)
        examples = [local_data[j] for j in indices]
        pred = predict_with_icl(examples, text)
        if pred == true_label:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_set)} "
                  f"(accuracy so far: {correct/(i+1):.1%})")

    accuracy = correct / len(eval_set)
    print(f"  Local-only accuracy: {accuracy:.1%} ({correct}/{len(eval_set)})")
    return accuracy


def run_fed_icl(server_queries, client_datasets, eval_set) -> dict:
    """
    Run the full Fed-ICL algorithm.

    Returns dict with per-round accuracy and timing.
    """
    print("\n" + "=" * 60)
    print("FED-ICL EXPERIMENT")
    print("=" * 60)
    print(f"  Clients:    {NUM_CLIENTS}")
    print(f"  Rounds:     {NUM_ROUNDS}")
    print(f"  Alpha:      {DIRICHLET_ALPHA}")
    print(f"  Shots:      {NUM_SHOTS}")
    print(f"  Selection:  {SELECTION_STRATEGY}")
    print(f"  Model:      {MODEL_NAME}")
    print("=" * 60)

    # Initialise server and clients
    server = FedICLServer(server_queries)
    clients = [
        FedICLClient(k, client_datasets[k])
        for k in range(NUM_CLIENTS)
    ]

    # Evaluate initial (random) context
    init_eval = server.evaluate_context()
    print(f"\n  Round 0 (random init): {init_eval['accuracy']:.1%}")

    results = {
        "config": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "num_shots": NUM_SHOTS,
            "selection_strategy": SELECTION_STRATEGY,
            "model": MODEL_NAME,
        },
        "rounds": [{"round": 0, "accuracy": init_eval["accuracy"]}],
        "baselines": {},
    }

    # ── Main Fed-ICL Loop ────────────────────────────────────
    total_start = time.time()

    for t in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"\n  ── Round {t}/{NUM_ROUNDS} ──")

        global_context = server.get_global_context()
        all_predictions = []

        for k, client in enumerate(clients):
            client_start = time.time()

            # Step 1: Client relabels local data using global context
            print(f"    Client {k}: relabelling local data...", end="", flush=True)
            client.relabel_local_data(global_context)
            print(" done.", flush=True)

            # Step 2: Client predicts answers for server queries
            print(f"    Client {k}: predicting server queries...", end="", flush=True)
            predictions = client.predict_server_queries(server_queries, global_context)
            all_predictions.append(predictions)

            client_time = time.time() - client_start
            print(f" done. ({client_time:.1f}s)", flush=True)

        # Server aggregates via majority vote
        server.aggregate_predictions(all_predictions)

        # Evaluate
        eval_result = server.evaluate_context()
        round_time = time.time() - round_start

        print(f"    → Round {t} accuracy: {eval_result['accuracy']:.1%} "
              f"({eval_result['correct']}/{eval_result['total']}) "
              f"[{round_time:.1f}s]")

        results["rounds"].append({
            "round": t,
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
            "time_seconds": round_time,
        })

    total_time = time.time() - total_start

    # ── Evaluate final context on held-out test set ──────────
    print(f"\n  ── Evaluating on held-out test set ({len(eval_set)} examples) ──")
    final_context = server.get_global_context()
    eval_correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        examples_for_eval = []
        n = min(NUM_SHOTS, len(final_context))
        indices = np.random.choice(len(final_context), size=n, replace=False)
        examples_for_eval = [final_context[j] for j in indices]
        pred = predict_with_icl(examples_for_eval, text)
        if pred == true_label:
            eval_correct += 1
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(eval_set)}", flush=True)

    eval_accuracy = eval_correct / len(eval_set)
    print(f"    Test set accuracy: {eval_accuracy:.1%} "
          f"({eval_correct}/{len(eval_set)})")

    results["eval_accuracy"] = eval_accuracy
    results["total_time_seconds"] = total_time

    return results


def print_summary(results: dict):
    """Print a clean summary of all results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Model:            {results['config']['model']}")
    print(f"  Clients:          {results['config']['num_clients']}")
    print(f"  Rounds:           {results['config']['num_rounds']}")
    print(f"  Dirichlet alpha:  {results['config']['dirichlet_alpha']}")
    print(f"  Selection:        {results['config']['selection_strategy']}")
    print(f"  Shots:            {results['config']['num_shots']}")
    print()

    print("  Round-by-round accuracy:")
    for r in results["rounds"]:
        bar = "█" * int(r["accuracy"] * 30)
        t = f" ({r.get('time_seconds', 0):.0f}s)" if r.get("time_seconds") else ""
        print(f"    Round {r['round']:2d}: {r['accuracy']:6.1%} {bar}{t}")

    print()
    if "baselines" in results and results["baselines"]:
        print("  Baselines:")
        for name, acc in results["baselines"].items():
            print(f"    {name:20s}: {acc:.1%}")
        print()

    if "eval_accuracy" in results:
        print(f"  Test set accuracy:  {results['eval_accuracy']:.1%}")

    fed_final = results["rounds"][-1]["accuracy"]
    fed_init = results["rounds"][0]["accuracy"]
    print(f"  Federation gain:    {fed_final - fed_init:+.1%} "
          f"(from {fed_init:.1%} to {fed_final:.1%})")
    print(f"  Total time:         {results.get('total_time_seconds', 0):.0f}s")
    print("=" * 60)


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Fed-ICL Replication — Sentiment Classification      ║")
    print("║     Wang et al. (ICML 2025) at Smaller Scale            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Check Ollama is ready
    if not check_ollama_ready():
        print("\nPlease start Ollama and try again.")
        sys.exit(1)

    # Prepare data
    server_queries, client_datasets, eval_set = prepare_experiment()

    # Run baselines
    print("\nRunning baselines first (this helps contextualise Fed-ICL results)...")
    zero_shot_acc = run_baseline_zero_shot(eval_set[:20])  # Subset for speed
    local_only_acc = run_baseline_local_only(client_datasets, eval_set[:20])

    # Run Fed-ICL
    results = run_fed_icl(server_queries, client_datasets, eval_set[:20])
    results["baselines"] = {
        "zero_shot": zero_shot_acc,
        "local_only": local_only_acc,
    }

    # Print summary
    print_summary(results)

    # Save results
    out_path = "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
