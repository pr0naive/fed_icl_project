import time
import json
import sys
import numpy as np

from config import (
    NUM_CLIENTS, NUM_ROUNDS, DIRICHLET_ALPHA, NUM_SHOTS,
    SELECTION_STRATEGY, ORDER_STRATEGY, MODEL_NAME, SEED,
    NUM_SERVER_QUERIES, EVAL_SIZE, CLIENT_POOL_SIZE
)
from data import prepare_experiment, LABEL_SPACE
from federation import FedICLClient, FedICLServer
from llm import check_ollama_ready, predict_with_icl, get_parse_stats

np.random.seed(SEED)


def run_baseline_zero_shot(eval_set: list) -> float:
    print("\n" + "=" * 60)
    print("BASELINE: Zero-Shot")
    print("=" * 60)
    correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        pred = predict_with_icl([], text)
        if pred == true_label:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_set)} (acc: {correct/(i+1):.1%})")
    accuracy = correct / len(eval_set)
    print(f"  Zero-shot accuracy: {accuracy:.1%} ({correct}/{len(eval_set)})")
    return accuracy


def run_baseline_local_only(client_datasets: list, eval_set: list) -> float:
    """Client 0's local pool only, with the SAME select+order pipeline as Fed-ICL."""
    print("\n" + "=" * 60)
    print(f"BASELINE: Local-Only (Client 0; ORDER_STRATEGY={ORDER_STRATEGY})")
    print("=" * 60)
    client = FedICLClient(client_id=0, local_data=client_datasets[0], model=MODEL_NAME)
    correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        examples = client.select_examples(text, client.local_data, NUM_SHOTS)
        examples = client.order_examples(examples, text)
        pred = predict_with_icl(examples, text, model=MODEL_NAME)
        if pred == true_label:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_set)} (acc: {correct/(i+1):.1%})")
    accuracy = correct / len(eval_set)
    print(f"  Local-only accuracy: {accuracy:.1%} ({correct}/{len(eval_set)})")
    return accuracy


def run_fed_icl(server_queries, client_datasets, eval_set) -> dict:
    print("\n" + "=" * 60)
    print("FED-ICL EXPERIMENT")
    print(f"  K={NUM_CLIENTS}  T={NUM_ROUNDS}  alpha={DIRICHLET_ALPHA}")
    print(f"  shots={NUM_SHOTS}  selection={SELECTION_STRATEGY}  order={ORDER_STRATEGY}")
    print(f"  model={MODEL_NAME}  seed={SEED}")
    print("=" * 60)

    server  = FedICLServer(server_queries)
    clients = [FedICLClient(k, client_datasets[k], model=MODEL_NAME)
               for k in range(NUM_CLIENTS)]

    init_eval = server.evaluate_context()
    print(f"\n  Round 0 (random init): {init_eval['accuracy']:.1%}")

    results = {
        "config": {
            "model": MODEL_NAME,
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "num_shots": NUM_SHOTS,
            "num_server_queries": NUM_SERVER_QUERIES,
            "eval_size": EVAL_SIZE,
            "selection_strategy": SELECTION_STRATEGY,
            "order_strategy": ORDER_STRATEGY,
            "seed": SEED,
            "client_pool_size": CLIENT_POOL_SIZE,
        },
        "rounds": [{"round": 0, "accuracy": init_eval["accuracy"]}],
        "baselines": {},
    }

    total_start = time.time()
    for t in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"\n  -- Round {t}/{NUM_ROUNDS} --")
        global_context = server.get_global_context()
        all_predictions = []
        for k, client in enumerate(clients):
            cs = time.time()
            print(f"  Client {k}: relabelling...", end="", flush=True)
            client.relabel_local_data(global_context)
            print(" done.", flush=True)
            print(f"  Client {k}: predicting server queries...", end="", flush=True)
            preds = client.predict_server_queries(server_queries, global_context)
            all_predictions.append(preds)
            print(f" done ({time.time()-cs:.1f}s).", flush=True)
        server.aggregate_predictions(all_predictions)
        eval_result = server.evaluate_context()
        rt = time.time() - round_start
        print(f"   → Round {t}: {eval_result['accuracy']:.1%} "
              f"({eval_result['correct']}/{eval_result['total']}) [{rt:.1f}s]")
        results["rounds"].append({
            "round": t,
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
            "time_seconds": rt,
        })

    total_time = time.time() - total_start

    # Held-out evaluation, now using the same select+order pipeline.
    print(f"\n  -- Held-out test ({len(eval_set)} examples; ordering applied) --")
    final_context = server.get_global_context()
    final_client  = FedICLClient(client_id=-1, local_data=final_context, model=MODEL_NAME)
    eval_correct = 0
    for i, (text, true_label) in enumerate(eval_set):
        ex = final_client.select_examples(text, final_context, NUM_SHOTS)
        ex = final_client.order_examples(ex, text)
        pred = predict_with_icl(ex, text, model=MODEL_NAME)
        if pred == true_label:
            eval_correct += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_set)}", flush=True)
    eval_accuracy = eval_correct / len(eval_set)
    print(f"  Held-out accuracy: {eval_accuracy:.1%} ({eval_correct}/{len(eval_set)})")

    results["eval_accuracy"]      = eval_accuracy
    results["total_time_seconds"] = total_time
    return results


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for k, v in results["config"].items():
        print(f"  {k}: {v}")
    print()
    print("  Round-by-round accuracy:")
    for r in results["rounds"]:
        bar = "█" * int(r["accuracy"] * 30)
        t = f" ({r.get('time_seconds', 0):.0f}s)" if r.get("time_seconds") else ""
        print(f"   Round {r['round']:2d}: {r['accuracy']:6.1%} {bar}{t}")
    if results.get("baselines"):
        print("\n  Baselines:")
        for name, acc in results["baselines"].items():
            print(f"   {name:20s}: {acc:.1%}")
    if "eval_accuracy" in results:
        print(f"\n  Held-out accuracy: {results['eval_accuracy']:.1%}")
    if "eval_accuracy" in results and results.get("baselines"):
        local_only = results["baselines"].get("local_only")
    if local_only is not None:
        fed_final = results["rounds"][-1]["accuracy"]
        fed_gain  = (fed_final - local_only) * 100
        print(f"  Federation gain over local-only: {fed_gain:+.1f}pp "
              f"(local-only {local_only:.1%} → Fed-ICL R{NUM_ROUNDS} {fed_final:.1%})")
    if results.get("parse_stats"):
        ps = results["parse_stats"]
        print(f"\n  Parse fallback rate: {ps['fallback_rate']:.1%} "
              f"({ps['fallback_count']}/{ps['total_calls']})")
    print(f"  Total time: {results.get('total_time_seconds', 0):.0f}s")
    print("=" * 60)
    


def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("  ║  Fed-ICL Replication  —  AG News Topic Classification║")
    print("  ╚══════════════════════════════════════════════════════╝\n")

    if not check_ollama_ready():
        sys.exit(1)

    server_queries, client_datasets, eval_set = prepare_experiment()

    print("\nRunning baselines first...")
    zero_shot_acc  = run_baseline_zero_shot(eval_set)
    local_only_acc = run_baseline_local_only(client_datasets, eval_set)

    results = run_fed_icl(server_queries, client_datasets, eval_set)
    results["baselines"]   = {"zero_shot": zero_shot_acc, "local_only": local_only_acc}
    results["parse_stats"] = get_parse_stats()

    print_summary(results)

    out_path = (
        f"results_{MODEL_NAME}"
        f"_alpha{DIRICHLET_ALPHA}"
        f"_K{NUM_SHOTS}"
        f"_T{NUM_ROUNDS}"
        f"_pool{CLIENT_POOL_SIZE}"
        f"_seed{SEED}"
        f"_order-{ORDER_STRATEGY}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()