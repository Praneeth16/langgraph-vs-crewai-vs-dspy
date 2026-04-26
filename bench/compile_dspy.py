"""Compile the DSPy refund program with BootstrapFewShot, then re-benchmark.

Ground-truth label: `eligible = (policy_fit == 'damage_yes' and days_since_delivery < 30)`
plus damage_maybe leans to eligible = True.

Splits 100 tickets into 50 train / 50 test. Compiles with BootstrapFewShot
(metric = exact match on `eligible`), evaluates uncompiled vs compiled on the
same test set. Captures latency + tokens per ticket on test.

Output: results/dspy_compiled_eval.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import dspy
import litellm

bucket = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def reset():
    for k in bucket:
        bucket[k] = 0


def cb(kwargs, completion_response, start_time, end_time):
    try:
        u = completion_response.get("usage") or {}
        bucket["prompt"] += int(u.get("prompt_tokens") or 0)
        bucket["completion"] += int(u.get("completion_tokens") or 0)
        bucket["total"] += int(u.get("total_tokens") or 0)
        bucket["calls"] += 1
    except Exception:
        pass


litellm.success_callback = [cb]


def label(meta):
    fit = meta.get("policy_fit")
    days = meta.get("days_since_delivery", 999)
    if fit == "damage_yes" and days < 30:
        return True
    if fit == "damage_maybe":
        return True
    return False


class Triage(dspy.Signature):
    """Decide refund eligibility given ticket and policy."""
    email_text: str = dspy.InputField()
    policy: str = dspy.InputField()
    eligible: bool = dspy.OutputField()
    rationale: str = dspy.OutputField()


def make_examples(tickets):
    out = []
    for t in tickets:
        ex = dspy.Example(
            email_text=t["email_text"],
            policy="Damaged items refundable within 30 days.",
            eligible=label(t["meta"]),
            rationale="",
        ).with_inputs("email_text", "policy")
        out.append(ex)
    return out


def metric(example, pred, trace=None):
    return bool(example.eligible) == bool(pred.eligible)


def evaluate(program, examples):
    correct = 0
    rows = []
    for ex in examples:
        reset()
        t0 = time.perf_counter()
        try:
            pred = program(email_text=ex.email_text, policy=ex.policy)
            ok = metric(ex, pred)
        except Exception as e:
            rows.append({"error": f"{type(e).__name__}: {e}"})
            continue
        wall = time.perf_counter() - t0
        if ok:
            correct += 1
        rows.append({
            "latency_s": round(wall, 3),
            "prompt": bucket["prompt"],
            "completion": bucket["completion"],
            "total": bucket["total"],
            "calls": bucket["calls"],
            "correct": ok,
            "predicted_eligible": getattr(pred, "eligible", None),
            "true_eligible": ex.eligible,
        })
    return rows, correct / max(len(examples), 1)


def main():
    import argparse, random as _r
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("set GEMINI_API_KEY", file=sys.stderr); sys.exit(1)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    print(f"Compiling against {model} (seed={args.seed})")
    dspy.configure(lm=dspy.LM(f"gemini/{model}", api_key=os.environ["GEMINI_API_KEY"], temperature=0, cache=False))

    tickets = json.loads((HERE / "tickets_100.json").read_text())
    train_t = tickets[:50]
    test_t = tickets[50:]
    # Shuffle trainset with provided seed so different seeds produce different demo selections
    _r.Random(args.seed).shuffle(train_t)
    trainset = make_examples(train_t)
    testset = make_examples(test_t)

    print(f"Train labels: True={sum(1 for e in trainset if e.eligible)}/{len(trainset)}")
    print(f"Test  labels: True={sum(1 for e in testset if e.eligible)}/{len(testset)}")

    base = dspy.Predict(Triage)
    print("\n--- Uncompiled eval ---")
    reset()
    base_rows, base_acc = evaluate(base, testset)
    base_calls = sum(r.get("calls", 0) for r in base_rows if "error" not in r)
    base_tok = sum(r.get("total", 0) for r in base_rows if "error" not in r)
    print(f"  acc={base_acc:.3f} calls={base_calls} total_tokens={base_tok}")

    print("\n--- Compiling with BootstrapFewShot (max_bootstrapped_demos=4) ---")
    optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
    t0 = time.perf_counter()
    compiled = optimizer.compile(student=dspy.Predict(Triage), trainset=trainset)
    compile_wall = time.perf_counter() - t0
    print(f"  compile wall: {compile_wall:.1f}s")

    print("\n--- Compiled eval ---")
    comp_rows, comp_acc = evaluate(compiled, testset)
    comp_calls = sum(r.get("calls", 0) for r in comp_rows if "error" not in r)
    comp_tok = sum(r.get("total", 0) for r in comp_rows if "error" not in r)
    print(f"  acc={comp_acc:.3f} calls={comp_calls} total_tokens={comp_tok}")

    out = {
        "model": model,
        "train_size": len(trainset),
        "test_size": len(testset),
        "uncompiled": {"accuracy": base_acc, "rows": base_rows, "total_tokens": base_tok, "total_calls": base_calls},
        "compiled": {"accuracy": comp_acc, "rows": comp_rows, "total_tokens": comp_tok, "total_calls": comp_calls,
                     "compile_wall_s": round(compile_wall, 1)},
    }
    out["seed"] = args.seed
    p = ROOT / "results" / f"dspy_compiled_eval_{model}_seed{args.seed}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")
    print(f"\nDelta: acc {base_acc:.3f} -> {comp_acc:.3f}; tokens {base_tok} -> {comp_tok}")


if __name__ == "__main__":
    main()
