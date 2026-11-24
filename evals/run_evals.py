"""
evals/run_evals.py
-------------------
Master evaluation runner — runs all eval suites and prints a combined report.

Usage:
    python -m evals.run_evals

Output:
    Guardrail Accuracy, Routing Precision, Faithfulness Judge Accuracy
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from evals.test_routing import load_dataset, run_guardrail_eval, run_routing_eval
from evals.test_faithfulness import FAITHFULNESS_CASES, evaluate_faithfulness


def main():
    print("\n" + "=" * 65)
    print("  TripMate AI — Full Offline Evaluation Suite")
    print("=" * 65)

    total_start = time.time()

    # 1. Routing & Guardrail Evals
    print("\n[Phase 1] Supervisor Guardrail & Routing Evals")
    print("-" * 65)
    dataset = load_dataset()
    print(f"  Loaded {len(dataset)} golden test cases\n")

    t = time.time()
    gr = run_guardrail_eval(dataset)
    rr = run_routing_eval(dataset)
    phase1_time = time.time() - t

    print(f"  Guardrail Accuracy : {gr['accuracy']:.1%}  ({gr['correct']}/{gr['total']})")
    print(f"  Routing Precision  : {rr['precision']:.1%}  ({rr['correct']}/{rr['total']})")
    print(f"  Completed in {phase1_time:.1f}s")

    # 2. Faithfulness Eval
    print("\n[Phase 2] Faithfulness Eval (LLM-as-a-Judge)")
    print("-" * 65)
    t = time.time()
    faith_results = [evaluate_faithfulness(c) for c in FAITHFULNESS_CASES]
    phase2_time = time.time() - t
    faith_acc = sum(r["judge_correct"] for r in faith_results) / len(faith_results)
    print(f"  Faithfulness Accuracy: {faith_acc:.1%}  ({sum(r['judge_correct'] for r in faith_results)}/{len(faith_results)})")
    print(f"  Completed in {phase2_time:.1f}s")

    # Final Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Guardrail Accuracy   : {gr['accuracy']:.1%}")
    print(f"  Routing Precision    : {rr['precision']:.1%}")
    print(f"  Faithfulness Score   : {faith_acc:.1%}")
    print(f"  Total eval time      : {total_time:.1f}s")
    print("=" * 65 + "\n")

    # Exit with non-zero if any metric falls below threshold
    THRESHOLDS = {
        "guardrail": 0.85,
        "routing": 0.75,
        "faithfulness": 0.80,
    }
    failures = []
    if gr["accuracy"] < THRESHOLDS["guardrail"]:
        failures.append(f"Guardrail accuracy {gr['accuracy']:.1%} below threshold {THRESHOLDS['guardrail']:.0%}")
    if rr["precision"] < THRESHOLDS["routing"]:
        failures.append(f"Routing precision {rr['precision']:.1%} below threshold {THRESHOLDS['routing']:.0%}")
    if faith_acc < THRESHOLDS["faithfulness"]:
        failures.append(f"Faithfulness score {faith_acc:.1%} below threshold {THRESHOLDS['faithfulness']:.0%}")

    if failures:
        print("❌ EVAL FAILED — Threshold violations:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print("✅ All eval metrics passed thresholds.\n")


if __name__ == "__main__":
    main()
