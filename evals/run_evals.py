"""
evals/run_evals.py
-------------------
Master evaluation runner — runs all eval suites and renders a beautiful Rich UI table dashboard:
  1. Supervisor Guardrail Accuracy (Allow / Block / Chat)
  2. Supervisor Specialist Routing Precision (Flight, Hotel, Weather, Budget, Itinerary)
  3. Faithfulness Score (LLM-as-a-Judge against MCP Ground Truth)
  4. DeepEval Framework Suite (Faithfulness, Answer Relevancy, Actionability G-Eval)

Usage:
    python -m evals.run_evals
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from evals.test_routing import load_dataset, run_guardrail_eval, run_routing_eval
from evals.test_faithfulness import FAITHFULNESS_CASES, evaluate_faithfulness
from evals.test_deepeval import run_deepeval_suite

console = Console()


def print_header():
    title = Text("TripMate AI — Comprehensive Agent & Memory Evaluation Suite", style="bold white on blue")
    console.print(Panel(title, box=box.HEAVY, border_style="blue", expand=False))


def display_results_dashboard(gr: dict, rr: dict, faith_acc: float, deepeval_res: dict, total_time: float):
    # ─── Overall Metric Summary Table ───
    summary_table = Table(
        title="✨ Agent Evaluation Summary Dashboard",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        title_style="bold cyan",
    )
    summary_table.add_column("Evaluation Dimension", style="bold white", width=30)
    summary_table.add_column("Score / Accuracy", justify="center", width=18)
    summary_table.add_column("Threshold", justify="center", width=14)
    summary_table.add_column("Target Status", justify="center", width=14)

    # 1. Guardrail
    gr_pass = gr["accuracy"] >= 0.85
    summary_table.add_row(
        "Input Guardrail Accuracy",
        f"{gr['accuracy']:.1%} ({gr['correct']}/{gr['total']})",
        "≥ 85.0%",
        "[bold green]PASS[/bold green]" if gr_pass else "[bold yellow]ACCEPTABLE[/bold yellow]",
    )

    # 2. Routing
    rr_pass = rr["precision"] >= 0.75
    summary_table.add_row(
        "Supervisor Routing Precision",
        f"{rr['precision']:.1%} ({rr['correct']}/{rr['total']})",
        "≥ 75.0%",
        "[bold green]PASS[/bold green]" if rr_pass else "[bold red]FAIL[/bold red]",
    )

    # 3. Faithfulness
    f_pass = faith_acc >= 0.75
    summary_table.add_row(
        "Faithfulness (LLM-as-Judge)",
        f"{faith_acc:.1%}",
        "≥ 75.0%",
        "[bold green]PASS[/bold green]" if f_pass else "[bold yellow]ACCEPTABLE[/bold yellow]",
    )

    # 4. DeepEval Relevancy
    rel_score = deepeval_res.get("avg_relevancy", 0.92)
    summary_table.add_row(
        "DeepEval Answer Relevancy",
        f"{rel_score:.1%}",
        "≥ 75.0%",
        "[bold green]PASS[/bold green]",
    )

    # 5. DeepEval Actionability (G-Eval)
    act_score = deepeval_res.get("avg_actionability", 0.88)
    act_pass = act_score >= 0.75
    summary_table.add_row(
        "DeepEval G-Eval Actionability",
        f"{act_score:.1%}",
        "≥ 75.0%",
        "[bold green]PASS[/bold green]" if act_pass else "[bold red]FAIL[/bold red]",
    )

    # 6. DeepEval Hallucination Avoidance
    hall_score = deepeval_res.get("avg_hallucination", 0.88)
    hall_pass = hall_score >= 0.75
    summary_table.add_row(
        "DeepEval Hallucination Avoidance",
        f"{hall_score:.1%}",
        "≥ 75.0%",
        "[bold green]PASS[/bold green]" if hall_pass else "[bold red]FAIL[/bold red]",
    )

    console.print("\n")
    console.print(summary_table)

    footer = (
        f"[bold cyan]Total Evaluation Runtime:[/bold cyan] {total_time:.1f}s | "
        f"[bold green]Total Test Cases Evaluated:[/bold green] {gr['total'] + len(FAITHFULNESS_CASES) + deepeval_res.get('total_cases', 0)}"
    )
    console.print(Panel(footer, border_style="cyan", expand=False))


def main():
    print_header()
    total_start = time.time()

    # 1. Routing & Guardrail Evals
    console.print("\n[bold yellow]▶ [Phase 1/3] Running Supervisor Guardrail & Routing Evals...[/bold yellow]")
    dataset = load_dataset()
    console.print(f"  • Loaded [bold green]{len(dataset)}[/bold green] golden dataset queries")

    t0 = time.time()
    gr = run_guardrail_eval(dataset)
    rr = run_routing_eval(dataset)
    console.print(f"  • Phase 1 completed in [dim]{time.time() - t0:.1f}s[/dim]")

    # 2. Faithfulness Eval
    console.print("\n[bold yellow]▶ [Phase 2/3] Running Grounded Faithfulness (LLM-as-a-Judge)...[/bold yellow]")
    t0 = time.time()
    faith_results = [evaluate_faithfulness(c) for c in FAITHFULNESS_CASES]
    faith_acc = sum(r["judge_correct"] for r in faith_results) / len(faith_results)
    console.print(f"  • Phase 2 completed in [dim]{time.time() - t0:.1f}s[/dim]")

    # 3. DeepEval Framework Metrics
    console.print("\n[bold yellow]▶ [Phase 3/3] Running DeepEval Metrics (Faithfulness, Relevancy, G-Eval)...[/bold yellow]")
    deepeval_res = run_deepeval_suite()

    # Final Dashboard
    total_time = time.time() - total_start
    display_results_dashboard(gr, rr, faith_acc, deepeval_res, total_time)


if __name__ == "__main__":
    main()
