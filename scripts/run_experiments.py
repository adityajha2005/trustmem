#!/usr/bin/env python3
"""
Experiment helpers — load .env.local and run sweeps / exports.

Usage:
  python scripts/run_experiments.py ablation [--seed 42]
  python scripts/run_experiments.py multiseed --seeds 42,43,44
  python scripts/run_experiments.py plots --run-dir experiments/runs/ablations/ablation_seed42
  python scripts/run_experiments.py techniques [--seed 42]
  python scripts/run_experiments.py demo [--retrieval-only]
  python scripts/run_experiments.py llm-eval [--suite complex] [--seed 42]
  python scripts/run_experiments.py check-env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when invoked as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_check_env(_args: argparse.Namespace | None = None) -> int:
    from utils.config import reload_config

    cfg = reload_config()
    print("Retrieval mode:", cfg.retrieval_mode)
    print("Offline retrieval:", cfg.offline_retrieval)
    print("Embedding model:", cfg.embedding_model)
    print("Reranker enabled:", cfg.use_reranker)
    print("Reranker model:", cfg.reranker_model)
    print("Runs dir:", cfg.experiments_runs_dir)
    print("OpenAI key set:", cfg.public_summary()["has_openai_key"])
    if cfg.offline_retrieval:
        print("\nWARNING: MEMORY_OFFLINE_RETRIEVAL=1 — not using real embeddings.")
        print("Set MEMORY_OFFLINE_RETRIEVAL=0 in .env.local for real-model runs.")
    else:
        print("\nReal semantic retrieval is ENABLED.")
        print(
            "If Python crashes (Aborted) on macOS during benchmarks, set "
            "MEMORY_OFFLINE_RETRIEVAL=1 or use Python 3.12/3.13 — see README Troubleshooting."
        )

    from baselines.runtime import warn_if_risky_benchmark_env

    for line in warn_if_risky_benchmark_env():
        print(f"WARNING: {line}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from experiments.runner import main as runner_main

    argv = ["--seed", str(args.seed)]
    if args.ci:
        argv.append("--ci")
    if args.no_plots:
        argv.append("--no-plots")
    if args.runs_dir:
        argv.extend(["--runs-dir", args.runs_dir])
    if args.variant:
        for v in args.variant:
            argv.extend(["--variant", v])
    return runner_main(argv)


def cmd_multiseed(args: argparse.Namespace) -> int:
    from experiments.multiseed import main as multiseed_main

    argv = ["--seeds", args.seeds]
    if args.no_plots:
        argv.append("--no-plots")
    if args.runs_dir:
        argv.extend(["--runs-dir", args.runs_dir])
    return multiseed_main(argv)


def cmd_techniques(args: argparse.Namespace) -> int:
    from experiments.technique_benchmark import main as technique_main

    argv = ["--seed", str(args.seed)]
    if args.no_plots:
        argv.append("--no-plots")
    if args.runs_dir:
        argv.extend(["--runs-dir", args.runs_dir])
    if args.technique:
        for t in args.technique:
            argv.extend(["--technique", t])
    if args.scenario:
        for s in args.scenario:
            argv.extend(["--scenario", s])
    return technique_main(argv)


def cmd_llm_eval(args: argparse.Namespace) -> int:
    from evaluation.llm_eval_runner import main as llm_main
    from utils.config import reload_config

    reload_config()
    argv: list[str] = ["--seed", str(args.seed)]
    if args.suite:
        argv.extend(["--suite", args.suite])
    if args.model:
        argv.extend(["--model", args.model])
    if args.technique:
        for t in args.technique:
            argv.extend(["--technique", t])
    if args.runs_dir:
        argv.extend(["--runs-dir", args.runs_dir])
    return llm_main(argv)


def cmd_demo(args: argparse.Namespace) -> int:
    from demo.e2e_llm import main as demo_main

    argv: list[str] = []
    if args.query:
        argv.extend(["--query", args.query])
    if args.technique:
        for t in args.technique:
            argv.extend(["--technique", t])
    if args.retrieval_only:
        argv.append("--retrieval-only")
    if args.model:
        argv.extend(["--model", args.model])
    return demo_main(argv)


def cmd_plots(args: argparse.Namespace) -> int:
    from experiments.visualize import main as viz_main

    argv = ["--run-dir", args.run_dir]
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    return viz_main(argv)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collaborative memory experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-env", help="print effective config from .env.local")
    p_check.set_defaults(func=cmd_check_env)

    p_ab = sub.add_parser("ablation", help="full ablation sweep + export + plots")
    p_ab.add_argument("--seed", type=int, default=42)
    p_ab.add_argument("--ci", action="store_true")
    p_ab.add_argument("--no-plots", action="store_true")
    p_ab.add_argument("--runs-dir", default=None)
    p_ab.add_argument("--variant", action="append")
    p_ab.set_defaults(func=cmd_ablation)

    p_ms = sub.add_parser("multiseed", help="run ablation across multiple seeds")
    p_ms.add_argument("--seeds", default="42,43,44")
    p_ms.add_argument("--no-plots", action="store_true")
    p_ms.add_argument("--runs-dir", default=None)
    p_ms.set_defaults(func=cmd_multiseed)

    p_tech = sub.add_parser(
        "techniques",
        help="benchmark mem0 vs simple RAG vs collaborative + plots",
    )
    p_tech.add_argument("--seed", type=int, default=42)
    p_tech.add_argument("--no-plots", action="store_true")
    p_tech.add_argument("--runs-dir", default=None)
    p_tech.add_argument("--technique", action="append")
    p_tech.add_argument("--scenario", action="append")
    p_tech.set_defaults(func=cmd_techniques)

    p_llm = sub.add_parser(
        "llm-eval",
        help="LLM eval: retrieve + xAI answer + judge (complex=10 tasks × 7 techniques)",
    )
    p_llm.add_argument("--seed", type=int, default=42)
    p_llm.add_argument(
        "--suite",
        choices=("basic", "complex"),
        default="complex",
        help="basic=2 tasks, complex=10 hard multi-agent tasks (default)",
    )
    p_llm.add_argument("--model", default=None)
    p_llm.add_argument("--technique", action="append")
    p_llm.add_argument("--runs-dir", default=None)
    p_llm.set_defaults(func=cmd_llm_eval)

    p_demo = sub.add_parser("demo", help="E2E retrieval + xAI Grok for each technique")
    p_demo.add_argument("--query", default=None)
    p_demo.add_argument("--technique", action="append")
    p_demo.add_argument("--retrieval-only", action="store_true")
    p_demo.add_argument("--model", default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_plots = sub.add_parser("plots", help="regenerate figures from saved summary JSON")
    p_plots.add_argument("--run-dir", required=True)
    p_plots.add_argument("--run-id", default=None)
    p_plots.set_defaults(func=cmd_plots)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
