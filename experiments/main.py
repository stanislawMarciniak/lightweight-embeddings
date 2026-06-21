from __future__ import annotations

import argparse

from config import DEFAULT_CONFIG
from pipeline import run_benchmark, run_train
from registry import all_model_names, resolve_names


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--all", action="store_true", help="Apply to all registered models.")
    p.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Model names or aliases (space-separated). Example: --models custom mpnet bge",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and benchmark STS embedding models.\n\n"
            "With no subcommand, runs the full pipeline: train ALL models, then "
            "benchmark ALL models.\n\n"
            f"Available models: {', '.join(all_model_names())}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No subcommand => full pipeline (train all + benchmark all).
    sub = parser.add_subparsers(dest="command", required=False)

    p_train = sub.add_parser("train", help="Train models and save checkpoints.")
    _add_selection(p_train)

    p_bench = sub.add_parser("benchmark", help="Evaluate models and write results/plots.")
    _add_selection(p_bench)
    p_bench.add_argument(
        "--verbose",
        action="store_true",
        help="Show every benchmark stage and a per-stage timing breakdown.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # No subcommand: full pipeline over all models (train, then benchmark).
    if args.command is None:
        names = resolve_names(None)
        print("=== Full pipeline: training all models ===")
        run_train(names, DEFAULT_CONFIG)
        print("\n=== Full pipeline: benchmarking all models ===")
        run_benchmark(names, DEFAULT_CONFIG, verbose=False)
        return

    if not args.all and not args.models:
        raise SystemExit("Select models with --all or --models NAME [NAME ...].")
    names = resolve_names(None if args.all else args.models)

    if args.command == "train":
        run_train(names, DEFAULT_CONFIG)
    elif args.command == "benchmark":
        run_benchmark(names, DEFAULT_CONFIG, verbose=args.verbose)


if __name__ == "__main__":
    main()
