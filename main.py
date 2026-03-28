from __future__ import annotations

import argparse
import sys

from src.experiments import run_baseline, run_ssfl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capstone FL experiment launcher")
    parser.add_argument(
        "mode",
        choices=["baseline", "ssfl"],
        help="Experiment mode to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forwarded_args = sys.argv[2:]
    if args.mode == "baseline":
        run_baseline.main(forwarded_args)
    else:
        run_ssfl.main(forwarded_args)


if __name__ == "__main__":
    main()
