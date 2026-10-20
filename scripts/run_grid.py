"""Full small grid. Closed-form moments; no data download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb1.experiment import run_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--out", type=str, default="results/grid")
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    summary = run_grid(out, seeds=seeds, n_train=args.n_train, n_test=args.n_test)
    print(f"elapsed_sec={summary['run']['elapsed_sec']}")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
