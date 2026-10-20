"""Predeclared SB1 expansions. GAMMA, BETA, and the locked site means are not arguments.

Campaigns
  nsweep   n in {500, 1000, 2000, 5000}, two donors, mean scale 1
  shift    scales weak 0.5 / medium 1 / strong 2, n = 2000, two donors
  donors   one / two / three donors per conditional mean, n = 2000, scale 1

Leave-one-site-out is not run. It is not a paper endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb1.design import SHIFT_SCALES  # noqa: E402
from sb1.experiment import run_grid  # noqa: E402

N_SEEDS = 100


def _jobs(campaign: str) -> list[dict]:
    if campaign == "nsweep":
        return [
            {"name": f"n{n}", "n": n, "layout": "two", "scale": 1.0}
            for n in (500, 1000, 2000, 5000)
        ]
    if campaign == "shift":
        return [
            {"name": name, "n": 2000, "layout": "two", "scale": SHIFT_SCALES[name]}
            for name in ("weak", "medium", "strong")
        ]
    if campaign == "donors":
        return [
            {"name": name, "n": 2000, "layout": name, "scale": 1.0}
            for name in ("one", "two", "three")
        ]
    raise KeyError(campaign)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=("nsweep", "shift", "donors"), required=True)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-root", type=str, default="")
    args = parser.parse_args()
    root_name = {
        "nsweep": "results/paper_nsweep",
        "shift": "results/paper_shift",
        "donors": "results/paper_donors",
    }[args.campaign]
    out_root = Path(args.out_root) if args.out_root else ROOT / root_name
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for job in _jobs(args.campaign):
        out = out_root / job["name"]
        print(
            f"START {args.campaign}/{job['name']} n={job['n']} "
            f"layout={job['layout']} scale={job['scale']} seeds={len(seeds)}",
            flush=True,
        )
        summary = run_grid(
            out,
            seeds=seeds,
            n_train=job["n"],
            n_test=job["n"],
            layout=job["layout"],
            mean_scale=job["scale"],
            skip_loso=True,
        )
        print(
            f"DONE {args.campaign}/{job['name']} elapsed_sec={summary['run']['elapsed_sec']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
