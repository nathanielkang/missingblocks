"""Draw the site-by-block observation pattern used in the synthetic design."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb1.design import SITES  # noqa: E402


def main() -> None:
    blocks = ("A", "B", "C")
    grid = np.zeros((len(SITES), len(blocks)))
    ylabels = []
    for row, spec in enumerate(SITES):
        records = {
            "A": spec.pattern in {"AB", "AC", "A"},
            "B": spec.pattern in {"AB", "BC"},
            "C": spec.pattern in {"AC", "BC"},
        }
        for col, block in enumerate(blocks):
            grid[row, col] = 1.0 if records[block] else 0.0
        ylabels.append(f"site {spec.site_id}  {spec.pattern}")
    figure, axis = plt.subplots(figsize=(4.4, 5.2))
    axis.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(blocks)))
    axis.set_xticklabels([f"block {name}" for name in blocks])
    axis.set_yticks(range(len(SITES)))
    axis.set_yticklabels(ylabels)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            label = "recorded" if grid[row, col] == 1 else "absent"
            color = "white" if grid[row, col] == 1 else "#444444"
            axis.text(col, row, label, ha="center", va="center", color=color, fontsize=9)
    axis.set_xlabel("Feature block (two coordinates each)")
    axis.tick_params(length=0)
    figure.tight_layout()
    out = ROOT / "figures" / "block_pattern.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    plt.close(figure)
    print(out)


if __name__ == "__main__":
    main()
