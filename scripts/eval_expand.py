"""Score SB1 expansion ledgers against the 2026-10-20 win bars.

Reads summary.json files. Does not retune anything. Prints one JSON object.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def l2(summary: dict, regime: str, method: str) -> float:
    return float(summary["regimes"][regime]["methods"][method]["vs_true"]["l2_rmse"])


def wald(summary: dict, regime: str, map_name: str) -> float | None:
    value = summary["regimes"][regime]["transport_rejection_rate_05"][map_name]
    if value is None:
        return None
    return float(value)


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def solves_ok(summary: dict) -> bool:
    allowed = {"solve", "solve/solve"}
    for row in summary["run"]["solves"]:
        for method, label in row.items():
            if method in {"regime", "seed"}:
                continue
            if label not in allowed:
                return False
    return True


def coef_checks(summary: dict, n: int) -> dict:
    gmm = l2(summary, "transport_correct", "federated_gmm")
    oracle = l2(summary, "transport_correct", "pooled_oracle")
    plugin = l2(summary, "transport_correct", "mcar_plugin")
    fed = l2(summary, "transport_correct", "mean_impute_fedavg")
    cap = 0.08 if n < 2000 else 0.05
    checks = {
        "gmm_l2": gmm,
        "oracle_l2": oracle,
        "plugin_l2": plugin,
        "fedavg_l2": fed,
        "gmm_cap": cap,
        "gmm_under_cap": finite(gmm) and gmm <= cap,
        "gmm_le_4x_oracle": finite(gmm) and finite(oracle) and oracle > 0 and gmm <= 4.0 * oracle,
        "plugin_ge_0.50": finite(plugin) and plugin >= 0.50,
        "fedavg_ge_0.50": finite(fed) and fed >= 0.50,
        "solves_ok": solves_ok(summary),
    }
    checks["coef_pass"] = all(
        checks[key]
        for key in (
            "gmm_under_cap",
            "gmm_le_4x_oracle",
            "plugin_ge_0.50",
            "fedavg_ge_0.50",
            "solves_ok",
        )
    )
    return checks


def wald_checks(summary: dict, expect_test: bool) -> dict:
    correct = [wald(summary, "transport_correct", name) for name in ("B_given_A", "C_given_A")]
    violated = [wald(summary, "transport_violated", name) for name in ("B_given_A", "C_given_A")]
    if not expect_test:
        absent = all(value is None for value in correct + violated)
        return {
            "expect_test": False,
            "correct": correct,
            "violated": violated,
            "wald_pass": absent,
        }
    size_ok = all(finite(value) and value <= 0.12 for value in correct)
    power_ok = all(finite(value) and value >= 0.85 for value in violated)
    return {
        "expect_test": True,
        "correct": correct,
        "violated": violated,
        "size_ok": size_ok,
        "power_ok": power_ok,
        "wald_pass": size_ok and power_ok,
    }


def main() -> None:
    base = ROOT / "results"
    report: dict = {"campaigns": {}}

    nsweep = {}
    for n in (500, 1000, 2000, 5000):
        summary = load(base / "paper_nsweep" / f"n{n}" / "summary.json")
        item = {"coef": coef_checks(summary, n), "wald": wald_checks(summary, True)}
        item["pass"] = item["coef"]["coef_pass"] and item["wald"]["wald_pass"]
        nsweep[str(n)] = item
    report["campaigns"]["nsweep"] = nsweep

    shift = {}
    for name in ("weak", "medium", "strong"):
        summary = load(base / "paper_shift" / name / "summary.json")
        item = {"coef": coef_checks(summary, 2000), "wald": wald_checks(summary, True)}
        item["pass_cell"] = item["coef"]["coef_pass"] and item["wald"]["wald_pass"]
        shift[name] = item
    plugin_m = shift["medium"]["coef"]["plugin_l2"]
    plugin_s = shift["strong"]["coef"]["plugin_l2"]
    gmm_near = all(
        shift[name]["coef"]["gmm_under_cap"] and shift[name]["coef"]["gmm_le_4x_oracle"]
        for name in ("weak", "medium", "strong")
    )
    stronger_plugin = finite(plugin_s) and finite(plugin_m) and (
        plugin_s + 1e-12 >= plugin_m or plugin_s >= 0.50
    )
    shift_meta = {
        "plugin_medium": plugin_m,
        "plugin_strong": plugin_s,
        "stronger_plugin_worse_or_still_bad": stronger_plugin,
        "gmm_near_oracle_all_scales": gmm_near,
    }
    report["campaigns"]["shift"] = {"cells": shift, "comparison": shift_meta}

    donors = {}
    for name, expect in (("one", False), ("two", True), ("three", True)):
        summary = load(base / "paper_donors" / name / "summary.json")
        item = {"coef": coef_checks(summary, 2000), "wald": wald_checks(summary, expect)}
        item["pass_cell"] = item["coef"]["coef_pass"] and item["wald"]["wald_pass"]
        donors[name] = item
    power_ge = True
    for index in (0, 1):
        two = donors["two"]["wald"]["violated"][index]
        three = donors["three"]["wald"]["violated"][index]
        if not (finite(two) and finite(three) and three + 1e-12 >= two):
            power_ge = False
    report["campaigns"]["donors"] = {
        "cells": donors,
        "three_power_ge_two": power_ge,
        "three_pass": donors["three"]["pass_cell"] and power_ge,
    }

    # Locked protocol must reproduce paper_s100 on the three copies of n=2000 / scale 1 / two donors.
    locked = load(base / "paper_s100" / "summary.json")
    copies = {
        "nsweep_n2000": load(base / "paper_nsweep" / "n2000" / "summary.json"),
        "shift_medium": load(base / "paper_shift" / "medium" / "summary.json"),
        "donors_two": load(base / "paper_donors" / "two" / "summary.json"),
    }
    match = {}
    for name, summary in copies.items():
        gaps = {}
        for method in ("pooled_oracle", "federated_gmm", "mcar_plugin", "mean_impute_fedavg"):
            gaps[method] = abs(
                l2(summary, "transport_correct", method) - l2(locked, "transport_correct", method)
            )
        for map_name in ("B_given_A", "C_given_A"):
            for regime in ("transport_correct", "transport_violated"):
                gaps[f"{regime}:{map_name}"] = abs(
                    (wald(summary, regime, map_name) or -1) - (wald(locked, regime, map_name) or -1)
                )
        match[name] = {"max_abs": max(gaps.values()), "gaps": gaps}
    report["locked_reproduction"] = match
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
