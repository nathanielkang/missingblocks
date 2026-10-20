"""Code-alignment runs after the influence rewrite. Does not edit printed cells.

One process. New directories only, under results/code_align_2026-10-20/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb1.design import (  # noqa: E402
    BETA_TRUE,
    DELTA_C,
    IDX_A,
    IDX_B,
    IDX_C,
    N_FEATURES,
    generate_site,
    generate_split,
    layout_sites,
    phi,
)
from sb1.estimators import fit_gmm, fit_oracle, fit_transport  # noqa: E402
from sb1.experiment import run_grid  # noqa: E402

SEEDS = list(range(100))
OUT = ROOT / "results" / "code_align_2026-10-20"
U0 = 0.25


def _l2(betas: list[np.ndarray], target: np.ndarray) -> float:
    errs = [float(np.linalg.norm(beta - target)) for beta in betas]
    return float(np.sqrt(np.mean(np.square(errs))))


def _shift_block(sample, columns: np.ndarray, delta_rows: np.ndarray) -> None:
    features = sample.X.copy()
    outcome = sample.y.copy()
    features[:, columns] = features[:, columns] + delta_rows
    outcome = outcome + delta_rows @ BETA_TRUE[columns]
    sample.X = features
    sample.y = outcome


def _power(samples) -> None:
    """Recipient-half failure at site 0 only. Donors of that block are untouched."""
    sample = next(item for item in samples if item.site_id == 0)
    delta = phi(sample.X[:, IDX_A]) @ DELTA_C
    _shift_block(sample, IDX_C, delta)


def _column_shift(samples) -> None:
    """Outcome shift Xt^T u with u on the intercept, carried by one missing block."""
    for sample in samples:
        if not sample.observes_anchor:
            continue
        if sample.pattern == "AC":
            columns = IDX_B
        else:
            columns = IDX_C
        direction = BETA_TRUE[columns]
        step = (U0 / float(direction @ direction)) * direction
        delta = np.broadcast_to(step, (sample.n, step.shape[0])).copy()
        if abs(float(delta[0] @ direction) - U0) > 1e-8:
            raise RuntimeError("The column shift does not reproduce the intercept direction.")
        _shift_block(sample, columns, delta)


def _summarize_violation(name: str, transform) -> dict:
    betas = []
    oracles = []
    moments = []
    designs = []
    j_reject = []
    combined = []
    wald = {"B_given_A": [], "C_given_A": []}
    meta = {}
    for seed in SEEDS:
        train = generate_split(2000, seed, "transport_correct", "train")
        transform(train)
        amap, tests = fit_transport(train)
        fit = fit_gmm(train, amap)
        oracle = fit_oracle(train)
        betas.append(fit.beta.copy())
        oracles.append(oracle.beta.copy())
        moment = fit.moment_cross - fit.design @ BETA_TRUE
        moments.append(moment)
        designs.append(fit.design.copy())
        n_stats = len(tests)
        cutoff = 0.05 / (n_stats + 1)
        j_reject.append(int(fit.j_pvalue < 0.05))
        combined.append(
            int(fit.j_pvalue < cutoff or any(item.p_value < cutoff for item in tests))
        )
        for item in tests:
            wald[item.map_name].append(int(item.p_value < 0.05))
        meta = {
            "R": int(fit.n_moments),
            "p": int(N_FEATURES),
            "L": int(n_stats),
            "j_df": int(fit.j_df),
            "stacked_sites": list(fit.stacked_site_ids),
            "bonferroni_cutoff": cutoff,
        }
    mean_moment = np.mean(np.stack(moments, axis=0), axis=0)
    mean_design = np.mean(np.stack(designs, axis=0), axis=0)
    coef, *_ = np.linalg.lstsq(mean_design, mean_moment, rcond=None)
    residual = mean_moment - mean_design @ coef
    u = np.zeros(N_FEATURES)
    u[0] = U0
    target = BETA_TRUE + u if name == "column" else BETA_TRUE
    report = {
        "name": name,
        "seeds": len(SEEDS),
        "n_per_site": 2000,
        **meta,
        "gmm_l2_vs_beta0": _l2(betas, BETA_TRUE),
        "gmm_l2_vs_beta0_plus_u": _l2(betas, BETA_TRUE + u),
        "oracle_l2_vs_beta0": _l2(oracles, BETA_TRUE),
        "j_rejection_rate_05": float(np.mean(j_reject)),
        "combined_bonferroni_rejection_rate": float(np.mean(combined)),
        "wald_rejection_rate_05": {key: float(np.mean(values)) for key, values in wald.items()},
        "mean_moment_l2": float(np.linalg.norm(mean_moment)),
        "moment_onto_design_coef": [float(value) for value in coef],
        "moment_residual_l2": float(np.linalg.norm(residual)),
        "u": [float(value) for value in u],
        "u_l2": float(np.linalg.norm(u)),
        "reference_target": "beta0_plus_u" if name == "column" else "beta0",
        "reference_l2": _l2(betas, target),
    }
    path = OUT / f"jn_{name}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _site_counts(n_total: int, n_sites: int) -> list[int]:
    base, remainder = divmod(n_total, n_sites)
    return [base + (1 if index < remainder else 0) for index in range(n_sites)]


def _run_donor_design(name: str, counts: list[int], skip_outcome: tuple[int, ...]) -> dict:
    specs = layout_sites("three")
    if len(counts) != len(specs):
        raise RuntimeError("Row counts do not match the three-donor layout.")
    betas = []
    oracles = []
    for seed in SEEDS:
        train = [
            generate_site(spec, n, seed, "transport_correct", "train")
            for spec, n in zip(specs, counts)
        ]
        amap, _ = fit_transport(train)
        fit = fit_gmm(train, amap, skip_outcome=skip_outcome)
        oracle = fit_oracle(train)
        betas.append(fit.beta.copy())
        oracles.append(oracle.beta.copy())
    report = {
        "name": name,
        "seeds": len(SEEDS),
        "regime": "transport_correct",
        "site_ids": [int(spec.site_id) for spec in specs],
        "rows_per_site": counts,
        "total_rows": int(sum(counts)),
        "skip_outcome": list(skip_outcome),
        "gmm_l2_vs_beta0": _l2(betas, BETA_TRUE),
        "oracle_l2_vs_beta0": _l2(oracles, BETA_TRUE),
    }
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _cell(summary: dict, regime: str, method: str) -> float:
    return float(summary["regimes"][regime]["methods"][method]["vs_true"]["l2_rmse"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shared = run_grid(
        OUT / "shared_n2000",
        seeds=SEEDS,
        n_train=2000,
        n_test=2000,
        skip_loso=True,
    )
    common = {}
    for n in (1000, 5000):
        common[n] = run_grid(
            OUT / f"commonlaw_n{n}",
            seeds=SEEDS,
            n_train=n,
            n_test=n,
            regimes=("common_law",),
            skip_loso=True,
        )
    power = _summarize_violation("power", _power)
    column = _summarize_violation("column", _column_shift)
    two_donor_rows = 7 * 2000
    specs = layout_sites("three")
    fixed_counts = _site_counts(two_donor_rows, len(specs))
    fixed = _run_donor_design("donors_fixed_n", fixed_counts, ())
    gram_only = _run_donor_design(
        "donors_gram_only",
        [2000] * len(specs),
        (7, 8),
    )
    null = shared["regimes"]["transport_correct"]
    report = {
        "shared_n2000": {
            "elapsed_sec": shared["run"]["elapsed_sec"],
            "transport_correct": {
                method: _cell(shared, "transport_correct", method)
                for method in (
                    "pooled_oracle",
                    "federated_gmm",
                    "mcar_plugin",
                    "mean_impute_fedavg",
                )
            },
            "common_law": {
                method: _cell(shared, "common_law", method)
                for method in (
                    "pooled_oracle",
                    "federated_gmm",
                    "mcar_plugin",
                    "mean_impute_fedavg",
                )
            },
            "prediction_median_transport_correct": {
                method: shared["regimes"]["transport_correct"]["methods"][method]["prediction_mse"][
                    "overall_median"
                ]
                for method in ("federated_gmm", "mcar_plugin", "mean_impute_fedavg")
            },
            "wald_transport_correct": null["transport_rejection_rate_05"],
            "wald_transport_violated": shared["regimes"]["transport_violated"][
                "transport_rejection_rate_05"
            ],
            "j_rejection_rate_05": null["j_rejection_rate_05"],
            "combined_bonferroni_rejection_rate": null["combined_bonferroni_rejection_rate"],
            "j_df": null["j_df"],
            "n_moments": null["n_moments"],
            "j_size_kill": bool(null["j_rejection_rate_05"] is not None and null["j_rejection_rate_05"] > 0.10),
        },
        "common_law_nsweep": {
            str(n): {
                "elapsed_sec": item["run"]["elapsed_sec"],
                "pooled_oracle": _cell(item, "common_law", "pooled_oracle"),
                "federated_gmm": _cell(item, "common_law", "federated_gmm"),
                "mcar_plugin": _cell(item, "common_law", "mcar_plugin"),
                "mean_impute_fedavg": _cell(item, "common_law", "mean_impute_fedavg"),
            }
            for n, item in common.items()
        },
        "jn_power": power,
        "jn_column": column,
        "donors_fixed_n": fixed,
        "donors_gram_only": gram_only,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["shared_n2000"], indent=2))
    print("J size kill", report["shared_n2000"]["j_size_kill"])
    print("power J", power["j_rejection_rate_05"], "gmm", power["gmm_l2_vs_beta0"])
    print("column J", column["j_rejection_rate_05"], "gmm", column["gmm_l2_vs_beta0"], "vs u", column["gmm_l2_vs_beta0_plus_u"])
    print("fixed", fixed["gmm_l2_vs_beta0"], "oracle", fixed["oracle_l2_vs_beta0"], "rows", fixed["rows_per_site"])
    print("gram", gram_only["gmm_l2_vs_beta0"], "oracle", gram_only["oracle_l2_vs_beta0"])


if __name__ == "__main__":
    main()
