"""Run one replicate or a seed grid and write CSV/JSON summaries."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from sb1.design import (
    BETA_TRUE,
    COEF_NAMES,
    REGIMES,
    SITES,
    apply_mean_scale,
    design_record,
    generate_split,
    layout_sites,
)
from sb1.estimators import fit_all, prediction_mse

METHODS = (
    "pooled_oracle",
    "federated_gmm",
    "mcar_plugin",
    "mean_impute_fedavg",
)


def run_replicate(
    seed: int,
    regime: str,
    n_train: int,
    n_test: int,
    sites: tuple | None = None,
    skip_loso: bool = False,
) -> dict:
    chosen = SITES if sites is None else sites
    train = generate_split(n_train, seed, regime, "train", sites=chosen)
    test = generate_split(n_test, seed, regime, "test", sites=chosen)
    fits = fit_all(train)
    oracle_beta = fits["pooled_oracle"].beta
    coefficient_rows = []
    for method in METHODS:
        beta = fits[method].beta
        for index, name in enumerate(COEF_NAMES):
            coefficient_rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    "coef_index": index,
                    "coef_name": name,
                    "beta_hat": float(beta[index]),
                    "beta_oracle": float(oracle_beta[index]),
                    "beta_true": float(BETA_TRUE[index]),
                    "error_vs_oracle": float(beta[index] - oracle_beta[index]),
                    "error_vs_true": float(beta[index] - BETA_TRUE[index]),
                    "solve": fits[method].solve,
                }
            )
    prediction_rows = []
    for method in METHODS:
        for site in test:
            mse = prediction_mse(site, fits[method])
            if mse is None:
                continue
            prediction_rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "method": method,
                    "site_id": site.site_id,
                    "pattern": site.pattern,
                    "mse": mse,
                    "n_test": site.n,
                    "split": "full_train",
                }
            )
    loso_rows = []
    if not skip_loso:
        for held in chosen:
            subset = [site for site in train if site.site_id != held.site_id]
            held_test = next(site for site in test if site.site_id == held.site_id)
            loso_fits = fit_all(subset)
            for method in METHODS:
                mse = prediction_mse(held_test, loso_fits[method])
                if mse is None:
                    continue
                loso_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "method": method,
                        "held_out_site": held.site_id,
                        "pattern": held.pattern,
                        "mse": mse,
                        "n_test": held_test.n,
                    }
                )
    gmm_fit = fits["federated_gmm"]
    test_rows = []
    for item in gmm_fit.tests:
        test_rows.append(
            {
                "regime": regime,
                "seed": seed,
                "map_name": item.map_name,
                "site_a": item.site_a,
                "site_b": item.site_b,
                "wald_stat": item.statistic,
                "df": item.df,
                "p_value": item.p_value,
                "reject_05": int(item.reject_05),
                "n_donors": int(item.n_donors),
                "donor_ids": ",".join(str(site_id) for site_id in item.donor_ids),
            }
        )
    n_stats = len(gmm_fit.tests)
    bonferroni = 0.05 / (n_stats + 1) if n_stats else 0.05
    j_p = gmm_fit.j_pvalue
    overid_rows = [
        {
            "regime": regime,
            "seed": seed,
            "j_stat": gmm_fit.j_stat,
            "j_df": gmm_fit.j_df,
            "j_pvalue": j_p,
            "reject_05": int(j_p is not None and j_p < 0.05),
            "n_moments": gmm_fit.n_moments,
            "n_donor_stats": n_stats,
            "bonferroni_cutoff": bonferroni,
            "reject_j_bonferroni": int(j_p is not None and j_p < bonferroni),
            "reject_combined_bonferroni": int(
                (j_p is not None and j_p < bonferroni)
                or any(item.p_value < bonferroni for item in gmm_fit.tests)
            ),
            "stacked_sites": ",".join(str(site_id) for site_id in gmm_fit.stacked_site_ids),
        }
    ]
    return {
        "coefficients": coefficient_rows,
        "prediction": prediction_rows,
        "loso": loso_rows,
        "transport": test_rows,
        "overid": overid_rows,
        "solves": {method: fits[method].solve for method in METHODS},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _l2(rows: list[dict], method: str, regime: str, key: str) -> float:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        if row["regime"] != regime or row["method"] != method:
            continue
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
    if not by_seed:
        return float("nan")
    norms = [float(np.linalg.norm(values)) for values in by_seed.values()]
    return float(np.sqrt(np.mean(np.square(norms))))


def _coef_vectors(rows: list[dict], method: str, regime: str, key: str) -> dict:
    buckets = {name: [] for name in COEF_NAMES}
    for row in rows:
        if row["regime"] != regime or row["method"] != method:
            continue
        buckets[row["coef_name"]].append(float(row[key]))
    bias = []
    rmse = []
    for name in COEF_NAMES:
        values = np.asarray(buckets[name], dtype=float)
        bias.append(float(np.mean(values)))
        rmse.append(float(np.sqrt(np.mean(np.square(values)))))
    return {
        "coef_names": list(COEF_NAMES),
        "bias": bias,
        "rmse": rmse,
    }


def _mean_mse(rows: list[dict], method: str, regime: str, site_key: str | None = None) -> dict:
    per_site: dict[int, list[float]] = {}
    for row in rows:
        if row["regime"] != regime or row["method"] != method:
            continue
        site_id = int(row["site_id"] if site_key is None else row[site_key])
        per_site.setdefault(site_id, []).append(float(row["mse"]))
    site_means = {str(site_id): float(np.mean(values)) for site_id, values in sorted(per_site.items())}
    flat = [value for values in per_site.values() for value in values]
    overall = float(np.mean(flat)) if flat else None
    median = float(np.median(flat)) if flat else None
    return {"overall_mean": overall, "overall_median": median, "by_site": site_means}


def summarize(collected: dict, meta: dict) -> dict:
    coefficients = collected["coefficients"]
    regimes = {}
    for regime in meta["regimes"]:
        methods = {}
        for method in METHODS:
            methods[method] = {
                "vs_oracle": {
                    **_coef_vectors(coefficients, method, regime, "error_vs_oracle"),
                    "l2_rmse": _l2(coefficients, method, regime, "error_vs_oracle"),
                },
                "vs_true": {
                    **_coef_vectors(coefficients, method, regime, "error_vs_true"),
                    "l2_rmse": _l2(coefficients, method, regime, "error_vs_true"),
                },
                "prediction_mse": _mean_mse(collected["prediction"], method, regime),
                "loso_mse": _mean_mse(
                    collected["loso"], method, regime, site_key="held_out_site"
                ),
            }
        rejection = {}
        for map_name in ("B_given_A", "C_given_A"):
            flags = [
                int(row["reject_05"])
                for row in collected["transport"]
                if row["regime"] == regime and row["map_name"] == map_name
            ]
            rejection[map_name] = float(np.mean(flags)) if flags else None
        overid_rows = [
            row for row in collected.get("overid", []) if row["regime"] == regime
        ]
        def _rate(key: str) -> float | None:
            if not overid_rows:
                return None
            return float(np.mean([int(row[key]) for row in overid_rows]))

        regimes[regime] = {
            "methods": methods,
            "transport_rejection_rate_05": rejection,
            "j_rejection_rate_05": _rate("reject_05"),
            "combined_bonferroni_rejection_rate": _rate("reject_combined_bonferroni"),
            "j_df": None if not overid_rows else int(overid_rows[0]["j_df"]),
            "n_moments": None if not overid_rows else int(overid_rows[0]["n_moments"]),
        }
    return {
        "design": meta.get("design", design_record()),
        "run": {key: value for key, value in meta.items() if key != "design"},
        "regimes": regimes,
    }


def run_grid(
    out_dir: Path,
    seeds: list[int],
    n_train: int,
    n_test: int,
    regimes: tuple[str, ...] = REGIMES,
    layout: str = "two",
    mean_scale: float = 1.0,
    skip_loso: bool = False,
) -> dict:
    started = time.perf_counter()
    sites = apply_mean_scale(layout_sites(layout), mean_scale)
    collected = {"coefficients": [], "prediction": [], "loso": [], "transport": [], "overid": []}
    solves = []
    for regime in regimes:
        for seed in seeds:
            replicate = run_replicate(
                seed,
                regime,
                n_train,
                n_test,
                sites=sites,
                skip_loso=skip_loso,
            )
            for key in collected:
                collected[key].extend(replicate[key])
            solves.append({"regime": regime, "seed": seed, **replicate["solves"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "coefficient_errors.csv", collected["coefficients"])
    _write_csv(out_dir / "prediction_mse.csv", collected["prediction"])
    _write_csv(out_dir / "loso_mse.csv", collected["loso"])
    _write_csv(out_dir / "transport_tests.csv", collected["transport"])
    _write_csv(out_dir / "overid.csv", collected["overid"])
    meta = {
        "seeds": list(seeds),
        "n_train_per_site": n_train,
        "n_test_per_site": n_test,
        "n_sites": len(sites),
        "regimes": list(regimes),
        "methods": list(METHODS),
        "layout": layout,
        "mean_scale": float(mean_scale),
        "skip_loso": bool(skip_loso),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "solves": solves,
        "design": design_record(sites, mean_scale=mean_scale, layout=layout),
    }
    summary = summarize(collected, meta)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
