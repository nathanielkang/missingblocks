"""Local smoke: two seeds, small n, then stop if the contrast is missing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb1.experiment import run_grid  # noqa: E402


def main() -> None:
    out = ROOT / "results" / "smoke"
    summary = run_grid(
        out_dir=out,
        seeds=[0, 1],
        n_train=800,
        n_test=800,
    )
    correct = summary["regimes"]["transport_correct"]["methods"]
    common = summary["regimes"]["common_law"]["methods"]
    violated = summary["regimes"]["transport_violated"]
    gmm = correct["federated_gmm"]["vs_true"]["l2_rmse"]
    plugin = correct["mcar_plugin"]["vs_true"]["l2_rmse"]
    fedavg = correct["mean_impute_fedavg"]["vs_true"]["l2_rmse"]
    oracle = correct["pooled_oracle"]["vs_true"]["l2_rmse"]
    plugin_common = common["mcar_plugin"]["vs_true"]["l2_rmse"]
    rejection = violated["transport_rejection_rate_05"]
    print(f"oracle l2 vs true (shift, correct transport): {oracle:.4f}")
    print(f"gmm    l2 vs true: {gmm:.4f}")
    print(f"plugin l2 vs true: {plugin:.4f}")
    print(f"fedavg l2 vs true: {fedavg:.4f}")
    print(f"plugin l2 vs true under common law: {plugin_common:.4f}")
    print(f"violated-transport rejection: {rejection}")
    problems = []
    if not np_finite(oracle, gmm, plugin, fedavg, plugin_common):
        problems.append("non-finite coefficient error")
    if oracle > 0.35:
        problems.append(f"oracle error {oracle:.3f} is too large for n=800")
    if gmm > 0.85:
        problems.append(f"gmm error {gmm:.3f} is too large for n=800")
    if not (gmm < plugin):
        problems.append("gmm is not closer to the truth than the common-law plug-in")
    if not (gmm < fedavg):
        problems.append("gmm is not closer to the truth than mean-imputation FedAvg")
    if not (plugin_common < plugin):
        problems.append("plug-in is not worse under site-specific laws than under a common law")
    if rejection.get("B_given_A") != 1.0 or rejection.get("C_given_A") != 1.0:
        problems.append(f"violated transport was not rejected on both maps: {rejection}")
    if problems:
        print("SMOKE FAIL")
        for item in problems:
            print(f"- {item}")
        raise SystemExit(1)
    print("SMOKE PASS")
    print(out / "summary.json")


def np_finite(*values: float) -> bool:
    import math

    return all(math.isfinite(value) for value in values)


if __name__ == "__main__":
    main()
