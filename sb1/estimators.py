"""Federated estimators that exchange aggregated moments only.

The GMM path never reads a column the site does not record. Comparison
estimators are the two procedures the design is meant to show failing under
site-specific covariate laws: mean-imputation with averaged local least
squares, and a component-wise second-moment plug-in that treats every site as
a draw from one common covariate law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from sb1.design import (
    IDX_A,
    IDX_B,
    IDX_C,
    N_BASIS,
    N_FEATURES,
    SiteSample,
    phi,
)


def _solve(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        return np.linalg.solve(left, right), "solve"
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(left, right, rcond=None)
        return coef, "lstsq"


def transported_design(
    view: np.ndarray,
    mask: np.ndarray,
    gamma_b: np.ndarray,
    gamma_c: np.ndarray,
) -> np.ndarray:
    """Regressor with the transported conditional mean in place of a missing block.

    ``view`` must already contain NaN on columns the site does not record.
    The anchor is required. A missing block is replaced by phi(anchor) @ gamma,
    which is a function of recorded columns only.
    """
    if not bool(mask[IDX_A].all()):
        raise ValueError("Outcome moments require the anchor block.")
    if not (bool(mask[IDX_B].all()) or bool(~mask[IDX_B].all())):
        raise ValueError("Block B must be wholly recorded or wholly missing.")
    anchor = view[:, IDX_A]
    if not np.isfinite(anchor).all():
        raise ValueError("Anchor columns are missing from the federated view.")
    basis = phi(anchor)
    n = view.shape[0]
    regressor = np.empty((n, N_FEATURES), dtype=float)
    regressor[:, 0] = 1.0
    regressor[:, IDX_A] = anchor
    if bool(mask[IDX_B].all()):
        observed = view[:, IDX_B]
        if not np.isfinite(observed).all():
            raise ValueError("Block B is marked recorded but is not finite.")
        regressor[:, IDX_B] = observed
    else:
        regressor[:, IDX_B] = basis @ gamma_b
    if bool(mask[IDX_C].all()):
        observed = view[:, IDX_C]
        if not np.isfinite(observed).all():
            raise ValueError("Block C is marked recorded but is not finite.")
        regressor[:, IDX_C] = observed
    else:
        regressor[:, IDX_C] = basis @ gamma_c
    if not np.isfinite(regressor).all():
        raise ValueError("Transported design is not finite.")
    return regressor


@dataclass
class TransportMap:
    gamma_b: np.ndarray
    gamma_c: np.ndarray
    donors_b: tuple[int, ...]
    donors_c: tuple[int, ...]
    solve_b: str
    solve_c: str


@dataclass
class TransportTest:
    map_name: str
    site_a: int
    site_b: int
    statistic: float
    df: int
    p_value: float
    reject_05: bool
    n_donors: int = 2
    donor_ids: tuple[int, ...] = ()


@dataclass
class FitResult:
    method: str
    beta: np.ndarray
    gamma_b: np.ndarray | None = None
    gamma_c: np.ndarray | None = None
    sigma: np.ndarray | None = None
    cross: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    solve: str = "solve"
    tests: tuple[TransportTest, ...] = ()
    j_stat: float | None = None
    j_df: int | None = None
    j_pvalue: float | None = None
    n_moments: int = 0
    stacked_site_ids: tuple[int, ...] = ()
    design: np.ndarray | None = None
    moment_cross: np.ndarray | None = None


def _basis_packets(sites: list[SiteSample], block: str) -> list[dict]:
    target = IDX_B if block == "B" else IDX_C
    packets = []
    for site in sites:
        if not site.observes_anchor or not bool(site.mask[target].all()):
            continue
        view = site.federated_view()
        anchor = view[:, IDX_A]
        outcome = view[:, target]
        basis = phi(anchor)
        packets.append(
            {
                "site_id": site.site_id,
                "n": site.n,
                "xtx": basis.T @ basis,
                "xty": basis.T @ outcome,
                "yty": outcome.T @ outcome,
            }
        )
    if not packets:
        raise RuntimeError(f"No donor co-observes block {block} with the anchor.")
    return packets


def _pooled_map(packets: list[dict]) -> tuple[np.ndarray, str]:
    xtx = sum(packet["xtx"] for packet in packets)
    xty = sum(packet["xty"] for packet in packets)
    gamma, how = _solve(xtx, xty)
    return gamma, how


def _theta_var(packet: dict) -> tuple[np.ndarray, np.ndarray]:
    gamma, _ = _solve(packet["xtx"], packet["xty"])
    # Residual second moment of the multivariate basis regression.
    sse = packet["yty"] - gamma.T @ packet["xty"]
    sse = 0.5 * (sse + sse.T)
    df_resid = packet["n"] - N_BASIS
    if df_resid <= 0:
        raise RuntimeError("Transport regression has no residual degrees of freedom.")
    scale = sse / df_resid
    gram_inv, _ = _solve(packet["xtx"], np.eye(N_BASIS))
    theta = gamma.ravel(order="F")
    variance = np.kron(scale, gram_inv)
    return theta, variance


def _wald_pair(left: dict, right: dict) -> TransportTest:
    theta_l, var_l = _theta_var(left)
    theta_r, var_r = _theta_var(right)
    diff = theta_l - theta_r
    variance = var_l + var_r
    variance = 0.5 * (variance + variance.T)
    statistic = float(diff @ np.linalg.solve(variance, diff))
    df = int(diff.shape[0])
    p_value = float(chi2.sf(statistic, df))
    return TransportTest(
        map_name="",
        site_a=int(left["site_id"]),
        site_b=int(right["site_id"]),
        statistic=statistic,
        df=df,
        p_value=p_value,
        reject_05=bool(p_value < 0.05),
    )


def _agreement_test(packets: list[dict], map_name: str) -> TransportTest:
    """Donor-agreement quadratic form over every donor of one map.

    Two donors stay on ``_wald_pair``. This path is only for three or more,
    where the chi-square degrees of freedom grow with the extra donor.
    """
    thetas = []
    variances = []
    ids = []
    for packet in packets:
        theta, variance = _theta_var(packet)
        thetas.append(theta)
        variances.append(0.5 * (variance + variance.T))
        ids.append(int(packet["site_id"]))
    precision_sum = np.zeros_like(variances[0])
    weighted = np.zeros_like(thetas[0])
    for theta, variance in zip(thetas, variances):
        solved = np.linalg.solve(variance, np.column_stack([theta, np.eye(theta.shape[0])]))
        solved_theta = solved[:, 0]
        precision = solved[:, 1:]
        precision_sum += precision
        weighted += solved_theta
    bar = np.linalg.solve(precision_sum, weighted)
    statistic = 0.0
    for theta, variance in zip(thetas, variances):
        diff = theta - bar
        statistic += float(diff @ np.linalg.solve(variance, diff))
    df = (len(packets) - 1) * int(thetas[0].shape[0])
    p_value = float(chi2.sf(statistic, df))
    ordered = tuple(ids)
    return TransportTest(
        map_name=map_name,
        site_a=int(ordered[0]),
        site_b=int(ordered[1]),
        statistic=statistic,
        df=df,
        p_value=p_value,
        reject_05=bool(p_value < 0.05),
        n_donors=len(packets),
        donor_ids=ordered,
    )


def fit_transport(sites: list[SiteSample]) -> tuple[TransportMap, tuple[TransportTest, ...]]:
    """Conditional means from co-observing donors, plus a pairwise Wald test."""
    packets_b = _basis_packets(sites, "B")
    packets_c = _basis_packets(sites, "C")
    gamma_b, how_b = _pooled_map(packets_b)
    gamma_c, how_c = _pooled_map(packets_c)
    tests = []
    if len(packets_b) == 2:
        test_b = _wald_pair(packets_b[0], packets_b[1])
        tests.append(
            TransportTest(
                "B_given_A",
                test_b.site_a,
                test_b.site_b,
                test_b.statistic,
                test_b.df,
                test_b.p_value,
                test_b.reject_05,
                2,
                (test_b.site_a, test_b.site_b),
            )
        )
    elif len(packets_b) > 2:
        tests.append(_agreement_test(packets_b, "B_given_A"))
    if len(packets_c) == 2:
        test_c = _wald_pair(packets_c[0], packets_c[1])
        tests.append(
            TransportTest(
                "C_given_A",
                test_c.site_a,
                test_c.site_b,
                test_c.statistic,
                test_c.df,
                test_c.p_value,
                test_c.reject_05,
                2,
                (test_c.site_a, test_c.site_b),
            )
        )
    elif len(packets_c) > 2:
        tests.append(_agreement_test(packets_c, "C_given_A"))
    amap = TransportMap(
        gamma_b=gamma_b,
        gamma_c=gamma_c,
        donors_b=tuple(int(packet["site_id"]) for packet in packets_b),
        donors_c=tuple(int(packet["site_id"]) for packet in packets_c),
        solve_b=how_b,
        solve_c=how_c,
    )
    return amap, tuple(tests)


@dataclass
class _BlockMap:
    """One transported block: dictionary h, pooled coefficient, donor rows."""

    name: str
    columns: np.ndarray
    gamma: np.ndarray
    donor_ids: tuple[int, ...]
    h_inv: np.ndarray


@dataclass
class _OutcomeSite:
    site_id: int
    n: int
    q: np.ndarray
    xt: np.ndarray
    y: np.ndarray
    h_by_map: dict[str, np.ndarray]
    offset: int = 0

    @property
    def width(self) -> int:
        return int(self.q.shape[1])


def _column_block(site: SiteSample, columns: np.ndarray) -> np.ndarray:
    if not bool(site.mask[columns].all()):
        raise ValueError(f"Site {site.site_id} does not record columns {columns.tolist()}.")
    block = site.X[:, columns]
    if not np.isfinite(block).all():
        raise ValueError(f"Site {site.site_id} has a non-finite recorded block.")
    return block


def _anchor_basis(site: SiteSample) -> np.ndarray:
    if not site.observes_anchor:
        raise ValueError(f"Site {site.site_id} does not record the anchor.")
    return phi(_column_block(site, IDX_A))


def _known_instrument(site: SiteSample) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Instrument q of recorded features and dictionaries, and the transported regressor.

    q does not depend on the conditional-mean coefficients. The manuscript's
    influence is derived for that case: every coordinate of the transported
    regressor is a linear function of q, with the coefficient in the map.
    """
    if not (bool(site.mask[IDX_B].all()) or bool(~site.mask[IDX_B].all())):
        raise ValueError(f"Site {site.site_id} records only part of block B.")
    if not (bool(site.mask[IDX_C].all()) or bool(~site.mask[IDX_C].all())):
        raise ValueError(f"Site {site.site_id} records only part of block C.")
    recorded = np.flatnonzero(site.mask)
    parts = [_column_block(site, recorded)]
    misses_nonanchor = (not bool(site.mask[IDX_B].all())) or (not bool(site.mask[IDX_C].all()))
    basis = _anchor_basis(site) if site.observes_anchor and misses_nonanchor else None
    if basis is not None:
        parts.append(basis)
    q = np.column_stack(parts)
    xt = np.zeros((site.n, N_FEATURES), dtype=float)
    xt[:, recorded] = site.X[:, recorded]
    return q, xt, basis


def _fill_transported(
    site: SiteSample,
    xt: np.ndarray,
    basis: np.ndarray | None,
    gamma_b: np.ndarray,
    gamma_c: np.ndarray,
) -> np.ndarray:
    if not site.observes_anchor:
        raise RuntimeError(
            f"Site {site.site_id} does not record the anchor, so the declared dictionaries are not functions of its records."
        )
    if not bool(site.mask[IDX_B].all()):
        if basis is None:
            raise RuntimeError(f"Site {site.site_id} misses block B without the anchor.")
        xt[:, IDX_B] = basis @ gamma_b
    if not bool(site.mask[IDX_C].all()):
        if basis is None:
            raise RuntimeError(f"Site {site.site_id} misses block C without the anchor.")
        xt[:, IDX_C] = basis @ gamma_c
    if not np.isfinite(xt).all():
        raise ValueError(f"Site {site.site_id} has a non-finite transported regressor.")
    return xt


def _map_inverse(raw_gram: np.ndarray, n_total: int) -> np.ndarray:
    """Inverse of the donor Gram divided by the federation sample size."""
    hat = raw_gram / float(n_total)
    solved, _ = _solve(hat, np.eye(hat.shape[0]))
    return solved


def _build_block_map(
    name: str,
    columns: np.ndarray,
    gamma: np.ndarray,
    donor_ids: tuple[int, ...],
    by_id: dict[int, SiteSample],
    n_total: int,
    basis_of,
) -> _BlockMap:
    raw = None
    for site_id in donor_ids:
        basis = basis_of(by_id[site_id])
        gram = basis.T @ basis
        raw = gram if raw is None else raw + gram
    if raw is None:
        raise RuntimeError(f"Block map {name} has no donors.")
    return _BlockMap(
        name=name,
        columns=columns,
        gamma=gamma,
        donor_ids=tuple(int(site_id) for site_id in donor_ids),
        h_inv=_map_inverse(raw, n_total),
    )


def _stabilize(omega: np.ndarray) -> np.ndarray:
    sym = 0.5 * (omega + omega.T)
    scale = float(np.trace(sym) / sym.shape[0]) + 1e-12
    smallest = float(np.linalg.eigvalsh(sym)[0])
    if smallest < 1e-8 * scale:
        sym = sym + (1e-8 * scale) * np.eye(sym.shape[0])
    return sym


def fit_gmm(
    sites: list[SiteSample],
    amap: TransportMap,
    *,
    skip_outcome: tuple[int, ...] = (),
) -> FitResult:
    """Two-step GMM for the transported moment system.

    The instrument at each site is the recorded feature vector, plus the
    anchor dictionary when a non-anchor block is missing. The first step uses
    the sample-size weight. The second step uses the inverse of the plug-in
    covariance of the influence: the outcome score scaled by the inverse site
    share, minus the donor term for every conditional mean that observation
    helps fit. The donor Gram in that term is the raw sum divided by the
    sample size of the sites that enter the influence.

    Sites that do not record the anchor are not stacked. The declared maps
    transport the non-anchor blocks given the anchor, and a site that never
    records the anchor cannot evaluate those dictionaries. A fill of the
    missing anchor from the intercept, or from the intercept and the second
    block, leaves the outcome moments biased.
    """
    if not sites:
        raise RuntimeError("The GMM received no sites.")
    by_id = {int(site.site_id): site for site in sites}
    skipped = {int(site_id) for site_id in skip_outcome}

    outcome: list[_OutcomeSite] = []
    for site in sorted(sites, key=lambda item: int(item.site_id)):
        if int(site.site_id) in skipped or not site.observes_anchor:
            continue
        q, xt, basis = _known_instrument(site)
        xt = _fill_transported(site, xt, basis, amap.gamma_b, amap.gamma_c)
        h_anchor = basis if basis is not None else _anchor_basis(site)
        outcome.append(
            _OutcomeSite(
                int(site.site_id),
                site.n,
                q,
                xt,
                site.y,
                {"B": h_anchor, "C": h_anchor},
            )
        )
    if not outcome:
        raise RuntimeError("No site entered the outcome stack.")

    offset = 0
    for site in outcome:
        site.offset = offset
        offset += site.width
    n_moments = offset
    outcome_ids = {site.site_id for site in outcome}
    participant_ids = set(outcome_ids)
    participant_ids.update(int(site_id) for site_id in amap.donors_b)
    participant_ids.update(int(site_id) for site_id in amap.donors_c)
    missing = participant_ids.difference(by_id)
    if missing:
        raise RuntimeError(f"Donor sites {sorted(missing)} are not in the federation.")
    n_total = int(sum(by_id[site_id].n for site_id in participant_ids))

    def _basis_bc(site: SiteSample) -> np.ndarray:
        return _anchor_basis(site)

    maps: list[_BlockMap] = [
        _build_block_map("B", IDX_B, amap.gamma_b, amap.donors_b, by_id, n_total, _basis_bc),
        _build_block_map("C", IDX_C, amap.gamma_c, amap.donors_c, by_id, n_total, _basis_bc),
    ]

    recipient_of: dict[str, list[_OutcomeSite]] = {item.name: [] for item in maps}
    for site in outcome:
        sample = by_id[site.site_id]
        for item in maps:
            recorded = bool(sample.mask[item.columns].all())
            if recorded or item.name not in site.h_by_map:
                continue
            recipient_of[item.name].append(site)

    blocks_a = []
    blocks_k = []
    for site in outcome:
        blocks_a.append((site.q.T @ site.xt) / site.n)
        blocks_k.append((site.q.T @ site.y) / site.n)
    hessian = np.zeros((N_FEATURES, N_FEATURES), dtype=float)
    rhs = np.zeros(N_FEATURES, dtype=float)
    for site, design, cross in zip(outcome, blocks_a, blocks_k):
        hessian += site.n * (design.T @ design)
        rhs += site.n * (design.T @ cross)
    beta1, how1 = _solve(hessian, rhs)

    outcome_by_id = {site.site_id: site for site in outcome}
    cross_omega = np.zeros((n_moments, n_moments), dtype=float)
    for sample in sites:
        if int(sample.site_id) not in participant_ids:
            continue
        psi = np.zeros((sample.n, n_moments), dtype=float)
        site = outcome_by_id.get(int(sample.site_id))
        if site is not None:
            residual = site.y - site.xt @ beta1
            share = site.n / float(n_total)
            psi[:, site.offset : site.offset + site.width] = (site.q * residual[:, None]) / share
        for item in maps:
            if int(sample.site_id) not in item.donor_ids:
                continue
            recipients = recipient_of[item.name]
            if not recipients:
                continue
            basis = _anchor_basis(sample)
            deviation = _column_block(sample, item.columns) - basis @ item.gamma
            scale = deviation @ beta1[item.columns]
            for recipient in recipients:
                # Q H^{-1} is (r, d); each donor row contributes that factor times h and r' beta.
                cross_h = recipient.q.T @ recipient.h_by_map[item.name]
                factor = (cross_h / recipient.n) @ item.h_inv.T
                loading = basis @ factor.T
                psi[:, recipient.offset : recipient.offset + recipient.width] -= loading * scale[:, None]
        cross_omega += psi.T @ psi
    omega = _stabilize(cross_omega / float(n_total))

    design = np.vstack(blocks_a)
    cross = np.concatenate(blocks_k)
    weighted, how_w = _solve(omega, np.column_stack([design, cross]))
    hessian2 = design.T @ weighted[:, :N_FEATURES]
    rhs2 = design.T @ weighted[:, N_FEATURES]
    beta2, how2 = _solve(hessian2, rhs2)
    moment = cross - design @ beta2
    weighted_moment, _ = _solve(omega, moment)
    j_stat = float(n_total * (moment @ weighted_moment))
    if j_stat < 0.0 and j_stat > -1e-6:
        j_stat = 0.0
    j_df = int(n_moments - N_FEATURES)
    j_p = float(chi2.sf(j_stat, j_df)) if j_df > 0 else float("nan")
    how_second = how2 if how_w == "solve" else how_w
    return FitResult(
        method="federated_gmm",
        beta=beta2,
        gamma_b=amap.gamma_b,
        gamma_c=amap.gamma_c,
        solve=f"{how1}/{how_second}",
        j_stat=j_stat,
        j_df=j_df,
        j_pvalue=j_p,
        n_moments=n_moments,
        stacked_site_ids=tuple(site.site_id for site in outcome),
        design=design,
        moment_cross=cross,
    )


def fit_oracle(sites: list[SiteSample]) -> FitResult:
    """Pooled least squares on the complete feature vectors, before masking."""
    features = np.vstack([site.X for site in sites])
    outcome = np.concatenate([site.y for site in sites])
    beta, how = _solve(features.T @ features, features.T @ outcome)
    return FitResult(method="pooled_oracle", beta=beta, solve=how)


def fit_mean_impute_fedavg(sites: list[SiteSample]) -> FitResult:
    """Local least squares on recorded columns, zeros on missing ones, then average.

    Prediction fills a missing block with the n-weighted mean from sites that
    record it. That is the mean-imputation step. Averaging the embedded local
    coefficients mixes omitted-variable estimands across observation patterns.
    """
    total_n = 0
    beta = np.zeros(N_FEATURES)
    solves = []
    for site in sites:
        view = site.federated_view()
        idx = np.flatnonzero(site.mask)
        local, how = _solve(view[:, idx].T @ view[:, idx], view[:, idx].T @ site.y)
        embedded = np.zeros(N_FEATURES)
        embedded[idx] = local
        beta += site.n * embedded
        total_n += site.n
        solves.append(how)
    beta /= total_n
    return FitResult(
        method="mean_impute_fedavg",
        beta=beta,
        feature_mean=_feature_means(sites),
        solve="lstsq" if "lstsq" in solves else "solve",
    )


def _feature_means(sites: list[SiteSample]) -> np.ndarray:
    sums = np.zeros(N_FEATURES)
    counts = np.zeros(N_FEATURES)
    for site in sites:
        view = site.federated_view()
        idx = np.flatnonzero(site.mask)
        sums[idx] += np.nansum(view[:, idx], axis=0)
        counts[idx] += site.n
    if np.any(counts <= 0):
        raise RuntimeError("A coordinate has no recording site.")
    return sums / counts


def fit_mcar_plugin(sites: list[SiteSample]) -> FitResult:
    """Component-wise second moments, each pair taken from sites that record it.

    This is consistent for a shared covariate law when every pair is
    co-observed somewhere. Under site-specific laws it splices cross-moments
    from one subset of sites onto marginals from another subset.
    """
    acc = np.zeros((N_FEATURES, N_FEATURES))
    weight = np.zeros((N_FEATURES, N_FEATURES))
    acc_y = np.zeros(N_FEATURES)
    weight_y = np.zeros(N_FEATURES)
    for site in sites:
        view = site.federated_view()
        idx = np.flatnonzero(site.mask)
        observed = view[:, idx]
        xtx = observed.T @ observed
        xty = observed.T @ site.y
        for a, i in enumerate(idx):
            acc_y[i] += xty[a]
            weight_y[i] += site.n
            for b, j in enumerate(idx):
                acc[i, j] += xtx[a, b]
                weight[i, j] += site.n
    if np.any(weight <= 0) or np.any(weight_y <= 0):
        raise RuntimeError("A feature pair is never co-observed; the plug-in is undefined.")
    sigma = acc / weight
    sigma = 0.5 * (sigma + sigma.T)
    cross = acc_y / weight_y
    beta, how = _solve(sigma, cross)
    return FitResult(
        method="mcar_plugin",
        beta=beta,
        sigma=sigma,
        cross=cross,
        feature_mean=_feature_means(sites),
        solve=how,
    )


def fit_all(sites: list[SiteSample]) -> dict[str, FitResult]:
    oracle = fit_oracle(sites)
    amap, tests = fit_transport(sites)
    gmm = fit_gmm(sites, amap)
    gmm.tests = tests
    plugin = fit_mcar_plugin(sites)
    fedavg = fit_mean_impute_fedavg(sites)
    return {
        oracle.method: oracle,
        gmm.method: gmm,
        plugin.method: plugin,
        fedavg.method: fedavg,
    }


def predict(site: SiteSample, fit: FitResult) -> np.ndarray | None:
    """Site-level predictions from recorded columns, except the complete-data oracle."""
    n = site.n
    if fit.method == "pooled_oracle":
        return site.X @ fit.beta
    if fit.method == "federated_gmm":
        if not site.observes_anchor:
            return None
        if fit.gamma_b is None or fit.gamma_c is None:
            raise ValueError("GMM fit is missing transport maps.")
        regressor = transported_design(
            site.federated_view(), site.mask, fit.gamma_b, fit.gamma_c
        )
        return regressor @ fit.beta
    if fit.method == "mean_impute_fedavg":
        if fit.feature_mean is None:
            raise ValueError("FedAvg fit is missing feature means.")
        completed = np.repeat(fit.feature_mean[None, :], n, axis=0)
        view = site.federated_view()
        idx = np.flatnonzero(site.mask)
        completed[:, idx] = view[:, idx]
        return completed @ fit.beta
    if fit.method == "mcar_plugin":
        # Deploy the spliced coefficient. Missing blocks are filled with the
        # donor means, so this prediction is mean imputation plus the plug-in.
        if fit.feature_mean is None:
            raise ValueError("Plug-in fit is missing feature means.")
        completed = np.repeat(fit.feature_mean[None, :], n, axis=0)
        view = site.federated_view()
        idx = np.flatnonzero(site.mask)
        completed[:, idx] = view[:, idx]
        return completed @ fit.beta
    raise KeyError(fit.method)


def prediction_mse(site: SiteSample, fit: FitResult) -> float | None:
    yhat = predict(site, fit)
    if yhat is None:
        return None
    return float(np.mean((site.y - yhat) ** 2))
