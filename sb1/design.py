"""Synthetic design for block-missing federated linear regression.

No site records every block. The outcome is linear in the full feature
vector, with one coefficient shared by every site. The anchor block A has a
site-specific Gaussian law. Blocks B and C are drawn from a common conditional
mean given A, plus independent Gaussian noise, so those two conditional means
are transportable. Two sites record B and C but not A. They are the only places the B–C
cross-moment is observed.

The numeric constants below are the design. Do not retune them to chase a
table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2
N_FEATURES = 1 + 3 * DIM  # intercept, A, B, C
COEF_NAMES = ("intercept", "A1", "A2", "B1", "B2", "C1", "C2")

# Fourier features of the anchor. Eight columns: sin(w·A), cos(w·A).
FREQUENCIES = np.array(
    [
        [1.7, 0.4],
        [0.3, 1.8],
        [1.4, -1.2],
        [-0.9, 1.1],
    ],
    dtype=float,
)
N_BASIS = int(FREQUENCIES.shape[0] * 2)

# Frozen conditional-mean coefficients, basis by block coordinate.
# Rows follow phi(); columns are the two coordinates of the target block.
GAMMA_B = np.array(
    [
        [0.1901, 0.4519],
        [0.1817, -0.7167],
        [0.4979, 0.2455],
        [-0.2953, 0.3196],
        [0.2005, 0.1618],
        [0.0156, 0.3007],
        [-0.4050, -0.0896],
        [-0.2652, 0.3294],
    ],
    dtype=float,
)
GAMMA_C = np.array(
    [
        [0.0218, -0.1609],
        [-0.4300, -0.1415],
        [0.0045, -0.1516],
        [0.7117, 0.5537],
        [-1.4911, -1.0390],
        [-0.0961, -0.2322],
        [0.1175, 0.1195],
        [1.1648, -0.6116],
    ],
    dtype=float,
)

BETA_TRUE = np.array([0.5, 1.0, -0.5, 0.8, 0.4, -0.6, 1.1], dtype=float)
SIGMA_COND = 0.45
SIGMA_Y = 0.50

# Added to a donor's conditional-mean coefficient under transport_violated.
DELTA_B = np.full((N_BASIS, DIM), 0.45, dtype=float)
DELTA_C = np.full((N_BASIS, DIM), -0.40, dtype=float)
VIOLATED_B_SITE = 3
VIOLATED_C_SITE = 4

# feature index sets
IDX_INTERCEPT = np.array([0])
IDX_A = np.array([1, 2])
IDX_B = np.array([3, 4])
IDX_C = np.array([5, 6])

PATTERNS = {
    "AB": np.array([1, 1, 1, 1, 1, 0, 0], dtype=bool),
    "AC": np.array([1, 1, 1, 0, 0, 1, 1], dtype=bool),
    "BC": np.array([1, 0, 0, 1, 1, 1, 1], dtype=bool),
    "A": np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool),
}

REGIMES = ("transport_correct", "transport_violated", "common_law")
REGIME_CODE = {name: i for i, name in enumerate(REGIMES)}


@dataclass(frozen=True)
class SiteSpec:
    site_id: int
    pattern: str
    mean_A: tuple[float, float]
    sd_A: float

    @property
    def mask(self) -> np.ndarray:
        return PATTERNS[self.pattern].copy()


# Seven sites. Means of A differ. No pattern records A, B, and C together.
# Each co-observed pair has two sites, so leave-one-site-out still sees that pair.
SITES: tuple[SiteSpec, ...] = (
    SiteSpec(0, "AB", (0.0, 0.0), 0.70),
    SiteSpec(1, "AC", (1.0, -0.6), 0.70),
    SiteSpec(2, "BC", (-0.9, 0.8), 0.80),
    SiteSpec(3, "AB", (-0.8, 1.0), 0.85),
    SiteSpec(4, "AC", (0.6, 0.7), 0.65),
    SiteSpec(5, "A", (-1.0, -0.4), 0.75),
    SiteSpec(6, "BC", (0.4, -1.0), 0.75),
)

COMMON_LAW_MEAN = (0.0, 0.0)
COMMON_LAW_SD = 0.70

# Expansion factors declared before the 2026-10-20 sweeps. Not fit to a table.
# Shift multiplies each site-mean deviation from the centroid of the seven
# anchor means. Medium is the locked design (scale 1 returns those means).
SHIFT_SCALES = {"weak": 0.5, "medium": 1.0, "strong": 2.0}

# One donor per conditional mean: drop the second AB site and the second AC site.
ONE_DONOR_IDS = (0, 1, 2, 5, 6)

# Third donors are origin reflections of site 1 and site 3, with standard
# deviations already used in the locked design. Site 7 records A with B.
# Site 8 records A with C. Neither id is a violated-transport site.
EXTRA_DONOR_SITES: tuple[SiteSpec, ...] = (
    SiteSpec(7, "AB", (-1.0, 0.6), 0.70),
    SiteSpec(8, "AC", (0.8, -1.0), 0.75),
)


def phi(anchor: np.ndarray) -> np.ndarray:
    """Basis of the anchor block. Shape (n, N_BASIS)."""
    columns = []
    for frequency in FREQUENCIES:
        angle = anchor @ frequency
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)


def anchor_law(spec: SiteSpec, regime: str) -> tuple[np.ndarray, float]:
    if regime == "common_law":
        return np.array(COMMON_LAW_MEAN, dtype=float), float(COMMON_LAW_SD)
    if regime not in REGIME_CODE:
        raise KeyError(regime)
    return np.array(spec.mean_A, dtype=float), float(spec.sd_A)


def conditional_coefficients(spec: SiteSpec, regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Population maps E[B|A] = phi(A) @ gamma_B and likewise for C."""
    gamma_b = GAMMA_B.copy()
    gamma_c = GAMMA_C.copy()
    if regime == "transport_violated":
        if spec.site_id == VIOLATED_B_SITE:
            gamma_b = GAMMA_B + DELTA_B
        if spec.site_id == VIOLATED_C_SITE:
            gamma_c = GAMMA_C + DELTA_C
    return gamma_b, gamma_c


def make_rng(seed: int, regime: str, site_id: int, split: str) -> np.random.Generator:
    split_code = 0 if split == "train" else 1
    sequence = np.random.SeedSequence(
        [int(seed), REGIME_CODE[regime], int(site_id), split_code]
    )
    return np.random.default_rng(sequence)


@dataclass
class SiteSample:
    """One site's sample.

    ``X`` stores the complete feature vector, including blocks the site does
    not record. Federated estimators must use ``federated_view``, which
    replaces those columns with NaN. The complete array is only for the
    pooled complete-data oracle and for scoring.
    """

    site_id: int
    pattern: str
    mask: np.ndarray
    X: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def observes_anchor(self) -> bool:
        return bool(self.mask[IDX_A].all())

    def federated_view(self) -> np.ndarray:
        view = self.X.copy()
        view[:, ~self.mask] = np.nan
        return view


def generate_site(
    spec: SiteSpec,
    n: int,
    seed: int,
    regime: str,
    split: str,
) -> SiteSample:
    rng = make_rng(seed, regime, spec.site_id, split)
    mean_a, sd_a = anchor_law(spec, regime)
    anchor = rng.normal(mean_a, sd_a, size=(n, DIM))
    gamma_b, gamma_c = conditional_coefficients(spec, regime)
    basis = phi(anchor)
    block_b = basis @ gamma_b + rng.normal(0.0, SIGMA_COND, size=(n, DIM))
    block_c = basis @ gamma_c + rng.normal(0.0, SIGMA_COND, size=(n, DIM))
    features = np.column_stack([np.ones(n), anchor, block_b, block_c])
    outcome = features @ BETA_TRUE + rng.normal(0.0, SIGMA_Y, size=n)
    return SiteSample(
        site_id=spec.site_id,
        pattern=spec.pattern,
        mask=spec.mask.copy(),
        X=features,
        y=outcome,
    )


def layout_sites(layout: str) -> tuple[SiteSpec, ...]:
    """Co-observation layout. ``two`` is the locked seven-site design."""
    if layout == "two":
        return SITES
    if layout == "one":
        return tuple(spec for spec in SITES if spec.site_id in ONE_DONOR_IDS)
    if layout == "three":
        return SITES + EXTRA_DONOR_SITES
    raise KeyError(layout)


def apply_mean_scale(sites: tuple[SiteSpec, ...], scale: float) -> tuple[SiteSpec, ...]:
    """Scale anchor-mean separation. Scale 1 returns the input objects unchanged."""
    if float(scale) == 1.0:
        return tuple(sites)
    means = np.array([spec.mean_A for spec in sites], dtype=float)
    centroid = means.mean(axis=0)
    scaled = []
    for spec, mean in zip(sites, means):
        moved = centroid + float(scale) * (mean - centroid)
        scaled.append(
            SiteSpec(
                spec.site_id,
                spec.pattern,
                (float(moved[0]), float(moved[1])),
                spec.sd_A,
            )
        )
    return tuple(scaled)


def generate_split(
    n: int,
    seed: int,
    regime: str,
    split: str,
    sites: tuple[SiteSpec, ...] | None = None,
) -> list[SiteSample]:
    chosen = SITES if sites is None else sites
    return [generate_site(spec, n, seed, regime, split) for spec in chosen]


def design_record(
    sites: tuple[SiteSpec, ...] | None = None,
    mean_scale: float = 1.0,
    layout: str = "two",
) -> dict:
    """JSON-serializable copy of the constants that define a run."""
    chosen = SITES if sites is None else sites
    return {
        "design_id": "sb1-synthetic-v1",
        "layout": layout,
        "mean_scale": float(mean_scale),
        "shift_scales": dict(SHIFT_SCALES),
        "block_dim": DIM,
        "blocks": ["A", "B", "C"],
        "coef_names": list(COEF_NAMES),
        "beta_true": BETA_TRUE.tolist(),
        "sigma_cond": SIGMA_COND,
        "sigma_outcome": SIGMA_Y,
        "frequencies": FREQUENCIES.tolist(),
        "gamma_B": GAMMA_B.tolist(),
        "gamma_C": GAMMA_C.tolist(),
        "delta_B": DELTA_B.tolist(),
        "delta_C": DELTA_C.tolist(),
        "violated_B_site": VIOLATED_B_SITE,
        "violated_C_site": VIOLATED_C_SITE,
        "common_law_mean_A": list(COMMON_LAW_MEAN),
        "common_law_sd_A": COMMON_LAW_SD,
        "sites": [
            {
                "site_id": spec.site_id,
                "pattern": spec.pattern,
                "mean_A": list(spec.mean_A),
                "sd_A": spec.sd_A,
                "records": {
                    "A": bool(spec.mask[IDX_A].all()),
                    "B": bool(spec.mask[IDX_B].all()),
                    "C": bool(spec.mask[IDX_C].all()),
                },
            }
            for spec in chosen
        ],
        "gmm_sites": "sites that record anchor A; known instruments; influence weight with donor term",
        "plugin_sites": "all sites",
        "regimes": list(REGIMES),
    }
