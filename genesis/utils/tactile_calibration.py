"""Fit identifiable tactile sensitivity models and report errors on held-out measurements."""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

import genesis as gs
from genesis.options.sensors.tactile_profile import TactileHardwareProfile, TactileTransferCurve


def fit_tactile_profile(
    samples: Mapping,
    base_profile: TactileHardwareProfile | None = None,
    *,
    curve_degree: int = 3,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[TactileHardwareProfile, dict]:
    """Fit normal-force gain/offset per taxel and a shared polynomial response from indenter data.

    ``samples`` contains one-dimensional ``taxel`` integer indices, ``f_normal`` loads in N, and
    ``reading_normal`` readings in N. Taxel indices must cover consecutive integers beginning at zero.
    The model is ``reading = offset[taxel] + curve(gain[taxel] * f_normal)``, with positive gains and unit
    small-signal curve slope to remove gain/curve scale ambiguity. ``curve_degree`` is between 1 and 5.

    Holdout samples are selected independently within each taxel using ``seed``, retaining one minimum-load and
    one maximum-load sample for training. Each taxel needs at least one held-out sample and ``curve_degree + 1``
    distinct training loads. Held-out readings never participate in fitting. ``base_profile`` supplies geometry
    and unfitted hardware parameters. Readings must already be corrected for those
    unfitted effects, including the base profile's bias and temperature response.
    Normal transfer limits are replaced with a monotone curve bounded by the largest
    fitted training input.

    Returns a validated profile and a JSON-compatible report containing split indices, fit parameters,
    training/holdout errors, and the fitted load range. Rank-deficient data, failed optimization, and a
    non-monotone fitted response raise GenesisException. The operation is synchronous and CPU-only.
    """
    if isinstance(curve_degree, bool) or not isinstance(curve_degree, int) or not 1 <= curve_degree <= 5:
        gs.raise_exception("Tactile calibration curve_degree must be an integer between 1 and 5.")
    if not np.isfinite(holdout_fraction) or not 0.0 < holdout_fraction < 1.0:
        gs.raise_exception("Tactile calibration holdout_fraction must be strictly between zero and one.")
    if not {"taxel", "f_normal", "reading_normal"} <= samples.keys():
        gs.raise_exception("Tactile calibration requires taxel, f_normal, and reading_normal arrays.")
    taxels = np.asarray(samples["taxel"])
    force = np.asarray(samples["f_normal"], dtype=np.float64)
    reading = np.asarray(samples["reading_normal"], dtype=np.float64)
    if taxels.ndim != 1 or force.shape != taxels.shape or reading.shape != taxels.shape:
        gs.raise_exception("Tactile calibration arrays must have the same one-dimensional shape.")
    if not np.issubdtype(taxels.dtype, np.integer) or taxels.size == 0:
        gs.raise_exception("Tactile calibration taxel indices must be a nonempty integer array.")
    if not np.isfinite(force).all() or not np.isfinite(reading).all() or np.any(force < 0.0):
        gs.raise_exception("Tactile calibration requires finite readings and finite nonnegative normal loads.")
    taxel_values = np.unique(taxels)
    n_taxels = len(taxel_values)
    if not np.array_equal(taxel_values, np.arange(n_taxels)):
        gs.raise_exception("Tactile calibration taxel indices must cover consecutive integers starting at zero.")
    if base_profile is not None and base_profile.n_taxels not in (None, n_taxels):
        gs.raise_exception("Tactile calibration must include every taxel in the supplied hardware profile.")

    rng = np.random.default_rng(seed)
    train_parts, holdout_parts = [], []
    for i_t in range(n_taxels):
        indices = rng.permutation(np.flatnonzero(taxels == i_t))
        endpoints = np.unique((indices[np.argmin(force[indices])], indices[np.argmax(force[indices])]))
        candidates = indices[~np.isin(indices, endpoints)]
        n_holdout = max(1, int(len(indices) * holdout_fraction))
        if len(candidates) < n_holdout:
            gs.raise_exception(f"Tactile calibration taxel {i_t} has insufficient samples for a held-out split.")
        holdout_parts.append(candidates[:n_holdout])
        train_parts.append(np.concatenate((endpoints, candidates[n_holdout:])))
        if len(np.unique(force[train_parts[-1]])) < curve_degree + 1:
            gs.raise_exception(f"Tactile calibration taxel {i_t} needs {curve_degree + 1} distinct training loads.")
    train = np.concatenate(train_parts)
    holdout = np.concatenate(holdout_parts)
    force_scale = force[train].max()
    scaled_force = force / force_scale
    initial_polynomials = []
    for indices in train_parts:
        design = np.polynomial.polynomial.polyvander(scaled_force[indices], curve_degree)
        coefficients, _, rank, _ = np.linalg.lstsq(design, reading[indices], rcond=None)
        if rank != curve_degree + 1 or coefficients[1] <= 0.0:
            gs.raise_exception("Tactile calibration requires an identifiable positive small-signal sensitivity.")
        initial_polynomials.append(coefficients)
    initial_polynomials = np.stack(initial_polynomials)
    initial_gain = initial_polynomials[:, 1]
    initial_curve = [
        np.mean(initial_polynomials[:, order] / initial_gain**order) for order in range(2, curve_degree + 1)
    ]
    initial = np.concatenate((np.log(initial_gain), initial_polynomials[:, 0], initial_curve))

    def predict(parameters, indices):
        gain = np.exp(parameters[:n_taxels])
        offset = parameters[n_taxels : 2 * n_taxels]
        magnitude = gain[taxels[indices]] * scaled_force[indices]
        value = magnitude.copy()
        for order, coefficient in enumerate(parameters[2 * n_taxels :], start=2):
            value += coefficient * magnitude**order
        return offset[taxels[indices]] + value

    def residual(parameters):
        return predict(parameters, train) - reading[train]

    result = least_squares(residual, initial, x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    if not result.success or not np.isfinite(result.x).all():
        gs.raise_exception(f"Tactile calibration optimization failed: {result.message}")
    if np.linalg.matrix_rank(result.jac) != len(initial):
        gs.raise_exception("Tactile calibration data do not identify the requested gains, offsets, and shared curve.")

    gains = np.exp(result.x[:n_taxels]) / force_scale
    offsets = result.x[n_taxels : 2 * n_taxels]
    coefficients = (1.0, *result.x[2 * n_taxels :])
    data = {} if base_profile is None else base_profile.to_dict()
    data.update(
        name="fitted_profile" if base_profile is None else base_profile.name,
        gain=gains,
        offset=offsets,
        normal_curve=TactileTransferCurve(
            kind="polynomial",
            coeffs=coefficients,
            input_max=np.max(gains[taxels[train]] * force[train]),
        ),
    )
    profile = TactileHardwareProfile(**data)
    train_error = predict(result.x, train) - reading[train]
    holdout_error = predict(result.x, holdout) - reading[holdout]
    reference_variance = np.sum((reading[holdout] - reading[holdout].mean()) ** 2)
    report = {
        "schema_version": 1,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "train_indices": train.tolist(),
        "holdout_indices": holdout.tolist(),
        "rmse_train": float(np.sqrt(np.mean(train_error**2))),
        "rmse_holdout": float(np.sqrt(np.mean(holdout_error**2))),
        "r2_holdout": float(1.0 - np.sum(holdout_error**2) / reference_variance) if reference_variance > 0.0 else None,
        "per_taxel_gain": gains.tolist(),
        "per_taxel_offset": offsets.tolist(),
        "curve_coeffs": list(coefficients),
        "force_range_n": [float(force[train].min()), float(force[train].max())],
    }
    return profile, report
