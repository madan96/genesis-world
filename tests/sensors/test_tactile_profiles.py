import json

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.tactile_calibration import fit_tactile_profile

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_tactile_profile_serialization(tmp_path, suffix):
    profile = gs.sensors.TactileHardwareProfile(
        name="two_taxel_fixture",
        taxel_pos=((0.0, 0.0, 0.0), (0.02, 0.0, 0.0)),
        taxel_normal=(0.0, 0.0, 2.0),
        support_radius=0.01,
        gain=(0.7, 1.4),
        offset=(0.2, -0.1),
        validity=(True, False),
        normal_curve=gs.sensors.TactileTransferCurve(
            kind="polynomial",
            coeffs=(1.0, 0.15),
            input_max=14.0,
        ),
    )
    path = tmp_path / ("profile" + suffix)
    profile.save(path)
    loaded = gs.sensors.TactileHardwareProfile.load(path)
    assert loaded.to_dict() == profile.to_dict()
    assert loaded.n_taxels == 2
    assert_equal(loaded.taxel_normal, ((0.0, 0.0, 1.0),))
    record = loaded.to_dict()
    record["gain"][0] = 99.0
    assert loaded.gain[0] == 0.7

    # A loaded profile supplies the taxel geometry and electronics of a sensor; explicit options take precedence.
    options = gs.sensors.SolverContactTaxel.from_profile(loaded, entity_idx=0)
    assert_allclose(options.probe_local_pos, ((0.0, 0.0, 0.0), (0.02, 0.0, 0.0)), tol=0)
    assert_allclose(options.probe_local_normal, (0.0, 0.0, 1.0), tol=0)
    assert_allclose(options.probe_radius, 0.01, tol=0)
    assert_allclose(options.probe_gain, (0.7, 1.4), tol=0)
    assert options.hardware_profile.to_dict() == loaded.to_dict()
    options = gs.sensors.SolverContactTaxel.from_profile(loaded, entity_idx=0, probe_radius=0.02, probe_gain=2.0)
    assert_allclose(options.probe_radius, 0.02, tol=0)
    assert_allclose(options.probe_gain, 2.0, tol=0)
    with pytest.raises(gs.GenesisException, match="taxel_pos"):
        gs.sensors.SolverContactTaxel.from_profile(gs.sensors.TactileHardwareProfile(), entity_idx=0)

    untouched = tmp_path / "profile.unsupported"
    untouched.write_text("existing contents")
    with pytest.raises(gs.GenesisException, match="destination"):
        profile.save(untouched)
    assert untouched.read_text() == "existing contents"

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(gs.GenesisException, match="schema_version"):
        gs.sensors.TactileHardwareProfile.load(invalid)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "gain": [0.7, 1.4],
                "offset": [0.2, -0.1],
                "offset_temp_coeff": 0.02,
                "normal_curve": {"kind": "linear", "scale": 2.0},
            }
        )
    )
    migrated = gs.sensors.TactileHardwareProfile.load(legacy)
    assert_allclose(migrated.gain, (0.7, 1.4), tol=0)
    assert_allclose(migrated.offset, (-0.2, 0.1), tol=0)
    assert_allclose(migrated.offset_temp_coeff, -0.02, tol=0)
    migrated.save(path)
    assert gs.sensors.TactileHardwareProfile.load(path).to_dict() == migrated.to_dict()
    record = profile.to_dict()
    del record["schema_version"]
    legacy.write_text(json.dumps(record))
    with pytest.raises(gs.GenesisException, match="require migration"):
        gs.sensors.TactileHardwareProfile.load(legacy)
    with pytest.raises(gs.GenesisException, match="require migration"):
        gs.sensors.TactileHardwareProfile.resolve(record)
    with pytest.raises(gs.GenesisException):
        gs.sensors.TactileHardwareProfile.resolve({"normal_curve": None})


@pytest.mark.required
def test_tactile_profile_validation():
    for fields in (
        {"taxel_normal": (0.0, 0.0, 0.0)},
        {"gain": (1.0, float("nan"))},
        {"validity": (True, 1)},
        {"gain": (1.0, 2.0), "offset": (0.0, 0.0, 0.0)},
        {"support_area": 0.0},
        {"gain_variation": (1.0, 0.5)},
        {"delay": 0.0, "jitter": 0.01},
        {"hysteresis_strength": 0.5, "hysteresis_tau": 0.0},
        {"crosstalk_strength": 0.5, "crosstalk_sigma": 0.0},
    ):
        with pytest.raises((ValueError, gs.GenesisException)):
            gs.sensors.TactileHardwareProfile(**fields)
    # This derivative is positive at both endpoints, but negative inside the calibrated interval.
    with pytest.raises(gs.GenesisException, match="nondecreasing"):
        gs.sensors.TactileTransferCurve(kind="polynomial", coeffs=(1.0, -3.0, 2.0), input_max=1.0)
    with pytest.raises(gs.GenesisException, match="input_max"):
        gs.sensors.TactileTransferCurve(kind="polynomial", coeffs=(1.0,))
    with pytest.raises(gs.GenesisException, match="nondecreasing"):
        gs.sensors.TactileTransferCurve(kind="lookup", input_points=(0.0, 1.0, 2.0), output_points=(0.0, 2.0, 1.0))


@pytest.mark.required
def test_tactile_transfer_curves(tol):
    magnitude = torch.tensor((0.0, 0.25, 0.5, 1.0, 2.0), dtype=gs.tc_float, device=gs.device)
    linear = gs.sensors.TactileTransferCurve(scale=2.0, dead_band=0.5, saturation=3.0)
    assert_allclose(linear.apply_magnitude(magnitude), (0.0, 0.0, 1.0, 2.0, 3.0), tol=tol)
    polynomial = gs.sensors.TactileTransferCurve(kind="polynomial", coeffs=(1.0, 0.5), input_max=1.0)
    assert_allclose(polynomial.apply_magnitude(magnitude), (0.0, 0.28125, 0.625, 1.5, 1.5), tol=tol)
    lookup = gs.sensors.TactileTransferCurve(kind="lookup", input_points=(0.0, 0.5, 1.0), output_points=(0.0, 1.0, 1.5))
    assert_allclose(lookup.apply_magnitude(magnitude), (0.0, 0.5, 1.0, 1.5, 1.5), tol=tol)
    evaluator = lookup.build(device=gs.device, dtype=gs.tc_float)
    assert_allclose(evaluator.apply_magnitude(magnitude), (0.0, 0.5, 1.0, 1.5, 1.5), tol=tol)
    lookup.output_points = (0.0, 2.0, 3.0)
    assert_allclose(evaluator.apply_magnitude(magnitude), (0.0, 0.5, 1.0, 1.5, 1.5), tol=tol)
    saturation_scale = magnitude.new_tensor((1.0, 0.5, 0.25, 0.5, 0.25))
    assert_allclose(linear.apply_magnitude(magnitude, saturation_scale), (0.0, 0.0, 0.75, 1.5, 0.75), tol=tol)
    snapshot = magnitude.clone()
    output = linear.apply_magnitude(magnitude)
    output.zero_()
    assert_equal(magnitude, snapshot)


@pytest.mark.required
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_tactile_calibration_identifiability_and_holdout(degree, tol):
    taxels = np.repeat(np.arange(2), 50)
    force = np.tile(np.linspace(0.0, 10.0, 50), 2)
    gains = np.array((0.7, 1.4))
    offsets = np.array((0.2, -0.1))
    coefficients = np.array((1.0, 0.15, 0.002, 0.00001, 0.0000001))[:degree]
    magnitude = gains[taxels] * force
    readings = offsets[taxels] + np.polynomial.polynomial.polyval(magnitude, (0.0, *coefficients))
    samples = {"taxel": taxels, "f_normal": force, "reading_normal": readings}
    profile, report = fit_tactile_profile(samples, curve_degree=degree, seed=7)
    assert_allclose(profile.gain, gains, tol=1e-8)
    assert_allclose(profile.offset, offsets, tol=1e-8)
    assert_allclose(profile.normal_curve.coeffs, coefficients, tol=1e-8)
    assert report["rmse_train"] < 1e-10
    assert report["rmse_holdout"] < 1e-10
    assert report["n_holdout"] == 20
    holdout = report["holdout_indices"]
    assert not set(report["train_indices"]) & set(holdout)
    tensor_magnitude = torch.tensor(magnitude, dtype=gs.tc_float, device=gs.device)
    tensor_offset = torch.tensor(offsets[taxels], dtype=gs.tc_float, device=gs.device)
    assert_allclose(profile.normal_curve.apply_magnitude(tensor_magnitude) + tensor_offset, readings, tol=tol)

    altered = readings.copy()
    altered[holdout] += 10.0
    other, other_report = fit_tactile_profile({**samples, "reading_normal": altered}, curve_degree=degree, seed=7)
    assert other.to_dict() == profile.to_dict()
    assert_allclose(other_report["rmse_holdout"], 10.0, tol=1e-10)


@pytest.mark.required
def test_tactile_calibration_rejects_unidentified_models():
    force = np.linspace(0.0, 10.0, 10)
    samples = {"taxel": np.zeros(10, dtype=int), "f_normal": force, "reading_normal": force}
    for changes in (
        {"taxel": np.ones(10, dtype=int)},
        {"taxel": np.zeros(10)},
        {"f_normal": np.ones(10)},
        {"reading_normal": -force},
        {"reading_normal": np.full(10, np.nan)},
        {"f_normal": force.reshape(2, 5)},
    ):
        with pytest.raises(gs.GenesisException):
            fit_tactile_profile({**samples, **changes}, curve_degree=2)
