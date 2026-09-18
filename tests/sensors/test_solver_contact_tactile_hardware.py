import math

import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("name", ["noise", "random_walk", "resolution", "bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_electronics_options(name, value):
    if name == "bias" and value == -1.0:
        gs.sensors.SolverContactTaxel(entity_idx=0, probe_local_pos=((0, 0, 0),), **{name: value})
    else:
        with pytest.raises(gs.GenesisException):
            gs.sensors.SolverContactTaxel(entity_idx=0, probe_local_pos=((0, 0, 0),), **{name: value})


@pytest.mark.required
def test_invalid_electronics_setters_and_filters(show_viewer):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005, gravity=(0, 0, 0)), show_viewer=show_viewer)
    floor = scene.add_entity(gs.morphs.Plane())
    options = {"entity_idx": floor.idx, "probe_local_pos": ((0, 0, 0),), "contact_sources": ("rigid",)}
    with pytest.raises(gs.GenesisException, match="unknown.*link"):
        scene.add_sensor(gs.sensors.SolverContactTaxel(**options, filter_links=(floor.n_links,)))
    sensor = scene.add_sensor(gs.sensors.SolverContactTaxel(**options, delay=0.005))
    scene.build(n_envs=2)
    for setter in (sensor.set_noise, sensor.set_random_walk, sensor.set_resolution, sensor.set_jitter):
        for value in (float("nan"), float("inf"), -1.0):
            with pytest.raises(gs.GenesisException):
                setter(value, envs_idx=[0])
    for value in (float("nan"), float("inf")):
        with pytest.raises(gs.GenesisException):
            sensor.set_bias(value, envs_idx=[0])
    sensor.set_bias(-0.2, envs_idx=[0])
    for _ in range(2):
        scene.step()
    assert_allclose(sensor.read(envs_idx=[0]).force, -0.2, atol=1e-6)
    assert_equal(sensor.read(envs_idx=[1]).force, 0)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_profile_response_and_faults(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, gravity=(0, 0, -10)),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.1)),
        material=gs.materials.Rigid(rho=250),
    )
    options = {
        "entity_idx": floor.idx,
        "probe_local_pos": ((0.03, 0, 0),),
        "probe_local_normal": (0, 0, 1),
        "probe_radius": 0.25,
        "contact_sources": ("rigid",),
    }
    calibrated = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(
                gain=1.5,
                offset=0.2,
                gain_temp_coeff=0.01,
                offset_temp_coeff=0.02,
                normal_curve=gs.sensors.TactileTransferCurve(scale=2, saturation=100),
            ),
        )
    )
    legacy = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile={
                "gain": 1.5,
                "offset": 0.2,
                "offset_temp_coeff": 0.02,
                "normal_curve": {"kind": "linear", "scale": 2.0},
            },
        )
    )
    quantized = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(offset=0.18, bias=0.04, resolution=0.25),
        )
    )
    broken = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(validity=(False,)),
        )
    )
    stuck = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(
                dead_taxel_probability=1.0,
                dead_taxel_value_range=(3, 3),
            ),
        )
    )
    noisy = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(noise=0.05),
        )
    )
    scene.build(n_envs=n_envs)
    for _ in range(80):
        scene.step()

    assert_allclose(calibrated.read_ground_truth().force, (0, 0, -20), atol=0.01)
    assert_allclose(calibrated.read_ground_truth().couple, (0, -0.6, 0), atol=0.001)
    assert_allclose(calibrated.read().force, (0, 0, -60.2), atol=0.02)
    assert_allclose(calibrated.read().couple, (0, -0.9, 0), atol=0.002)
    assert_allclose(legacy.read().force, (0, 0, -59.8), atol=0.02)
    assert_allclose(quantized.read().force, (0, 0, -20.25), atol=1e-6)
    assert_allclose(quantized.read().couple, (0, -0.5, 0), atol=1e-6)
    assert_equal(broken.read().force, 0)
    assert_allclose(broken.read_ground_truth().force, (0, 0, -20), atol=0.01)
    assert_equal(stuck.read().force, 3)
    assert_equal(stuck.read().couple, 3)
    assert_allclose(stuck.read_ground_truth().force, (0, 0, -20), atol=0.01)

    selected = [0] if n_envs else None
    calibrated.set_temperature(35, envs_idx=selected)
    legacy.set_temperature(35, envs_idx=selected)
    scene.step()
    assert_allclose(calibrated.read(envs_idx=selected).force, (0, 0, -66.4), atol=0.02)
    assert_allclose(legacy.read(envs_idx=selected).force, (0, 0, -59.6), atol=0.02)
    if n_envs:
        assert_allclose(calibrated.read(envs_idx=[1]).force, (0, 0, -60.2), atol=0.02)
    calibrated.set_saturation_scale(0.5, envs_idx=selected)
    scene.step()
    assert_allclose(calibrated.read(envs_idx=selected).force, (0, 0, -50.4), atol=1e-5)
    calibrated.set_probe_gain(0.5, envs_idx=selected)
    scene.step()
    assert_allclose(calibrated.read(envs_idx=selected).force, (0, 0, -22.4), atol=0.01)
    assert_allclose(calibrated.read_ground_truth().force, (0, 0, -20), atol=0.01)

    errors = []
    for _ in range(40):
        scene.step()
        reading = noisy.read()
        assert_equal(noisy.read().force, reading.force)
        errors.append(reading.force - noisy.read_ground_truth().force)
    errors = torch.stack(errors)
    assert errors.mean().abs() < 0.015
    assert 0.035 < errors.std() < 0.07


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_temporal_response_and_selective_reset(n_envs, show_viewer):
    DT = 0.005
    TAU = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0, 0, -10)),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.1)),
        material=gs.materials.Rigid(rho=250),
    )
    options = {
        "entity_idx": floor.idx,
        "probe_local_pos": ((0, 0, 0),),
        "probe_local_normal": (0, 0, 1),
        "probe_radius": 0.25,
        "contact_sources": ("rigid",),
    }
    filtered = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(filter_tau=TAU),
        )
    )
    hysteretic = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(hysteresis_strength=0.25, hysteresis_tau=TAU),
        )
    )
    held = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(sample_rate=50),
        )
    )
    delayed = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=gs.sensors.TactileHardwareProfile(delay=2 * DT),
        )
    )
    history = scene.add_sensor(gs.sensors.SolverContactTaxel(**options, history_length=3))
    scene.build(n_envs=n_envs)
    for _ in range(80):
        scene.step()
    for sensor in (filtered, hysteretic, held, delayed, history):
        assert_allclose(sensor.read().force, (0, 0, -20), atol=0.01)

    selected = [0] if n_envs else None
    saved = held.read().force
    saved_copy = saved.clone()
    box.set_pos((0, 0, 1), envs_idx=selected)
    for step in range(1, 5):
        scene.step()
        assert_equal(filtered.read_ground_truth(envs_idx=selected).force, 0)
        assert_allclose(filtered.read(envs_idx=selected).force, (0, 0, -20 * math.exp(-step * DT / TAU)), atol=0.002)
        assert_allclose(
            hysteretic.read(envs_idx=selected).force, (0, 0, 5 * math.exp(-(step - 1) * DT / TAU)), atol=0.002
        )
        assert_allclose(delayed.read(envs_idx=selected).force, (0, 0, -20 if step <= 2 else 0), atol=0.01)
        assert_allclose(held.read(envs_idx=selected).force, (0, 0, -20 if step <= 3 else 0), atol=0.01)
        if step == 1:
            assert_allclose(
                history.read(envs_idx=selected).force, (((0, 0, 0),), ((0, 0, -20),), ((0, 0, -20),)), atol=0.01
            )
        if n_envs:
            assert_allclose(filtered.read(envs_idx=[1]).force, (0, 0, -20), atol=0.01)
    assert_equal(saved, saved_copy)
    before_reset = filtered.read().force
    scene.reset(envs_idx=selected)
    for sensor in (filtered, hysteretic, held, delayed, history):
        assert_equal(sensor.read(envs_idx=selected).force, 0)
    if n_envs:
        assert_equal(filtered.read(envs_idx=[1]).force, before_reset[1:2])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_crosstalk_wrench_and_gain_resampling(n_envs, show_viewer):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005, gravity=(0, 0, -10)), show_viewer=show_viewer)
    floor = scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.08, 0, 0.02)),
        material=gs.materials.Rigid(rho=31250),
    )
    options = {
        "entity_idx": floor.idx,
        "probe_local_pos": ((-0.1, 0, 0), (0, 0, 0), (0.1, 0, 0), (0.08, 0, 0)),
        "probe_local_normal": (0, 0, 1),
        "probe_radius": (0.03, 0.03, 0.03, 0),
        "contact_sources": ("rigid",),
        "crosstalk_strength": 0.5,
        "crosstalk_sigma": 0.1,
    }
    sensor = scene.add_sensor(gs.sensors.SolverContactTaxel(**options))
    randomized = scene.add_sensor(gs.sensors.SolverContactTaxel(**options, probe_gain_resample_range=(0.8, 1.2)))
    scene.build(n_envs=n_envs)
    for _ in range(80):
        scene.step()
    raw = sensor.read_ground_truth()
    assert_equal(raw.force[..., :2, :], 0)
    assert_equal(raw.force[..., 3, :], 0)
    assert_allclose(raw.force[..., 2, :], (0, 0, -20), atol=0.01)
    assert_allclose(raw.couple[..., 2, :], (0, -0.4, 0), atol=0.001)
    weights = torch.tensor((math.exp(-2), math.exp(-0.5), 1.0), dtype=gs.tc_float, device=gs.device)
    weights *= 0.5 / weights.sum()
    weights[2] += 0.5
    measured = sensor.read()
    assert_allclose(measured.force[..., :3, 2], -20 * weights, atol=0.01)
    x = torch.tensor((-0.1, 0, 0.1), dtype=gs.tc_float, device=gs.device)
    assert_allclose(measured.couple[..., :3, 1], (1.6 - 20 * x) * weights, atol=0.002)
    positions = torch.tensor(options["probe_local_pos"], dtype=gs.tc_float, device=gs.device)
    moments = measured.couple + positions.expand_as(measured.force).cross(measured.force, dim=-1)
    assert_allclose(measured.force.sum(dim=-2), (0, 0, -20), atol=0.01)
    assert_allclose(moments.sum(dim=-2), (0, 1.6, 0), atol=0.002)
    assert_equal(measured.force[..., 3, :], 0)
    assert_equal(measured.couple[..., 3, :], 0)

    gains_before = randomized.read().force[..., :3, 2] / measured.force[..., :3, 2]
    assert ((gains_before >= 0.8) & (gains_before <= 1.2)).all()
    assert gains_before.std() > 0.01
    scene.reset(envs_idx=[0] if n_envs else None)
    for _ in range(80):
        scene.step()
    gains_after = randomized.read().force[..., :3, 2] / sensor.read().force[..., :3, 2]
    assert ((gains_after >= 0.8) & (gains_after <= 1.2)).all()
    if n_envs:
        assert_allclose(gains_after[1], gains_before[1], atol=1e-7)
        assert not torch.equal(gains_after[0], gains_before[0])
    else:
        assert not torch.equal(gains_after, gains_before)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_drift_increments_and_jitter_bounds(n_envs, show_viewer):
    DT = 0.005
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, gravity=(0, 0, 0)), show_viewer=show_viewer)
    floor = scene.add_entity(gs.morphs.Plane())
    options = {"entity_idx": floor.idx, "probe_local_pos": ((0, 0, 0),), "contact_sources": ("rigid",)}
    drifting = scene.add_sensor(gs.sensors.SolverContactTaxel(**options, random_walk=0.05))
    jittered = scene.add_sensor(
        gs.sensors.SolverContactTaxel(**(options | {"probe_local_pos": ((0, 0, 0), (0.01, 0, 0))}), delay=DT, jitter=DT)
    )
    scene.build(n_envs=n_envs)
    previous = drifting.read().force
    increments = []
    observed_delays = []
    for step in range(1, 61):
        jittered.set_bias(step)
        scene.step()
        value = drifting.read().force
        increments.append(value - previous)
        previous = value
        assert_equal(drifting.read().force, value)
        assert_equal(drifting.read_ground_truth().force, 0)
        if step >= 3:
            delay = step - jittered.read().force
            assert ((delay == 1) | (delay == 2)).all()
            observed_delays.append(delay)
    increments = torch.stack(increments)
    assert increments.mean().abs() < 0.015
    assert 0.035 < increments.std() < 0.07
    observed_delays = torch.stack(observed_delays)
    assert (observed_delays == 1).any() and (observed_delays == 2).any()
    scene.reset(envs_idx=[0] if n_envs else None)
    assert_equal(drifting.read(envs_idx=[0] if n_envs else None).force, 0)
    if n_envs:
        assert_equal(drifting.read(envs_idx=[1]).force, previous[1:2])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_overflow_sample_delay_and_recovery(n_envs, show_viewer):
    DT = 0.0002
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0, 0, 0)),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.375, 0.375, 0.375),
            grid_density=64,
            particle_size=0.0078125,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(contact_buffer_capacity=1),
        show_viewer=show_viewer,
    )
    soft = scene.add_entity(
        gs.morphs.Box(size=(0.03125, 0.03125, 0.03125), pos=(0.125, 0.0625, 0.01)),
        material=gs.materials.MPM.Elastic(E=1000, rho=1000, sampler="regular"),
    )
    floor = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0))
    sensor = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            entity_idx=floor.idx,
            probe_local_pos=((0.125, 0.0625, 0),),
            probe_local_normal=(0, 0, 1),
            probe_radius=0.05,
            contact_sources=("hybrid",),
            hardware_profile=gs.sensors.TactileHardwareProfile(
                filter_tau=5 * DT,
                hysteresis_strength=0.25,
                hysteresis_tau=5 * DT,
                sample_rate=1 / (2 * DT),
                delay=2 * DT,
            ),
        )
    )
    scene.build(n_envs=n_envs)
    selected = [0] if n_envs else None
    soft.set_particles_vel((0, 0, -1), envs_idx=selected)
    scene.step()
    contributions = scene.get_contact_contributions(source="hybrid")
    assert contributions.dropped_count[0] > 0
    assert torch.isnan(sensor.read_ground_truth(envs_idx=selected).force).all()
    exact = floor.get_links_contact_wrench(source="hybrid")
    assert torch.isfinite(exact.force).all()
    assert exact.force[0, 0, 2] < 0
    assert_equal(sensor.read(envs_idx=selected).force, 0)
    soft.set_particles_active(False)
    scene.step()
    assert_equal(sensor.read_ground_truth().force, 0)
    assert_equal(sensor.read().force, 0)
    scene.step()
    assert torch.isnan(sensor.read(envs_idx=selected).force).all()
    assert not torch.atleast_1d(sensor.is_complete)[0]
    if n_envs:
        assert sensor.is_complete[1]
        assert_equal(sensor.read(envs_idx=[1]).force, 0)
    scene.step()
    assert_equal(sensor.read().force, 0)
    assert sensor.is_complete.all()
    assert_equal(floor.get_links_contact_wrench(source="hybrid").force, 0)
    scene.reset(envs_idx=selected)
    assert_equal(sensor.read(envs_idx=selected).force, 0)
    assert sensor.is_complete.all()


@pytest.mark.required
def test_sensor_snapshot_trajectory_and_partial_restore(show_viewer):
    DT = 0.005
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, gravity=(0, 0, -10)), show_viewer=show_viewer)
    floor = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.1)),
        material=gs.materials.Rigid(rho=250),
    )
    options = {
        "entity_idx": floor.idx,
        "probe_local_pos": ((-0.05, 0, 0), (0.05, 0, 0)),
        "probe_local_normal": (0, 0, 1),
        "probe_radius": 0.25,
        "contact_sources": ("rigid",),
        "history_length": 4,
    }
    profile = gs.sensors.TactileHardwareProfile(
        filter_tau=0.03,
        hysteresis_strength=0.25,
        hysteresis_tau=0.02,
        sample_rate=50,
        delay=2 * DT,
        gain_variation=(0.8, 1.2),
        gain_temp_coeff=0.01,
    )
    deterministic = scene.add_sensor(gs.sensors.SolverContactTaxel(**options, hardware_profile=profile))
    noisy = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **options,
            hardware_profile=profile.model_copy(
                update={
                    "noise": 0.02,
                    "drift": 0.01,
                    "jitter": DT,
                    "dead_taxel_probability": 0.25,
                    "dead_taxel_value_range": (2.0, 3.0),
                }
            ),
        )
    )
    scene.build(n_envs=2)
    for _ in range(80):
        scene.step()
    deterministic.set_temperature(35)
    deterministic.set_hysteresis_scale(0.7)
    snapshot = scene.get_state()
    saved_deterministic = deterministic.read().force
    saved_noisy = noisy.read().force
    saved_copy = saved_noisy.clone()
    expected_deterministic = []
    expected_noisy = []
    torch.manual_seed(17)
    box.set_pos((0, 0, 1), envs_idx=[0])
    for _ in range(6):
        scene.step()
        expected_deterministic.append(deterministic.read().force)
        expected_noisy.append(noisy.read().force)
    before_restore_deterministic = deterministic.read().force
    before_restore_noisy = noisy.read().force
    deterministic.set_temperature(65, envs_idx=[0])
    deterministic.set_hysteresis_scale(0.0, envs_idx=[0])
    deterministic.set_probe_gain(2.0, envs_idx=[0])
    noisy.set_noise(1.0, envs_idx=[0])
    noisy.set_random_walk(0.0, envs_idx=[0])
    noisy.set_bias(0.7, envs_idx=[0])
    noisy.set_resolution(2.0, envs_idx=[0])
    torch.manual_seed(121)
    expected_random_draw = torch.rand(8, device=gs.device)
    torch.manual_seed(121)
    scene.reset(snapshot, envs_idx=[0])
    assert_equal(torch.rand(8, device=gs.device), expected_random_draw)
    assert_equal(deterministic.read(envs_idx=[0]).force, saved_deterministic[0:1])
    assert_equal(noisy.read(envs_idx=[0]).force, saved_noisy[0:1])
    assert_equal(deterministic.read(envs_idx=[1]).force, before_restore_deterministic[1:2])
    assert_equal(noisy.read(envs_idx=[1]).force, before_restore_noisy[1:2])
    assert_equal(saved_noisy, saved_copy)
    # RNG belongs to the caller. Supplying the same seed reproduces future noisy samples without making a partial
    # state restore rewind the global random stream used by unselected environments or other consumers.
    torch.manual_seed(17)
    box.set_pos((0, 0, 1), envs_idx=[0])
    for expected, expected_random in zip(expected_deterministic, expected_noisy):
        scene.step()
        assert_allclose(deterministic.read(envs_idx=[0]).force, expected[0:1], atol=1e-5)
        assert_allclose(noisy.read(envs_idx=[0]).force, expected_random[0:1], atol=1e-5)
    scene.reset(snapshot, envs_idx=[0])
    assert_equal(noisy.read(envs_idx=[0]).force, saved_copy[0:1])
