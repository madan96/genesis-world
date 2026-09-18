import math

import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("n_envs", [0, 2])
def test_solver_taxel_many_contact_reduction(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            substeps=10,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.375, 0.375, 0.375),
            grid_density=64,
            particle_size=0.0078125,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            enable_contact_telemetry=True,
            contact_buffer_capacity=16384,
        ),
        show_viewer=show_viewer,
    )
    soft = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.125, 0.0625, 0.01),
            size=(0.125, 0.125, 0.03125),
        ),
        material=gs.materials.MPM.Elastic(
            E=1000,
            sampler="regular",
        ),
        surface=gs.surfaces.Default(
            vis_mode="particle",
        ),
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0,
        ),
    )
    points = ((0.05, -0.05, 0), (0.05, 0.2, 0), (0.3, -0.05, 0), (0.3, 0.2, 0))
    sensor = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            entity_idx=floor.idx,
            probe_local_pos=points,
            probe_local_normal=(0, 0, 1),
            probe_radius=1.0,
            frame="world",
        ),
    )
    scene.build(n_envs=n_envs)
    soft.set_particles_vel((0, 0, -1))
    scene.step()

    rows = scene.get_contact_contributions(source="hybrid")
    assert rows.is_complete.all()
    assert (rows.count > 1000).all()
    reading = sensor.read_ground_truth()
    force = reading.force.double().reshape(max(n_envs, 1), 4, 3)
    couple = reading.couple.double().reshape_as(force)
    positions = torch.tensor(points, dtype=torch.float64, device=gs.device)[None]
    # A float64 reference isolates the float32 accumulation error in the observed forces and moments.
    reference_force = rows.force.to(device="cpu", dtype=torch.float64)
    reference_position = rows.position.to(device="cpu", dtype=torch.float64)
    assert_allclose(force.sum(dim=1), reference_force.sum(dim=1), atol=1e-5, rtol=1e-6)
    wrench = floor.get_links_contact_wrench(source="hybrid")
    assert_allclose(force.sum(dim=1), wrench.force[:, 0], atol=1e-5, rtol=1e-6)
    assert_allclose(
        (positions.cross(force, dim=-1) + couple).sum(dim=1),
        reference_position.cross(reference_force, dim=-1).sum(dim=1),
        atol=1e-5,
        rtol=1e-6,
    )
    assert_allclose((positions.cross(force, dim=-1) + couple).sum(dim=1), wrench.torque[:, 0], atol=1e-5, rtol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_solver_taxel_rigid_wrench_frames_and_support(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=2,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.3, 0.2, 0.1),
            euler=(0.0, 90.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=2.0 / (4.0 * math.pi * 0.1**3 / 3.0),
            friction=0.01,
        ),
    )
    # +x of the link points down after the 90-degree rotation around y. The third probe is inactive padding.
    shared = dict(
        entity_idx=sphere.idx,
        probe_local_pos=((0.1, -0.02, 0.0), (0.1, 0.02, 0.0), (0.1, 0.0, 0.0)),
        probe_local_normal=(1.0, 0.0, 0.0),
        probe_radius=(0.05, 0.05, 0.0),
        support_area=(0.001, 0.002, 0.001),
        contact_sources=("rigid",),
        filter_entities=(floor.idx,),
    )
    local = scene.add_sensor(gs.sensors.SolverContactTaxel(**shared))
    world = scene.add_sensor(gs.sensors.SolverContactTaxel(**shared, frame="world"))
    fallback = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            **{
                **shared,
                "probe_local_pos": ((0.1, 0.03, 0.0), (0.1, 0.1, 0.0), (0.1, 0.0, 0.0)),
                "probe_radius": (0.0001, 0.0001, 0.0),
            },
        )
    )
    excluded = scene.add_sensor(gs.sensors.SolverContactTaxel(**{**shared, "filter_entities": (sphere.idx,)}))
    scene.build(n_envs=n_envs)
    for _ in range(80):
        scene.step()

    local_data = local.read_ground_truth()
    world_data = world.read_ground_truth()
    # Mesh-derived rigid mass can differ from the analytical sphere by its geometric approximation error.
    load = float(sphere.get_mass().reshape(-1)[0]) * 10.0
    assert_allclose(load, 20.0, atol=0.1)
    assert_allclose(local_data.force, ((-load / 2, 0, 0), (-load / 2, 0, 0), (0, 0, 0)), atol=0.01)
    assert_allclose(local_data.couple, ((0, 0, load * 0.01), (0, 0, -load * 0.01), (0, 0, 0)), atol=0.001)
    assert_allclose(world_data.force, ((0, 0, load / 2), (0, 0, load / 2), (0, 0, 0)), atol=0.01)
    assert_allclose(world_data.couple, ((load * 0.01, 0, 0), (-load * 0.01, 0, 0), (0, 0, 0)), atol=0.001)
    assert_allclose(fallback.read_ground_truth().force, ((-load, 0, 0), (0, 0, 0), (0, 0, 0)), atol=0.01)
    assert_allclose(fallback.read_ground_truth().couple, ((0, 0, -load * 0.03), (0, 0, 0), (0, 0, 0)), atol=0.001)
    assert_equal(excluded.read_ground_truth().force, 0)
    assert_equal(excluded.read_ground_truth().couple, 0)
    assert_allclose(local.normal_force(ground_truth=True), (load / 2, load / 2, 0), atol=0.01)
    assert_allclose(local.pressure(ground_truth=True), (load / 0.002, load / 0.004, 0), atol=10.0)
    assert local.is_complete.all()
    assert_equal(local.read().force, local_data.force)
    assert_allclose(world.taxel_normals, ((0, 0, -1), (0, 0, -1), (0, 0, -1)), atol=1e-6)
    bulk = scene.read_sensors()[gs.sensors.types.SolverContactTaxel]
    assert_equal(bulk[..., :9].reshape(local_data.force.shape), local_data.force)
    assert_equal(bulk[..., 9:18].reshape(local_data.couple.shape), local_data.couple)

    saved = local_data.force.clone()
    normals = local.taxel_normals
    normals.zero_()
    assert_allclose(local.taxel_normals, ((1, 0, 0), (1, 0, 0), (1, 0, 0)), atol=1e-6)
    sphere.set_pos((0.3, 0.2, 0.4), envs_idx=[0] if n_envs else None)
    scene.step()
    assert_equal(local_data.force, saved)
    assert_equal(local.read_ground_truth(envs_idx=[0] if n_envs else None).force, 0)
    if n_envs:
        assert_allclose(local.read_ground_truth(envs_idx=[1]).force, saved[1:2], atol=0.01)
    scene.reset(envs_idx=[0] if n_envs else None)
    assert_equal(local.read_ground_truth(envs_idx=[0] if n_envs else None).force, 0)
