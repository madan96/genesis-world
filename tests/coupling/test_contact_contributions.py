import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_contact_wrench(n_envs, show_viewer):
    DT = 0.005
    MASS = 2.0
    GRAVITY = 10.0
    POSITION = (0.3, 0.2, 0.1)
    REFERENCE = (0.1, -0.2, 0.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2,
            gravity=(0, 0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=POSITION,
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=POSITION,
        ),
        material=gs.materials.Rigid(rho=MASS / 0.2**3),
    )
    scene.build(n_envs=n_envs)
    box.get_contacts(is_padded=True)
    assert_equal(scene.get_contact_contributions(source="rigid").count, 0)
    for _ in range(80):
        scene.step()

    rows = scene.get_contact_contributions(source="rigid")
    assert_equal(scene.get_contact_contributions().force, rows.force)
    snapshot = rows.force.clone()
    assert rows.valid.any(dim=1).all()
    assert rows.is_complete.all()
    assert (rows.normal_force >= -1e-5).all()
    assert_equal(rows.count, rows.valid.sum(dim=1))
    assert_equal(rows.force[~rows.valid], 0)
    assert_allclose(rows.impulse, rows.force * rows.interval[..., None], atol=1e-7)
    assert_allclose(rows.interval[rows.valid], DT / 2, atol=1e-7)
    box_rows = box.get_contact_contributions(source="rigid")
    for name, value in vars(rows.select(entity_idx=box.idx)).items():
        assert_equal(vars(box_rows)[name], value)
    assert_equal(box_rows.valid, rows.valid)
    assert_equal(rows.select(region_idx=0).count, 0)
    assert_equal(rows.select(link_idx=box.link_start).valid, rows.valid)
    sign = torch.where(rows.entity_a == box.idx, 1.0, -1.0)
    assert_allclose((rows.force * sign[..., None]).sum(dim=1), (0, 0, MASS * GRAVITY), atol=0.01)
    assert (
        ((rows.entity_a == floor.idx) & (rows.entity_b == box.idx))
        | ((rows.entity_b == floor.idx) & (rows.entity_a == box.idx))
        | ~rows.valid
    ).all()
    assert_allclose((rows.normal * sign[..., None])[rows.valid], (0, 0, 1), atol=1e-6)

    wrench = scene.get_links_contact_wrench(source="rigid", reference=REFERENCE)
    box_wrench = box.get_links_contact_wrench(source="rigid", reference=REFERENCE)
    assert box_wrench.force.shape == (max(n_envs, 1), box.n_links, 3)
    for name, value in vars(box_wrench).items():
        scene_value = vars(wrench)[name]
        assert_equal(value, scene_value if name.endswith("interval") else scene_value[:, box.link_start : box.link_end])
    assert_allclose(wrench.force.sum(dim=1), 0, atol=1e-6)
    assert_allclose(wrench.torque.sum(dim=1), 0, atol=1e-6)
    assert_allclose(box_wrench.force, (0, 0, MASS * GRAVITY), atol=0.01)
    assert_allclose(box_wrench.torque, (8, -4, 0), atol=0.01)
    assert_allclose(box_wrench.center_of_pressure[..., :2], POSITION[:2], atol=1e-4)
    assert_allclose(box_wrench.impulse, box_wrench.force * DT / 2, atol=1e-7)

    for _ in range(2):
        scene.step()
    assert_equal(rows.force, snapshot)
    rows.force.zero_()
    origin_wrench = box.get_links_contact_wrench(source="rigid")
    assert_allclose(origin_wrench.force, (0, 0, MASS * GRAVITY), atol=0.01)
    assert_allclose(origin_wrench.torque, (4, -6, 0), atol=0.01)
    scene_snapshot = scene.get_state()
    if n_envs:
        scene.reset(envs_idx=[1])
    before_reset = scene.get_contact_contributions(source="rigid")
    expected_net_force = box.get_links_net_contact_force()
    scene.reset(envs_idx=[0] if n_envs else None)
    after_reset = scene.get_contact_contributions(source="rigid")
    assert_equal(after_reset.count[0], 0)
    if n_envs:
        assert_equal(after_reset.count[1], before_reset.count[1])
        assert_equal(after_reset.force[1], before_reset.force[1])

    for selection in ([0], torch.tensor([True, False], device=gs.device)) if n_envs else (None,):
        scene.reset(scene_snapshot, envs_idx=selection)
        restored = scene.get_contact_contributions(source="rigid")
        for name, value in vars(before_reset).items():
            assert_allclose(vars(restored)[name], value, atol=1e-8)
        assert_allclose(box.get_links_net_contact_force(), expected_net_force, atol=1e-8)
        scene.reset(envs_idx=selection)
        assert_equal(scene.get_contact_contributions(source="rigid").count[0], 0)
    scene.reset(scene_snapshot)
    scene.step()
    assert_allclose(box.get_links_contact_wrench(source="rigid").force, (0, 0, MASS * GRAVITY), atol=0.01)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("capacity", [1, 16384])
def test_mpm_contact_wrench(n_envs, capacity, show_viewer):
    SUBSTEPS = 4
    DT = SUBSTEPS * 2e-4
    REFERENCE = (0.02, -0.03, 0.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.375, 0.375, 0.375),
            grid_density=64,
            particle_size=0.0078125,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            contact_buffer_capacity=capacity,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.3, 0.2),
            camera_lookat=(0.125, 0.0625, 0.01),
        ),
        show_viewer=show_viewer,
    )
    soft = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.03125, 0.03125, 0.03125),
            pos=(0.125, 0.0625, 0.01),
        ),
        # Symmetric sampling makes the expected contact moment independent of grid interpolation.
        material=gs.materials.MPM.Elastic(E=1000, rho=1000, sampler="regular"),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(-0.1, -0.1, 0.0199),
        ),
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_friction=0.5),
    )
    scene.build(n_envs=n_envs)
    initial_state = scene.get_state()
    floor.set_coup_friction_scale(((0,), (1,)) if n_envs else 0.25)
    saved_scale = floor.get_coup_friction_scale()
    friction_snapshot = scene.get_state()
    # Environment 0 hits the floor head-on, so its contact moment follows from the particle centroid. Environment 1
    # slides along x on a floor with friction coefficient 0.5.
    soft.set_particles_vel((0, 0, -1))
    if n_envs:
        soft.set_particles_vel((1, 0, -1), envs_idx=[1])
    box.set_dofs_velocity((0, 0, -0.5, 0, 0, 0))
    mass = soft.get_mass()
    center = soft.get_particles_pos().reshape(max(n_envs, 1), -1, 3).mean(dim=1)
    velocity_before = soft.get_particles_vel().reshape(max(n_envs, 1), -1, 3).mean(dim=1)
    scene.step()
    velocity_after = soft.get_particles_vel().reshape(max(n_envs, 1), -1, 3).mean(dim=1)
    expected_impulse = mass[:, None] * (velocity_before - velocity_after)

    rows = scene.get_contact_contributions(source="hybrid")
    rigid_rows = scene.get_contact_contributions(source="rigid")
    assert rows.valid.any(dim=1).all()
    assert_equal(rows.entity_a[rows.valid], floor.idx)
    assert_equal(rows.entity_b[rows.valid], soft.idx)
    assert_equal(rows.link_a[rows.valid], floor.link_start)
    assert_equal(rows.link_b[rows.valid], -1)
    assert (rows.region_b[rows.valid] >= 0).all()
    assert ((rows.substep[rows.valid] >= 0) & (rows.substep[rows.valid] < SUBSTEPS)).all()
    assert_allclose(rows.normal[rows.valid], (0, 0, -1), atol=1e-6)
    assert (rows.normal_force >= -1e-6).all()
    assert_allclose(rows.impulse, rows.force * rows.interval[..., None], atol=1e-8)
    assert_allclose(rows.interval[rows.valid], DT, atol=1e-9)
    assert_equal(rows.select(entity_idx=soft.idx).valid, rows.valid)
    assert rigid_rows.valid.any(dim=1).all()
    assert_equal(rigid_rows.select(entity_idx=box.idx).valid, rigid_rows.valid)
    assert_equal(scene.get_contact_contributions().count, rows.count + rigid_rows.count)

    wrench = floor.get_links_contact_wrench(source="hybrid", reference=REFERENCE)
    assert (expected_impulse[..., 2] < -1e-5).all()
    assert_allclose(wrench.impulse[:, 0], expected_impulse, atol=1e-7)
    assert_allclose(wrench.force[:, 0], expected_impulse / DT, atol=5e-4)
    assert_allclose(wrench.hybrid_interval, DT, atol=1e-9)
    assert_allclose(wrench.rigid_interval, 0, atol=0)
    assert_allclose(wrench.impulse[:, 0, 0] / -wrench.impulse[:, 0, 2], (0, 0.5) if n_envs else 0, atol=1e-5)
    arm = center - torch.tensor(REFERENCE, dtype=gs.tc_float, device=gs.device)
    assert_allclose(wrench.angular_impulse[:1, 0], arm[:1].cross(expected_impulse[:1], dim=-1), atol=1e-7)
    assert_allclose(wrench.center_of_pressure[:1, 0, :2], center[:1, :2], atol=1e-5)
    assert (wrench.area > 0).all()
    assert (wrench.peak_normal_force > 0).all()
    if capacity == 1:
        assert (rows.dropped_count > 0).all()
        assert not rows.is_complete.any()
        assert_equal(rows.count, 1)
    else:
        assert rows.is_complete.all()
        assert_allclose(rows.impulse.sum(dim=1), expected_impulse, atol=1e-7)
        moments = (rows.position - torch.tensor(REFERENCE, dtype=gs.tc_float, device=gs.device)).cross(
            rows.impulse, dim=-1
        )
        assert_allclose(moments.sum(dim=1), wrench.angular_impulse[:, 0], atol=1e-7)
        assert_allclose(rows.area.sum(dim=1), wrench.area[:, 0], atol=1e-7)

    # The floor link receives both the box's rigid contact and the particle contact.
    all_wrench = scene.get_links_contact_wrench()
    rigid_wrench = scene.get_links_contact_wrench(source="rigid")
    hybrid_wrench = scene.get_links_contact_wrench(source="hybrid")
    assert (rigid_wrench.normal_force[:, floor.link_start] > 0).all()
    for name in ("force", "torque", "impulse", "angular_impulse", "normal_force", "area"):
        assert_allclose(vars(all_wrench)[name], vars(rigid_wrench)[name] + vars(hybrid_wrench)[name], atol=1e-5)
    assert_equal(
        all_wrench.peak_normal_force, torch.maximum(rigid_wrench.peak_normal_force, hybrid_wrench.peak_normal_force)
    )
    assert_allclose(all_wrench.rigid_interval, DT / SUBSTEPS, atol=1e-9)
    assert_allclose(all_wrench.hybrid_interval, DT, atol=1e-9)

    saved_force = rows.force.clone()
    scene_snapshot = scene.get_state()
    scene.step()
    continued_pos = soft.get_particles_pos().clone()
    continued_impulse = scene.get_links_contact_wrench(source="hybrid").impulse
    scene.reset(scene_snapshot)
    restored = scene.get_contact_contributions(source="hybrid")
    for name, value in vars(rows).items():
        assert_equal(vars(restored)[name], value)
    if n_envs:
        scene.reset(envs_idx=[1])
    expected_rows = scene.get_contact_contributions(source="hybrid")
    expected_wrench = scene.get_links_contact_wrench(source="hybrid")
    scene.reset(envs_idx=[0] if n_envs else None)
    reset_rows = scene.get_contact_contributions(source="hybrid")
    reset_wrench = scene.get_links_contact_wrench(source="hybrid")
    assert_equal(reset_rows.count[0], 0)
    assert_equal(reset_rows.dropped_count[0], 0)
    assert_equal(reset_wrench.hybrid_interval[0], 0)
    assert_equal(reset_wrench.impulse[0], 0)
    if n_envs:
        assert_equal(reset_rows.force[1], expected_rows.force[1])
        assert_equal(reset_rows.dropped_count[1], expected_rows.dropped_count[1])
        assert_equal(reset_wrench.impulse[1], expected_wrench.impulse[1])
    for selection in ([0], torch.tensor([True, False], device=gs.device)) if n_envs else (None,):
        scene.reset(scene_snapshot, envs_idx=selection)
        restored = scene.get_contact_contributions(source="hybrid")
        restored_wrench = scene.get_links_contact_wrench(source="hybrid")
        for name, value in vars(expected_rows).items():
            assert_equal(vars(restored)[name], value)
        for name, value in vars(expected_wrench).items():
            assert_equal(vars(restored_wrench)[name], value)
        scene.reset(envs_idx=selection)
    assert_equal(rows.force, saved_force)
    rows.force.zero_()
    assert_equal(scene.get_contact_contributions(source="hybrid").force, reset_rows.force)

    scene.reset(scene_snapshot)
    scene.step()
    assert_allclose(soft.get_particles_pos(), continued_pos, atol=1e-8)
    assert_allclose(scene.get_links_contact_wrench(source="hybrid").impulse, continued_impulse, atol=1e-8)

    # Resetting to a state registers it, so the post-build state is registered again before a plain reset.
    scene.reset(initial_state)
    floor.set_coup_friction_scale(0.5)
    scene.reset()
    assert_equal(floor.get_coup_friction_scale(), 1)
    selections = ([0], torch.tensor([True, False], device=gs.device)) if n_envs else (None,)
    for selection in selections:
        scene.reset(initial_state)
        assert_equal(floor.get_coup_friction_scale(), 1)
        floor.set_coup_friction_scale(0.5)
        scene.reset(friction_snapshot, envs_idx=selection)
        soft.set_particles_vel((1, 0, -1))
        scene.step()
        wrench = floor.get_links_contact_wrench(source="hybrid")
        assert_allclose(wrench.impulse[:, 0, 0] / -wrench.impulse[:, 0, 2], (0, 0.25) if n_envs else 0.125, atol=1e-5)
        assert_equal(floor.get_coup_friction_scale(), ((0,), (0.5,)) if n_envs else 0.25)
    assert_equal(saved_scale, ((0,), (1,)) if n_envs else 0.25)
    saved_scale.zero_()
    assert_equal(floor.get_coup_friction_scale(), ((0,), (0.5,)) if n_envs else 0.25)
    for selection in selections:
        floor.set_coup_friction_scale(0.125, envs_idx=selection)
        assert_equal(floor.get_coup_friction_scale(), ((0.125,), (0.5,)) if n_envs else 0.125)
    if n_envs:
        floor.set_coup_friction_scale(((0.25,), (0.75,)), envs_idx=[1, 0])
        assert_equal(floor.get_coup_friction_scale(), ((0.75,), (0.25,)))
        assert_equal(floor.get_coup_friction_scale(envs_idx=[1, 0]), ((0.25,), (0.75,)))
        floor.set_coup_friction_scale(0.5, envs_idx=-1)
        assert_equal(floor.get_coup_friction_scale(), ((0.75,), (0.5,)))
        floor.set_coup_friction_scale(0.125, envs_idx=range(-n_envs, 0))
        assert_equal(floor.get_coup_friction_scale(), 0.125)
        floor.set_coup_friction_scale(((0.25,), (0.75,)), envs_idx=range(n_envs - 1, -1, -1))
        assert_equal(floor.get_coup_friction_scale(), ((0.75,), (0.25,)))
        floor.set_coup_friction_scale(((0.5,), (0.125,)), envs_idx=slice(None, None, -1))
        assert_equal(floor.get_coup_friction_scale(), ((0.125,), (0.5,)))
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(gs.GenesisException, match="finite and nonnegative"):
            floor.set_coup_friction_scale(value)
    with pytest.raises(gs.GenesisException):
        floor.set_coup_friction_scale(1, geoms_idx_local=[floor.n_geoms])
    with pytest.raises(gs.GenesisException):
        floor.set_coup_friction_scale(1, envs_idx=n_envs)

    soft.set_particles_active(False)
    scene.step()
    assert_equal(scene.get_contact_contributions(source="hybrid").count, 0)
    empty_wrench = scene.get_links_contact_wrench(source="hybrid")
    assert_allclose(empty_wrench.hybrid_interval, DT, atol=1e-9)
    assert_equal(empty_wrench.force, 0)
