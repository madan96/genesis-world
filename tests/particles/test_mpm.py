import math

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("grid_mode", ["fixed_dense", "moving_window"])
def test_grid_state_reset(grid_mode, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            substeps_local=4,
            gravity=(0, 0, -2),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.04,
            lower_bound=(-1, -1, -1),
            upper_bound=(1, 1, 1),
            grid_mode=grid_mode,
            window_size=(1, 1, 1) if grid_mode == "moving_window" else None,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Elastic(
            E=1000,
            nu=0.2,
            sampler="regular",
        ),
    )
    scene.build(n_envs=3)
    solver = scene.sim.mpm_solver
    centers = torch.tensor([[-0.3, 0, 0.1], [0.25, 0, 0.15], [0, -0.2, 0.2]], dtype=gs.tc_float, device=gs.device)
    velocity = torch.tensor([[0.4, 0, 0.1], [-0.2, 0.1, 0.2], [0.3, -0.2, 0]], dtype=gs.tc_float, device=gs.device)
    initial_position = entity.get_particles_pos().clone() + centers[:, None]
    entity.set_position(centers)
    entity.set_velocity(velocity[:, None])
    scene.step()
    snapshot = scene.get_state()
    saved_diagnostics = solver.get_grid_diagnostics()
    entity.set_particles_active(False, envs_idx=[2])
    for _ in range(2):
        scene.step()
    before = entity.get_particles_pos().clone()
    diagnostics = solver.get_grid_diagnostics()
    assert_allclose(diagnostics.active_particles, [entity.n_particles, entity.n_particles, 0], atol=0)
    selection = torch.tensor([True, False, True], device=gs.device)
    scene.reset(snapshot, envs_idx=selection)
    assert_allclose(entity.get_particles_pos(envs_idx=[1]), before[1:2], atol=1e-7)
    restored_diagnostics = solver.get_grid_diagnostics()
    expected_origin = diagnostics.origin_cell.clone()
    expected_origin[selection] = saved_diagnostics.origin_cell[selection]
    assert_allclose(restored_diagnostics.origin_cell, expected_origin, atol=0)
    assert_allclose(restored_diagnostics.active_particles, entity.n_particles, atol=0)
    assert_allclose(restored_diagnostics.active_cell_substeps[selection], 0, atol=0)
    assert_allclose(restored_diagnostics.active_cell_substeps[1], diagnostics.active_cell_substeps[1], atol=0)
    if grid_mode == "moving_window":
        assert restored_diagnostics.allocated_cells_per_env < restored_diagnostics.dense_cells_per_env / 5
        assert not torch.equal(expected_origin[0], expected_origin[1])
    scene.step()
    steps = torch.tensor([2, 4, 2], dtype=gs.tc_float, device=gs.device)
    expected_velocity = velocity.clone()
    expected_velocity[:, 2] -= 2 * steps * scene.dt
    displacement = velocity * (steps * scene.dt)[:, None]
    displacement[:, 2] -= steps * (steps + 1) * scene.dt**2
    assert_allclose(entity.get_particles_vel(), expected_velocity[:, None], atol=3e-5)
    assert_allclose(entity.get_particles_pos(), initial_position + displacement[:, None], atol=2e-6)
    # The saved state owns its buffers and can be restored again using integer indices after further stepping.
    scene.reset(snapshot, envs_idx=[1])
    expected_velocity[1] = velocity[1]
    expected_velocity[1, 2] -= 2 * scene.dt
    assert_allclose(entity.get_particles_vel(), expected_velocity[:, None], atol=3e-5)


@pytest.mark.required
def test_window_stencil_recovery(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.04,
            lower_bound=(-1, -1, -1),
            upper_bound=(1, 1, 1),
            grid_mode="moving_window",
            window_size=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Elastic(
            E=1000,
            nu=0.2,
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)
    solver = scene.sim.mpm_solver
    initial = entity.get_particles_pos().clone()
    displaced = initial[1:2].clone()
    displaced[:, 0, 0] = 0.6
    entity.set_particles_pos(displaced, envs_idx=[1])
    with pytest.raises(gs.GenesisException, match="stencil padding"):
        for _ in range(11):
            scene.step()
    assert_allclose(solver.get_grid_diagnostics().window_too_small, [False, True], atol=0)
    scene.reset(envs_idx=[1])
    scene.step()
    assert_allclose(solver.get_grid_diagnostics().window_too_small, False, atol=0)
    assert_allclose(entity.get_particles_pos(), initial, atol=2e-6)
    assert_allclose(entity.get_particles_vel(), 0, atol=3e-5)
    invalid = initial[1:2].clone()
    invalid[:, 0, 0] = float("nan")
    entity.set_particles_pos(invalid, envs_idx=[1])
    with pytest.raises(gs.GenesisException, match="finite"):
        solver.check_errno()
    scene.reset(envs_idx=[1])
    solver.check_errno()
    assert_allclose(solver.get_grid_diagnostics().invalid_particle_state, False, atol=0)


@pytest.mark.required
def test_muscle_state_reset(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            substeps_local=4,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.04,
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Muscle(
            E=1000,
            nu=0.2,
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)
    initial = entity.get_particles_pos().clone()
    entity.set_particles_actu(0.4)
    snapshot = scene.get_state()
    for _ in range(2):
        scene.step()
    reference_position = entity.get_particles_pos().clone()
    reference_velocity = entity.get_particles_vel().clone()
    assert_allclose(entity.get_particles_actu(), 0.4, atol=1e-7)
    assert_allclose(reference_position.mean(dim=1), initial.mean(dim=1), atol=1e-7)
    assert (reference_velocity.norm(dim=-1).max(dim=1).values > 1e-4).all()
    assert (reference_position[:, :, 2].amax(dim=1) < initial[:, :, 2].amax(dim=1)).all()
    entity.set_particles_actu(0, envs_idx=[0])
    entity.set_particles_actu(-0.2, envs_idx=[1])
    scene.reset(snapshot, envs_idx=[0])
    assert_allclose(entity.get_particles_actu(), [[0.4], [-0.2]], atol=1e-7)
    for _ in range(2):
        scene.step()
    assert_allclose(entity.get_particles_pos(envs_idx=[0]), reference_position[:1], atol=2e-6)
    assert_allclose(entity.get_particles_vel(envs_idx=[0]), reference_velocity[:1], atol=3e-5)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_window_world_contact(n_envs, show_viewer):
    final_states = {}
    for grid_mode in ("fixed_dense", "moving_window"):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.002,
                gravity=(0, 0, 0),
            ),
            mpm_options=gs.options.MPMOptions(
                grid_density=16,
                particle_size=0.04,
                lower_bound=(-2, -0.5, -0.5),
                upper_bound=(2, 0.5, 0.5),
                grid_mode=grid_mode,
                window_size=(0.75, 0.75, 0.75) if grid_mode == "moving_window" else None,
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(1, 0, 0),
                size=(0.2, 0.5, 0.5),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                coup_friction=0,
            ),
        )
        entity = scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.12, 0.12, 0.12),
            ),
            material=gs.materials.MPM.Elastic(
                E=1000,
                nu=0.2,
                sampler="regular",
            ),
        )
        scene.build(n_envs=n_envs)
        solver = scene.sim.mpm_solver
        initial = entity.get_particles_pos().clone()
        initial_origin = solver.get_grid_diagnostics().origin_world
        velocity = torch.tensor([[4, 0, 0], [-4, 0, 0]][: max(n_envs, 1)], dtype=gs.tc_float, device=gs.device)
        entity.set_particles_vel(velocity[:, None] if n_envs else velocity[0])
        for _ in range(100):
            scene.step()
        expected_velocity = velocity[:, None] if n_envs else velocity[0]
        assert_allclose(entity.get_particles_vel(), expected_velocity, atol=5e-5)
        assert_allclose(entity.get_particles_pos() - initial, expected_velocity * (100 * scene.dt), atol=1e-5)
        if grid_mode == "moving_window":
            displacement = solver.get_grid_diagnostics().origin_world - initial_origin
            assert (displacement[:, 0].abs() > 0.5).all()
        for _ in range(50):
            scene.step()
        positions = entity.get_particles_pos()
        velocities = entity.get_particles_vel()
        if n_envs == 0:
            positions, velocities = positions[None], velocities[None]
        diagnostics = solver.get_grid_diagnostics()
        assert_allclose(diagnostics.active_particles, entity.n_particles, atol=0)
        assert (diagnostics.active_cell_substeps > 0).all()
        assert (diagnostics.active_cell_substeps <= diagnostics.allocated_cells_per_env * scene.substeps).all()
        # Grid-node coupling stops the material at the first grid node inside the wall face at x = 0.9
        wall_node_x = math.ceil(0.9 / solver.dx) * solver.dx
        assert_allclose(positions[0, :, 0].max(), wall_node_x, atol=1e-3)
        assert velocities[0, :, 0].mean() < 0.4
        if n_envs:
            assert_allclose(entity.get_particles_vel(envs_idx=[1]), velocity[1], atol=5e-5)
            assert_allclose(
                entity.get_particles_pos(envs_idx=[1]) - initial[1:2], velocity[1] * (150 * scene.dt), atol=1e-5
            )
        final_states[grid_mode] = (positions, velocities)
    assert_allclose(final_states["moving_window"][0], final_states["fixed_dense"][0], atol=1e-5)
    assert_allclose(final_states["moving_window"][1], final_states["fixed_dense"][1], atol=5e-5)


@pytest.mark.required
def test_particle_constraints(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=20,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.2, 1.0),
            camera_lookat=(0.0, 0.3, 0.35),
            camera_fov=32,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    rigid_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.55),
            size=(0.12, 0.12, 0.05),
            fixed=True,
        ),
    )
    mpm_cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4,
            nu=0.3,
            rho=1000,
        ),
        morph=gs.morphs.Box(
            pos=(0, 0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
    )
    # Two copies of one box offset along its local y axis, each placed with its own member pose
    box = trimesh.creation.box(
        extents=(0.04, 0.12, 0.04), transform=trimesh.transformations.translation_matrix((0.0, 0.1, 0.0))
    )
    mpm_mesh_set = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(box, box),
            poss=((0.5, 0.5, 0.3), (-0.5, 0.5, 0.3)),
            eulers=((0.0, 0.0, 90.0), (90.0, 0.0, 0.0)),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)

    # Each member is rotated about its own position: the box offset lands at -x for the member rotated about z and at
    # +z for the one rotated about x, and the long axis follows. A regular lattice spans the box minus one particle.
    particles = mpm_mesh_set.init_particles
    particle_size = mpm_mesh_set.particle_size
    long_span, short_span = 0.12 - particle_size, 0.04 - particle_size
    particles_z90 = particles[particles[:, 0] > 0.0]
    particles_x90 = particles[particles[:, 0] < 0.0]
    assert_allclose(particles_z90.mean(axis=0), (0.4, 0.5, 0.3), tol=tol)
    assert_allclose(np.ptp(particles_z90, axis=0), (long_span, short_span, short_span), tol=tol)
    assert_allclose(particles_x90.mean(axis=0), (-0.5, 0.5, 0.4), tol=tol)
    assert_allclose(np.ptp(particles_x90, axis=0), (short_span, short_span, long_span), tol=tol)
    # The visual mesh carries the same member poses
    assert_allclose(mpm_mesh_set.vmesh.verts.min(axis=0), (-0.52, 0.48, 0.28), tol=tol)
    assert_allclose(mpm_mesh_set.vmesh.verts.max(axis=0), (0.46, 0.52, 0.46), tol=tol)
    # Positioning the set as a whole moves the mean member position there, keeping the members' layout
    mpm_mesh_set.set_position((0.0, 0.5, 0.5))
    mpm_mesh_set.process_input()
    assert_allclose(mpm_mesh_set.get_particles_pos(), particles + (0.0, 0.0, 0.2), tol=tol)

    # Test get_particles_in_bbox - returns (n_envs, n_particles) mask
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    assert mask.shape == (2, mpm_cube.n_particles), "mask should be (n_envs, n_particles)"
    assert mask.any(), "bbox should select some particles"
    assert not mask.all(), "bbox should not select all particles"

    # Attach and test following
    link_idx = rigid_box.links[0].idx
    mpm_cube.set_particle_constraints(mask, link_idx, stiffness=1e5)
    initial_rigid_pos = rigid_box.get_pos()
    initial_mpm_x = mpm_cube.get_particles_pos()[:, mask[0], 0].mean()

    pos_diff = torch.tensor([0.2, 0, 0], device=gs.device)
    rigid_box.set_pos(initial_rigid_pos + pos_diff, zero_velocity=False)
    for _ in range(30):
        scene.step()

    mpm_diff = mpm_cube.get_particles_pos()[:, mask[0], 0].mean() - initial_mpm_x
    assert mpm_diff > pos_diff[0] * 0.3, f"MPM should follow rigid link. Got {mpm_diff:.3f}"


@pytest.mark.required
@pytest.mark.parametrize("variant", ["svd", "no_svd"])
def test_perf_dispatch(variant, show_viewer):
    if variant == "svd":
        elastic_model = "corotation"
        liquid_viscous = True
    else:
        elastic_model = "neohooken"
        liquid_viscous = False

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-0.4, -0.4, -0.05),
            upper_bound=(0.4, 0.4, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    elastic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.1, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Elastic(
            E=3e5,
            nu=0.2,
            rho=200.0,
            model=elastic_model,
        ),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Liquid(
            viscous=liquid_viscous,
        ),
    )
    scene.build(n_envs=2)

    # Aggregate SVD flag must match the variant for the dispatch path under test to actually run.
    assert scene.sim.mpm_solver.needs_svd == (variant == "svd")

    init_elastic_pos = elastic.get_particles_pos()
    init_liquid_pos = liquid.get_particles_pos()

    for _ in range(100):
        scene.step()

    final_elastic_pos = elastic.get_particles_pos()
    final_liquid_pos = liquid.get_particles_pos()

    # Gravity acted: particles moved down on average.
    assert init_elastic_pos[..., 2].mean() - final_elastic_pos[..., 2].mean() > 0.05
    assert init_liquid_pos[..., 2].mean() - final_liquid_pos[..., 2].mean() > 0.05

    # No ground penetration (plane is at z=0). Non-viscous liquid spreads thin enough that a sub-grid-cell penetration
    # is normal for the MPM coupling, so allow up to 2mm.
    assert final_elastic_pos[..., 2].min() > -1e-3
    assert final_liquid_pos[..., 2].min() > -2e-3
