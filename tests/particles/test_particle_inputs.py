import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("input_kind", ["position", "velocity", "active", "actuation"])
def test_queued_particle_input_selection(input_kind, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.04,
            lower_bound=(-1.0, -1.0, -1.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        gs.morphs.Box(
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Muscle(
            E=1000.0,
            nu=0.2,
            sampler="regular",
        ),
    )
    scene.build(n_envs=3)
    initial = entity.get_particles_pos().clone()
    mask = torch.tensor((False, True, False), dtype=torch.bool, device=gs.device)
    if input_kind == "position":
        entity.set_position((0.2, -0.1, 0.3), envs_idx=mask)
        assert_equal(entity.get_particles_pos(), initial)
        entity.process_input()
        expected = initial.clone()
        expected[1] += torch.tensor((0.2, -0.1, 0.3), dtype=gs.tc_float, device=gs.device)
        assert_allclose(entity.get_particles_pos(), expected, atol=1e-7)
    elif input_kind == "velocity":
        entity.set_velocity((0.2, -0.1, 0.3), envs_idx=[1])
        assert_equal(entity.get_particles_vel(), 0)
        entity.process_input()
        expected = torch.zeros_like(initial)
        expected[1] = torch.tensor((0.2, -0.1, 0.3), dtype=gs.tc_float, device=gs.device)
        assert_allclose(entity.get_particles_vel(), expected, atol=1e-7)
    elif input_kind == "active":
        entity.set_active(False, envs_idx=mask)
        assert_equal(entity.get_particles_active(), True)
        entity.process_input()
        expected = torch.ones_like(entity.get_particles_active())
        expected[1] = False
        assert_equal(entity.get_particles_active(), expected)
        entity.set_active(True, envs_idx=[1])
        entity.process_input()
        assert_equal(entity.get_particles_active(), True)
        entity.deactivate()
        entity.process_input()
        assert_equal(entity.get_particles_active(), False)
        entity.activate()
        entity.process_input()
        assert_equal(entity.get_particles_active(), True)
    else:
        target = torch.tensor(0.3, dtype=gs.tc_float, device=gs.device)
        selected = torch.tensor([1], dtype=gs.tc_int, device=gs.device)
        entity.set_actuation(target, envs_idx=selected)
        target.fill_(0.8)
        selected.fill_(0)
        assert_equal(scene.get_state().solvers_state[scene.solvers.index(scene.mpm_solver)].actu, 0)
        entity.process_input()
        expected = torch.zeros((3, entity.n_particles), dtype=gs.tc_float, device=gs.device)
        expected[1] = 0.3
        assert_allclose(
            scene.get_state().solvers_state[scene.solvers.index(scene.mpm_solver)].actu, expected, atol=1e-7
        )


@pytest.mark.required
@pytest.mark.parametrize("input_kind", ["position", "velocity"])
def test_queued_input_gradients(input_kind, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            substeps_local=2,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=16,
            particle_size=0.04,
            lower_bound=(-1.0, -1.0, -1.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Elastic(
            E=1000.0,
            nu=0.2,
            sampler="regular",
        ),
    )
    scene.build(n_envs=3)
    initial = gs.tensor([[0.05, 0.02, -0.03]] * 3, requires_grad=True)
    replacement = gs.tensor([0.02, -0.01, 0.04], requires_grad=True)
    if input_kind == "position":
        entity.set_position(initial)
    else:
        entity.set_velocity(initial[:, None])
    for _ in range(3):
        scene.step()
    if input_kind == "position":
        entity.set_position(replacement, envs_idx=[1])
    else:
        entity.set_velocity(replacement, envs_idx=[1])
    for _ in range(3):
        scene.step()
    weights = gs.tensor([[1.0], [2.0], [3.0]])
    centers = entity.get_state().pos.mean(dim=1)
    loss = (weights * centers).sum()
    loss.backward()
    if input_kind == "position":
        expected = [[1.0] * 3, [0.0] * 3, [3.0] * 3]
        expected_replacement = 2.0
    else:
        expected = [[6 * scene.dt] * 3, [6 * scene.dt] * 3, [18 * scene.dt] * 3]
        expected_replacement = 6 * scene.dt
    assert_allclose(initial.grad, expected, atol=2e-5)
    assert_allclose(replacement.grad, expected_replacement, atol=2e-5)
