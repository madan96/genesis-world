import numpy as np
import pytest

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("joint_type", ["hinge", "slide"])
def test_passive_actuator_coexistence(independent_scalar_joints, show_viewer, joint_type):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            substeps=2,
            gravity=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=independent_scalar_joints(joint_type),
            default_armature=0.0,
        ),
    )
    model = entity.configure_passive_joints()
    spring = entity.configure_passive_joints([0])
    scene.build(n_envs=2)
    assert_allclose(entity.get_qpos(), [[0.4, -0.3]] * 2, atol=1e-7)
    entity.set_dofs_force_range(-0.25, 0.25)
    entity.set_dofs_stiffness([0.6, 0.8])
    entity.set_dofs_damping([0.1, 0.2])
    model.set_params(
        q_neutral=[0.1, -0.2],
        k_neutral=[2, 3],
        soft_lo=[-0.15, -0.3],
        soft_hi=[0.25, 0.4],
        k_soft=[4, 5],
        soft_exponent=2,
        damping=[0.4, 0.5],
        friction=[0.1, 0.2],
        force_limit=[[0.9, np.inf], [0.8, np.inf]],
    )
    spring.set_params(k_neutral=1)
    with pytest.raises(gs.GenesisException):
        model.set_params(force_limit=-1)
    position = np.array([[0.6, -0.8], [-0.4, 0.7]])
    velocity = np.array([[0.2, -0.3], [-0.1, 0.4]])
    stiffness = np.array([[2.0, 3.0], [2.0, 3.0]])
    force_limit = np.array([[0.9, np.inf], [0.8, np.inf]])
    entity.set_dofs_position(position)
    entity.set_dofs_velocity(velocity)
    for i in range(5):
        if i == 2:
            scene.reset(envs_idx=[0])
            position[0] = 0
            velocity[0] = 0
            model.set_params(k_neutral=[4, 5], envs_idx=[0])
            stiffness[0] = [4, 5]
        command = np.array([5.0, -5.0]) * (-1) ** i
        entity.control_dofs_force(command)
        passive = -stiffness * (position - [0.1, -0.2]) - [0.4, 0.5] * velocity - [0.1, 0.2] * np.tanh(velocity / 0.05)
        limits = [4, 5] * (np.maximum([-0.15, -0.3] - position, 0) ** 2 - np.maximum(position - [0.25, 0.4], 0) ** 2)
        assert_allclose(model.get_passive_dofs_force(), passive, atol=3e-6)
        assert_allclose(model.get_limit_dofs_force(), limits, atol=3e-6)
        assert_allclose(model.get_applied_dofs_force(), np.clip(passive + limits, -force_limit, force_limit), atol=3e-6)
        forces = model.get_dofs_forces()
        assert_allclose(forces.passive, passive, atol=3e-6)
        assert_allclose(forces.limit, limits, atol=3e-6)
        assert_allclose(forces.applied, np.clip(passive + limits, -force_limit, force_limit), atol=3e-6)
        assert (np.abs(passive + limits) > force_limit).any()
        for _ in range(2):
            elastic = -stiffness * (position - [0.1, -0.2]) + [4, 5] * (
                np.maximum([-0.15, -0.3] - position, 0) ** 2 - np.maximum(position - [0.25, 0.4], 0) ** 2
            )
            applied = np.clip(
                elastic - [0.4, 0.5] * velocity - [0.1, 0.2] * np.tanh(velocity / 0.05), -force_limit, force_limit
            )
            # The substep integrates implicitly the slope of the chord of the applied force from zero to the current
            # velocity, whose limit at rest is the damping plus friction / friction_vel_eps unless the force saturates.
            slope = np.where(np.abs(elastic) < force_limit, [0.4, 0.5] + np.array([0.1, 0.2]) / 0.05, 0.0)
            rest = np.clip(elastic, -force_limit, force_limit)
            np.divide(rest - applied, velocity, out=slope, where=np.abs(velocity) > gs.EPS)
            force = np.clip(command, -0.25, 0.25) + applied - [0.6, 0.8] * position - [0.1, 0.2] * velocity
            force[:, 0] -= position[:, 0]
            # Revolute inertia is 2/5 * 2 * 0.2^2 + armature 0.068 = 0.1; each slider has mass 0.1.
            velocity += 0.001 * force / (0.1 + 0.001 * ([0.1, 0.2] + slope))
            position += 0.001 * velocity
        scene.step()
        assert_allclose(entity.get_dofs_control_force(), np.tile(np.clip(command, -0.25, 0.25), (2, 1)), atol=1e-7)
        assert_allclose(entity.get_dofs_velocity(), velocity, atol=3e-6)
        assert_allclose(entity.get_dofs_position(), position, atol=3e-6)
    energy = (
        np.sum(
            0.5 * stiffness * (position - [0.1, -0.2]) ** 2
            + [4, 5] * (np.maximum([-0.15, -0.3] - position, 0) ** 3 + np.maximum(position - [0.25, 0.4], 0) ** 3) / 3,
            axis=-1,
        )
        + 0.5 * position[:, 0] ** 2
        + np.sum([0.3, 0.4] * position**2, axis=-1)
    )
    assert_allclose(entity.get_potential_energy(), energy, atol=3e-6)


@pytest.mark.required
@pytest.mark.parametrize("force_limit", [np.inf, 0.5])
def test_coupled_limit_potentials(independent_scalar_joints, show_viewer, tmp_path, force_limit):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.0001,
            gravity=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=independent_scalar_joints(),
        ),
    )
    model = entity.configure_passive_joints()
    model.add_coupled_limit_group([0, 1], name="ellipse")
    model.add_coupled_limit_group([0, 1], kind="polytope", name="faces", n_faces=3)
    scene.build(n_envs=4)
    model.set_params(force_limit=force_limit)
    center = np.array([0.1, -0.1])
    radii_pos = np.array([[0.5, 1], [0.25, 2], [2, 0.5], [1, 2]])
    radii_neg = np.array([1, 0.5])
    normals = np.array([[1, 2], [-2, 1], [0, -1]])
    offsets = np.array([0.3, 0.4, 0.2])
    model.set_coupled_limit_params("ellipse", center=center, radii_pos=radii_pos, radii_neg=radii_neg, k=3, exponent=2)
    model.set_coupled_limit_params(
        "faces", center=center, faces_normal=normals, faces_offset=offsets, k=1.5, exponent=3
    )
    position = np.array([[0.6, 0.4], [-0.7, 0.6], [0.1, -0.1], [0.2, -0.2]])
    entity.set_dofs_position(position)

    def potential(position):
        delta = position - center
        radii = np.where(delta >= 0, radii_pos, radii_neg)
        excess_ellipse = np.maximum(np.linalg.norm(delta / radii, axis=-1) - 1, 0)
        excess_faces = np.maximum(delta @ normals.T - offsets, 0)
        return excess_ellipse**3 + 1.5 * np.sum(excess_faces**4, axis=-1) / 4

    gradient = np.empty_like(position)
    for i in range(2):
        displacement = np.zeros_like(position)
        displacement[:, i] = 1e-5
        gradient[:, i] = (potential(position + displacement) - potential(position - displacement)) / 2e-5
    assert_allclose(model.get_coupled_limit_dofs_force(), -gradient, atol=2e-5)
    applied_force = np.clip(-gradient, -force_limit, force_limit)
    assert_allclose(model.get_applied_dofs_force(), applied_force, atol=2e-5)
    assert_allclose(entity.get_potential_energy(), potential(position), atol=2e-5)
    radii = np.where(position >= center, radii_pos, radii_neg)
    violations = np.stack(
        (
            np.maximum(np.linalg.norm((position - center) / radii, axis=-1) - 1, 0),
            np.maximum((position - center) @ normals.T - offsets, 0).max(axis=-1),
        ),
        axis=-1,
    )
    assert_allclose(model.get_coupled_limit_violation(), violations, atol=2e-6)
    with pytest.raises(gs.GenesisException):
        model.set_coupled_limit_params("ellipse", center=100, radii_pos=0)
    with pytest.raises(gs.GenesisException):
        model.set_params(q_neutral=100, k_neutral=-1)
    assert_allclose(entity.get_potential_energy(), potential(position), atol=2e-5)
    entity.control_dofs_force(0.3)
    checkpoint_path = tmp_path / "passive.gstraj"
    scene.save_checkpoint(checkpoint_path)
    scene.step()
    assert_allclose(entity.get_dofs_velocity(), 0.0001 * (0.3 + applied_force) / 0.1, atol=2e-6)
    restored = gs.Scene.load_checkpoint(checkpoint_path, show_viewer=show_viewer)
    restored.step()
    assert_allclose(restored.entities[0].get_dofs_velocity(), 0.0001 * (0.3 + applied_force) / 0.1, atol=2e-6)
    restored.entities[0].passive_joint_models[0].set_coupled_limit_params("ellipse", k=0, envs_idx=[0])


@pytest.mark.required
@pytest.mark.parametrize(
    "integrator", [gs.integrator.Euler, gs.integrator.implicitfast, gs.integrator.approximate_implicitfast]
)
@pytest.mark.parametrize("dt, substeps", [(2e-4, 1), (1 / 120, 2), (0.004, 2)])
def test_passive_dissipation(independent_scalar_joints, light_hinge, show_viewer, integrator, dt, substeps):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
            use_hibernation=True,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=independent_scalar_joints(),
        ),
    )
    hand = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=light_hinge,
            default_armature=0.0,
        ),
    )
    model = entity.configure_passive_joints()
    wrist = hand.configure_passive_joints()
    scene.build()
    model.set_params(damping=[2, 3], friction=[0.2, 0.5])
    hand.set_dofs_damping(1.5)
    wrist.set_params(damping=0.15, friction=0.3)
    entity.set_dofs_velocity([2, -3])
    hand.set_dofs_velocity(0.5)
    energy = entity.get_kinetic_energy()
    hand_velocity = hand.get_dofs_velocity()
    for _ in range(10):
        assert (model.get_passive_dofs_force() * entity.get_dofs_velocity() <= 0).all()
        scene.step()
        next_energy = entity.get_kinetic_energy()
        assert next_energy < energy
        energy = next_energy
        next_hand_velocity = hand.get_dofs_velocity()
        assert ((0.0 <= next_hand_velocity) & (next_hand_velocity <= hand_velocity)).all()
        hand_velocity = next_hand_velocity
    # Damping alone scales each velocity by 1 / (1 + h * damping / inertia) per substep, and friction removes more.
    h = dt / substeps
    assert energy < 0.5 * 0.1 * np.sum(np.array([2, 3]) ** 2 * (1 + h * np.array([2, 3]) / 0.1) ** (-20 * substeps))
    assert_allclose(hand_velocity, 0.0, atol=1e-6)
    scene.reset()
    for _ in range(12):
        scene.step()
    # The implicit damping and friction slope at rest enter the mass matrix under approximate_implicitfast only, and a
    # hibernated entity keeps it unchanged.
    mass_diag = np.full(2, 0.1)
    if integrator == gs.integrator.approximate_implicitfast:
        mass_diag += h * (np.array([2, 3]) + np.array([0.2, 0.5]) / 0.05)
    assert_allclose(entity.get_mass_mat(), np.diag(mass_diag), tol=gs.EPS)
    model.set_params(q_neutral=[0.3, -0.2], k_neutral=[2, 3])
    scene.step()
    # Each substep integrates the damping and the friction chord slope at the current velocity implicitly.
    position, velocity = np.zeros(2), np.zeros(2)
    for _ in range(substeps):
        ratio = np.ones(2)
        np.divide(np.tanh(velocity / 0.05), velocity / 0.05, out=ratio, where=np.abs(velocity) > gs.EPS)
        force = -np.array([2, 3]) * (position - [0.3, -0.2]) - [2, 3] * velocity - [0.2, 0.5] * np.tanh(velocity / 0.05)
        velocity += h * force / (0.1 + h * ([2, 3] + np.array([0.2, 0.5]) * ratio / 0.05))
        position += h * velocity
    assert_allclose(entity.get_dofs_velocity(), velocity, atol=2e-7)
