"""Native rigid and particle scene restart, including fresh-process continuation."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


def checkpoint_observations(scene):
    """Read physical and measured observables that a continuation must preserve."""
    hybrid = scene.entities[-1]
    soft = hybrid.part_soft.get_state()
    sensor = scene.entities[0].sensors[0]
    wrench = scene.get_links_contact_wrench(source="hybrid")
    diagnostics = hybrid.get_binding_diagnostics()
    return {
        "time": scene.get_time(),
        "qpos": hybrid.part_rigid.get_qpos(),
        "dofs_vel": hybrid.part_rigid.get_dofs_velocity(),
        "passive_force": hybrid.part_rigid.passive_joint_models[0].get_applied_dofs_force(),
        "pos": soft.pos,
        "vel": soft.vel,
        "C": soft.C,
        "F": soft.F,
        "Jp": soft.Jp,
        "force": wrench.force,
        "angular_impulse": wrench.angular_impulse,
        "interval": wrench.hybrid_interval,
        "attachment_force": hybrid.get_attachment_link_wrench()[0],
        "attachment_error": diagnostics.attach_pos_err_max,
        "measured": sensor.read().force,
        "ground_truth": sensor.read_ground_truth().force,
    }


@pytest.mark.required
@pytest.mark.parametrize("n_envs,grid_mode", [(0, "fixed_dense"), (2, "moving_window")])
def test_hybrid_disk_restart(n_envs, grid_mode, skin_articulation, show_viewer, tmp_path, precision):
    DT = 2e-4
    surface = trimesh.creation.box(extents=(0.0625, 0.0625, 0.0625))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2,
            gravity=(0, 0, -9.81),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.5, 0.5, 0.5),
            grid_density=32,
            particle_size=0.015625,
            grid_mode=grid_mode,
            window_size=(0.5, 0.5, 0.5) if grid_mode == "moving_window" else None,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            contact_buffer_capacity=1024,
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    hybrid = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(0, 0, 0.015625),
            file=skin_articulation,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(),
            material_soft=gs.materials.MPM.Elastic(
                E=1000,
                sampler="regular",
            ),
            material_soft_palette=(gs.materials.MPM.Elastic(E=2000, sampler="regular"),),
            skin=gs.options.HybridSkinOptions(
                surface=surface,
                vertex_material_id=(surface.vertices[:, 0] > 0).astype(gs.np_int),
            ),
            binding=gs.options.HybridBindingOptions(
                timeconst=0.02,
            ),
        ),
    )
    passive = hybrid.configure_passive_joints()
    sensor = scene.add_sensor(
        gs.sensors.SolverContactTaxel(
            entity_idx=floor.idx,
            probe_local_pos=((0, 0, 0),),
            probe_radius=0.2,
            contact_sources=("hybrid",),
            history_length=3,
            hardware_profile=gs.sensors.TactileHardwareProfile(
                filter_tau=0.001,
                hysteresis_strength=0.1,
                hysteresis_tau=0.001,
                sample_rate=2500,
                delay=DT,
                jitter=DT,
                noise=0.005,
                drift=0.002,
                gain_variation=(0.8, 1.2),
                gain_temp_coeff=0.01,
            ),
        ),
    )
    scene.build(n_envs=n_envs)
    passive.set_params(q_neutral=-0.1, k_neutral=0.5, damping=0.01)
    hybrid.part_soft.set_particles_vel((0.2, 0, -0.3))
    for _ in range(3):
        scene.step()
    assert (scene.get_contact_contributions(source="hybrid").count > 0).all()
    assert (hybrid.part_soft.get_state().F - torch.eye(3, device=gs.device)).abs().max() > 1e-6
    sensor.set_temperature(35)
    sensor.set_hysteresis_scale(0.7)
    hybrid.set_attachment_params(timeconst=0.03, damping_ratio=0.6, envs_idx=[0] if n_envs else None)
    hybrid.part_rigid.set_dofs_kp(1.0)
    hybrid.part_rigid.set_dofs_kv(0.1)
    hybrid.part_rigid.control_dofs_position(0.2)
    hybrid.part_soft.set_velocity((0.1, 0, -0.1), envs_idx=[0] if n_envs else None)
    saved = checkpoint_observations(scene)
    path = tmp_path / "hybrid.gstraj"
    scene.save_checkpoint(path)
    assert_equal(checkpoint_observations(scene)["measured"], saved["measured"])
    expected = [saved]
    for _ in range(3):
        scene.step()
        expected.append(checkpoint_observations(scene))

    loaded = gs.Scene.load_checkpoint(path, restore_rng=True)
    for index, observations in enumerate(expected):
        if index:
            loaded.step()
        for name, actual in checkpoint_observations(loaded).items():
            if index == 0:
                assert_equal(actual, observations[name])
            else:
                assert_allclose(actual, observations[name], atol=1e-5, rtol=1e-6)

    result_path = tmp_path / "resumed.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.coupling.test_checkpoint",
            str(path),
            str(result_path),
            gs.backend.name,
            precision,
        ],
        env=dict(os.environ, GS_HEADLESS="1", PYGLET_HEADLESS="1"),
        check=True,
        timeout=180,
        capture_output=True,
        text=True,
    )
    with np.load(result_path) as resumed:
        for index, observations in enumerate(expected):
            for name, value in observations.items():
                assert_allclose(resumed[f"{index}.{name}"], value, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    backend = {"cpu": gs.cpu, "cuda": gs.cuda, "metal": gs.metal}[sys.argv[3]]
    gs.init(backend=backend, precision=sys.argv[4], logging_level="warning")
    restored = gs.Scene.load_checkpoint(Path(sys.argv[1]), restore_rng=True)
    observations = {}
    for index in range(4):
        if index:
            restored.step()
        observations.update(
            (f"{index}.{name}", tensor_to_array(value)) for name, value in checkpoint_observations(restored).items()
        )
    np.savez(sys.argv[2], **observations)
