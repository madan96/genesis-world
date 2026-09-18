import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
from genesis.utils import geom

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_generated_skin_nearest_link_binding(n_envs, skin_collision_union, show_viewer):
    PITCH = 0.005
    RADIUS = 0.1
    position = (0.15, -0.1, 0.1)
    quaternion = geom.xyz_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            particle_size=0.02,
            grid_density=32,
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=position,
            quat=quaternion,
            file=skin_collision_union,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(),
            material_soft=gs.materials.MPM.Elastic(
                E=1000.0,
            ),
            skin=gs.options.HybridSkinOptions(
                from_urdf_geoms=True,
                urdf_inflate=0.0,
                voxel_pitch=PITCH,
            ),
            binding=gs.options.HybridBindingOptions(
                nearest_link_blend_radius=RADIUS,
            ),
        ),
    )
    scene.build(n_envs=n_envs)
    initial = entity.part_soft.get_particles_pos().clone()
    points = initial[0] if n_envs else initial
    offset = points - points.new_tensor(position)
    local = torch.stack((offset[:, 1], -offset[:, 0], offset[:, 2]), dim=-1)
    assert (local >= local.new_tensor((-0.06, -0.04, -0.04)) - PITCH).all()
    assert (local <= local.new_tensor((0.24, 0.04, 0.04)) + PITCH).all()
    assert (local[:, 0] < -0.03).any()
    assert (local[:, 0] > 0.22).any()
    assert ((local[:, 0] <= 0.06 + PITCH) | (local[:, 0] >= 0.16 - PITCH)).all()

    centers = local.new_tensor(((-0.02, 0, 0), (0.02, 0, 0), (0.2, 0, 0)))
    box_offset = (local[:, None] - centers).abs() - 0.04
    signed_distance = torch.linalg.vector_norm(box_offset.clamp_min(0), dim=-1) + box_offset.amax(dim=-1).clamp_max(0)
    distances = torch.stack((signed_distance[:, :2].abs().amin(dim=-1), signed_distance[:, 2].abs()), dim=-1)
    expected_weights = torch.exp(-distances.square() / RADIUS**2)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    diagnostics = entity.get_binding_diagnostics()
    for index, link in enumerate(entity.part_rigid.links):
        weight = torch.where(diagnostics.link_idx == link.idx, diagnostics.weights, 0.0).sum(dim=-1)
        assert_allclose(weight, expected_weights[:, index], atol=1e-6)
    assert_equal(diagnostics.active_count, 2)
    assert_equal(diagnostics.region_id, 0)
    assert_equal(diagnostics.material_id, 0)
    assert_equal(diagnostics.surface_contact, True)
    assert diagnostics.region_names == ("skin",)
    scene.step()
    assert_allclose(entity.part_soft.get_particles_pos(), initial, atol=1e-6)
    assert_allclose(entity.get_binding_diagnostics().attach_pos_err_max, 0.0, atol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_surface_reference_transforms_and_owned_diagnostics(n_envs, skin_articulation, show_viewer, tol):
    TIMECONST = 0.02
    surface = trimesh.creation.box(extents=(0.08, 0.12, 0.16))
    surface.apply_translation((0.01, 0.02, -0.01))
    original_vertices = surface.vertices.copy()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        mpm_options=gs.options.MPMOptions(
            particle_size=0.02,
            grid_density=32,
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(0.1, 0.05, 0.15),
            quat=geom.xyz_to_quat(np.array([0.0, np.pi / 2, 0.0])),
            file=skin_articulation,
            scale=1.5,
            default_armature=0.0,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(),
            material_soft=gs.materials.MPM.Elastic(
                E=1000.0,
            ),
            skin=gs.options.HybridSkinOptions(
                surface=surface,
                skinning={
                    "link_names": ["base", "tip"],
                    "indices": np.tile([0, 1], (len(surface.vertices), 1)),
                    "weights": np.tile([0.5, 0.5], (len(surface.vertices), 1)),
                },
                reference_pos=(0.02, -0.01, 0.03),
                reference_quat=geom.xyz_to_quat(np.array([0.0, 0.0, np.pi / 2])),
                link_reference_poses={"base": (0.11, 0.03, 0.16, 1.0, 0.0, 0.0, 0.0)},
            ),
            binding=gs.options.HybridBindingOptions(
                timeconst=TIMECONST,
                damping_ratio=0.0,
            ),
        ),
    )
    scene.build(n_envs=n_envs)
    initial = entity.part_soft.get_particles_pos().clone()
    points = initial[0] if n_envs else initial
    # R_y(pi/2) R_z(pi/2) maps mesh axes to world +y, +z, +x.
    expected_center = points.new_tensor((0.115, 0.055, 0.16))
    expected_half_size = points.new_tensor((0.12, 0.06, 0.09))
    assert ((points - expected_center).abs() <= expected_half_size + 1e-6).all()
    assert_allclose(points.amin(dim=0), expected_center - expected_half_size, atol=0.021)
    assert_allclose(points.amax(dim=0), expected_center + expected_half_size, atol=0.021)
    assert_equal(surface.vertices, original_vertices)
    diagnostics = entity.get_binding_diagnostics()
    mass = entity.get_mass_breakdown()
    base_local = points - points.new_tensor((0.11, 0.03, 0.16))
    base_anchor = torch.stack((base_local[:, 2], base_local[:, 1], -base_local[:, 0]), dim=-1)
    base_anchor += points.new_tensor((0.1, 0.05, 0.15))
    force = diagnostics.particle_mass[:, None] * 0.5 * (base_anchor - points) / TIMECONST**2
    scene.step()
    link_force, link_torque = entity.get_attachment_link_wrench()
    assert_allclose(link_force, -0.5 * force.sum(dim=0), tol=tol)
    assert_allclose(link_torque[:, 0], -0.5 * torch.linalg.cross(base_anchor, force).sum(dim=0), tol=tol)
    assert_allclose(link_torque[:, 1], -0.5 * torch.linalg.cross(points, force).sum(dim=0), tol=tol)
    current = entity.get_binding_diagnostics()
    assert_allclose(current.attach_impulse_sum, force.sum(dim=0) * scene.dt, atol=2e-7)
    potential = 0.125 * (diagnostics.particle_mass * (base_anchor - points).square().sum(dim=-1)).sum() / TIMECONST**2
    assert_allclose(current.spring_potential, potential, tol=tol)
    assert_allclose(diagnostics.attach_impulse_sum, 0.0, atol=0.0)
    assert_allclose(diagnostics.spring_potential, 0.0, atol=0.0)
    expected_binding = tuple(
        value.clone()
        for value in (
            current.link_idx,
            current.weights,
            current.active_count,
            current.region_id,
            current.material_id,
            current.surface_contact,
            current.particle_mass,
        )
    )
    current.link_idx.fill_(-1)
    current.weights.zero_()
    current.active_count.zero_()
    current.region_id.fill_(1)
    current.material_id.fill_(1)
    current.surface_contact.fill_(False)
    current.particle_mass.zero_()
    preserved = entity.get_binding_diagnostics()
    for actual, expected in zip(
        (
            preserved.link_idx,
            preserved.weights,
            preserved.active_count,
            preserved.region_id,
            preserved.material_id,
            preserved.surface_contact,
            preserved.particle_mass,
        ),
        expected_binding,
    ):
        assert_equal(actual, expected)
    impulse = current.attach_impulse_sum.clone()
    current.attach_impulse_sum.zero_()
    assert_equal(entity.get_binding_diagnostics().attach_impulse_sum, impulse)
    masses = mass.per_rigid_link.clone()
    mass.per_rigid_link.zero_()
    assert_equal(entity.get_mass_breakdown().per_rigid_link, masses)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_semantic_regions_and_external_collision_roles(n_envs, skin_articulation, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            particle_size=0.02,
            grid_density=32,
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.5, 1.25, 0.5),
        ),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0.0,
        ),
    )
    boxes = []
    for x in (0.0, 0.2):
        box = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
        box.apply_translation((x, 0.0, -0.015))
        boxes.append(box)
    surface = trimesh.util.concatenate(boxes)
    region_ids = (surface.vertices[:, 0] > 0.1).astype(gs.np_int)
    entities = []
    for row, (role, mask_distal) in enumerate((("soft", False), ("rigid", False), ("both", False), ("both", True))):
        opposite_role = "rigid" if role == "soft" else "soft"
        entity = scene.add_entity(
            morph=gs.morphs.URDF(
                pos=(0.0, 0.3 * row, 0.038),
                file=skin_articulation,
            ),
            material=gs.materials.Hybrid(
                material_rigid=gs.materials.Rigid(),
                material_soft=gs.materials.MPM.Elastic(
                    E=1000.0,
                ),
                skin=gs.options.HybridSkinOptions(
                    surface=surface,
                    skinning={
                        "link_names": ["base"],
                        "indices": np.zeros((len(surface.vertices), 1), dtype=gs.np_int),
                        "weights": np.ones((len(surface.vertices), 1)),
                    },
                    regions={"region_names": ["proximal", "distal"], "vertex_region": region_ids},
                    vertex_surface_contact=region_ids == 0 if mask_distal else None,
                ),
                binding=gs.options.HybridBindingOptions(
                    damping_ratio=0.0,
                    phase="before_coupling",
                ),
                collision=gs.options.HybridCollisionOptions(
                    external_surface=role,
                    link_overrides={"tip": opposite_role} if role != "both" else None,
                    region_overrides={"distal": opposite_role} if role != "both" else None,
                ),
            ),
        )
        entities.append((entity, role, mask_distal))
    scene.build(n_envs=n_envs)
    for entity, role, mask_distal in entities:
        entity.part_soft.set_particles_vel((0.0, 0.0, -1.0))
        diagnostics = entity.get_binding_diagnostics()
        points = entity.part_soft.get_particles_pos()
        points = points[0] if n_envs else points
        assert_equal(diagnostics.region_id, points[:, 0] > 0.1)
        assert_equal(diagnostics.surface_contact, points[:, 0] <= 0.1 if mask_distal else True)
        assert diagnostics.region_names == ("proximal", "distal")
    scene.step()
    soft_rows = scene.get_contact_contributions(source="hybrid")
    rigid_rows = scene.get_contact_contributions(source="rigid")
    assert soft_rows.is_complete.all()
    assert_equal(soft_rows.entity_a[soft_rows.valid], floor.idx)
    assert ((rigid_rows.entity_a == floor.idx) | (rigid_rows.entity_b == floor.idx) | ~rigid_rows.valid).all()
    for entity, role, mask_distal in entities:
        for region, link in enumerate(entity.part_rigid.links):
            soft_enabled = role == "both" or (role == "soft") == (region == 0)
            rigid_enabled = role == "both" or not soft_enabled
            soft_enabled &= not (mask_distal and region == 1)
            contact = soft_rows.select(entity_idx=entity.part_soft.idx, region_idx=region)
            rigid_contact = rigid_rows.select(link_idx=link.idx)
            if soft_enabled:
                assert (contact.count > 0).all()
                assert (contact.impulse.sum(dim=1)[:, 2] < -1e-7).all()
            else:
                assert_equal(contact.count, 0)
            if rigid_enabled:
                assert (rigid_contact.count > 0).all()
                assert (rigid_contact.normal_force.sum(dim=1) > 0.0).all()
            else:
                assert_equal(rigid_contact.count, 0)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_non_neighbor_skin_contact(n_envs, skin_chain, show_viewer):
    surface = trimesh.creation.box(extents=(0.0625, 0.0625, 0.0625))
    surface.apply_translation((0.4, 0.0, 0.03125))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            particle_size=0.015625,
            grid_density=32,
            lower_bound=(-0.25, -0.25, -0.25),
            upper_bound=(0.75, 0.375, 0.375),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=skin_chain,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(
                coup_friction=0.0,
            ),
            material_soft=gs.materials.MPM.Elastic(
                E=1000.0,
            ),
            skin=gs.options.HybridSkinOptions(
                surface=surface,
                skinning={
                    "link_names": ["base"],
                    "indices": np.zeros((len(surface.vertices), 1), dtype=gs.np_int),
                    "weights": np.ones((len(surface.vertices), 1)),
                },
            ),
            binding=gs.options.HybridBindingOptions(
                damping_ratio=0.0,
                phase="before_coupling",
            ),
            collision=gs.options.HybridCollisionOptions(
                external_surface="both",
            ),
        ),
    )
    scene.build(n_envs=n_envs)
    entity.part_soft.set_particles_vel((0.0, 0.0, -1.0))
    mass = entity.part_soft.get_mass()
    velocity_before = entity.part_soft.get_particles_vel().clone()
    scene.step()
    velocity_after = entity.part_soft.get_particles_vel()
    expected_impulse = mass[..., None] * (velocity_before - velocity_after).mean(dim=-2)
    contact = scene.get_contact_contributions(source="hybrid")
    assert contact.is_complete.all()
    assert (contact.count > 0).all()
    assert_equal(contact.entity_a[contact.valid], entity.part_rigid.idx)
    assert_equal(contact.entity_b[contact.valid], entity.part_soft.idx)
    assert_equal(contact.link_a[contact.valid], entity.part_rigid.links[2].idx)
    assert (expected_impulse[..., 2] < -1e-5).all()
    assert_allclose(contact.impulse.sum(dim=1), expected_impulse, atol=1e-7)
