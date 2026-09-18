from dataclasses import dataclass

import numpy as np
import torch
import trimesh

import quadrants as qd

import genesis as gs
from genesis.engine.hybrid_attachment import kernel_apply_attachment, kernel_attachment_stability
from genesis.engine.materials.base import Material
from genesis.engine.states.contacts import HybridAttachmentState, restore_contact_field
from genesis.options.hybrid import HybridBindingOptions, HybridCollisionOptions, HybridMassOptions, HybridSkinOptions
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils import geom, hybrid_skin
from genesis.utils.misc import qd_to_torch


@dataclass
class HybridBindingDescription:
    """Resolved attachment geometry and mass allocation, referencing preceding scene entities by index."""

    rigid_entity_idx: int
    soft_entity_idx: int
    options: HybridBindingOptions
    collision_options: HybridCollisionOptions
    surface_world: trimesh.Trimesh
    links_reference_pose: dict[int, tuple[np.ndarray, np.ndarray]]
    links_idx: np.ndarray
    weights: np.ndarray
    active_count: np.ndarray
    rest_local_pos: np.ndarray
    region_names: list[str]
    vertex_regions: np.ndarray | None
    particles_region: np.ndarray
    soft_materials: list[Material]
    vertex_material_id: np.ndarray
    vertex_surface_contact: np.ndarray
    particle_material_id: np.ndarray
    particle_surface_contact: np.ndarray
    particle_mass: np.ndarray
    soft_mass_by_link: np.ndarray


@dataclass(frozen=True)
class HybridMassBreakdown:
    """Owned device tensors with a leading environment axis. Mass is in kg and center of mass is world xyz in m."""

    total: torch.Tensor
    rigid_total: torch.Tensor
    soft_total: torch.Tensor
    per_rigid_link: torch.Tensor
    soft_mass_by_weighted_link: torch.Tensor
    combined_center_of_mass: torch.Tensor
    link_names: tuple[str, ...]


@dataclass(frozen=True)
class HybridBindingDiagnostics:
    """Owned binding data and attachment diagnostics on the simulation device.

    Link indices and weights have shape ``[P, K]`` in particle and influence order. Other binding tensors have shape
    ``[P]``. Global link indices use -1 for unused slots. Region IDs index ``region_names`` and material IDs index the
    hybrid material palette. Particle masses are in kg. Attachment impulses have shape ``[B, 3]`` in world axes
    (N s and N m s about the world origin), accumulated over the last scene step. Remaining diagnostics have shape
    ``[B]``. Damping work is the accumulated dissipative work in joules and spring potential is the last substep's
    uncapped spring energy in joules. Position error is the step maximum in meters. The dimensionless stability index
    describes the last checked substep and must be at most one.
    """

    link_idx: torch.Tensor
    weights: torch.Tensor
    active_count: torch.Tensor
    region_id: torch.Tensor
    region_names: tuple[str, ...]
    material_id: torch.Tensor
    surface_contact: torch.Tensor
    particle_mass: torch.Tensor
    attach_pos_err_max: torch.Tensor
    attach_impulse_sum: torch.Tensor
    attach_ang_impulse_sum: torch.Tensor
    damping_work: torch.Tensor
    spring_potential: torch.Tensor
    stability_index: torch.Tensor


class HybridBinding:
    """Reference geometry, particle attachment and mass allocation for one continuous skin."""

    def __init__(self, entity, scene, material, morph, surface, desc: HybridBindingDescription | None = None):
        if desc is not None:
            self.entity = entity
            self.options = desc.options
            self.is_force_limited = self.options.max_correction_force is not None
            self.collision_options = desc.collision_options
            self.part_rigid = scene.entities[desc.rigid_entity_idx]
            self.part_soft = scene.entities[desc.soft_entity_idx]
            self.surface_world = desc.surface_world
            self.links_reference_pose = desc.links_reference_pose
            self.links_idx = desc.links_idx
            self.weights = desc.weights
            self.active_count = desc.active_count
            self.rest_local_pos = desc.rest_local_pos
            self.region_names = desc.region_names
            self.vertex_regions = desc.vertex_regions
            self.particles_region = desc.particles_region
            self.n_particles = self.part_soft.n_particles
            self.soft_materials = [self.part_soft.material]
            for soft_material in desc.soft_materials[1:]:
                self.part_soft.solver.add_material(soft_material)
                self.soft_materials.append(soft_material)
            self.vertex_material_id = desc.vertex_material_id
            self.vertex_surface_contact = desc.vertex_surface_contact
            self.particle_material_id = desc.particle_material_id
            self.particle_surface_contact = desc.particle_surface_contact
            self.particle_mass = desc.particle_mass
            self.soft_mass_by_link = desc.soft_mass_by_link
            self.part_soft.solver.register_hybrid_binding(self)
            self.desc = desc
            return
        if not isinstance(morph, gs.morphs.URDF):
            gs.raise_exception("Continuous Hybrid skin requires one URDF articulation.")
        if not isinstance(material.material_soft, gs.materials.MPM.Base):
            gs.raise_exception("Continuous Hybrid skin requires an MPM soft material.")
        if not isinstance(scene.options.coupler, LegacyCouplerOptions):
            gs.raise_exception("Continuous Hybrid attachment requires LegacyCouplerOptions.")
        if scene.requires_grad:
            gs.raise_exception(
                "Continuous Hybrid attachment supports forward simulation. Differentiation is unavailable."
            )
        if scene.sim.mpm_solver.enable_CPIC or not scene.options.coupler.rigid_mpm:
            gs.raise_exception("Continuous Hybrid attachment requires rigid-MPM coupling with CPIC disabled.")
        if material.use_default_coupling:
            gs.raise_exception("Typed Hybrid attachment defines its coupling. use_default_coupling must be False.")
        if any(
            callback is not None
            for callback in (
                material.func_instantiate_rigid_from_soft,
                material.func_instantiate_soft_from_rigid,
                material.func_instantiate_rigid_soft_association,
            )
        ):
            gs.raise_exception("Typed Hybrid skin options and custom instantiation callbacks are mutually exclusive.")

        self.entity = entity
        self.options = material.binding or HybridBindingOptions()
        self.is_force_limited = self.options.max_correction_force is not None
        self.skin_options = material.skin or HybridSkinOptions(from_urdf_geoms=True)
        self.collision_options = material.collision or HybridCollisionOptions()
        material_rigid = material.material_rigid.model_copy(deep=True)
        material_rigid.needs_coup = True
        material_soft = type(material.material_soft)(**material.material_soft.model_dump())
        material_soft.sampler = self.skin_options.sampler
        self.part_rigid = scene.add_entity(morph=morph, material=material_rigid, surface=gs.surfaces.Smooth())
        self.links_reference_pose = hybrid_skin.reference_link_poses(self.part_rigid.links)
        particle_size = scene.sim.mpm_solver.particle_size
        if self.skin_options.from_urdf_geoms:
            surface_world = hybrid_skin.build_urdf_union_surface(
                self.part_rigid,
                self.links_reference_pose,
                pitch=self.skin_options.voxel_pitch or particle_size,
                inflate=self.skin_options.urdf_inflate,
            )
        else:
            surface_world = hybrid_skin.load_surface(self.skin_options.surface, scale=morph.scale)
            hybrid_skin.validate_solid_surface(surface_world)
            pos, quat = geom.transform_pos_quat_by_trans_quat(
                np.asarray(self.skin_options.reference_pos),
                np.asarray(self.skin_options.reference_quat),
                np.asarray(morph.pos),
                np.asarray(morph.quat),
            )
            surface_world.apply_transform(geom.trans_quat_to_T(pos, quat))
        self.surface_world = surface_world
        self.part_soft = scene.add_entity(
            morph=gs.morphs.MeshSet(files=[surface_world], poss=[(0.0, 0.0, 0.0)], eulers=[(0.0, 0.0, 0.0)]),
            material=material_soft,
            surface=surface,
        )
        particles = np.asarray(self.part_soft.init_particles, dtype=np.float64)
        if not len(particles):
            gs.raise_exception(
                "Hybrid skin sampling produced no particles. Reduce particle_size or increase the volume."
            )
        boundary = scene.sim.mpm_solver.boundary
        if np.any(particles < boundary.lower) or np.any(particles > boundary.upper):
            gs.raise_exception("Hybrid skin particles extend outside the padded MPM domain.")
        link_names_to_idx = {link.name: link.idx for link in self.part_rigid.links}
        if self.skin_options.skinning is None:
            links, weights, count = hybrid_skin.nearest_link_fallback_weights(
                particles,
                self.part_rigid.links,
                self.links_reference_pose,
                self.options.nearest_link_blend_radius,
                self.options.max_influences,
            )
        else:
            vertex_links, vertex_weights = hybrid_skin.load_skinning(
                self.skin_options.skinning, surface_world, link_names_to_idx
            )
            links, weights, count = hybrid_skin.transfer_weights_to_particles(
                particles, surface_world, vertex_links, vertex_weights, self.options.max_influences
            )
        self.links_reference_pose = hybrid_skin.reference_link_poses(
            self.part_rigid.links, self.skin_options.link_reference_poses
        )
        self.links_idx = links
        self.weights = weights
        self.active_count = count
        self.rest_local_pos = hybrid_skin.compute_rest_local_positions(particles, links, self.links_reference_pose)
        self.region_names, vertex_region = hybrid_skin.load_regions(self.skin_options.regions, surface_world)
        self.vertex_regions = vertex_region
        if vertex_region is None:
            self.region_names = ["skin"]
            self.particles_region = np.zeros(len(particles), dtype=gs.np_int)
        else:
            self.particles_region = hybrid_skin.transfer_regions_to_particles(particles, surface_world, vertex_region)
        self.n_particles = len(particles)
        self.soft_materials = [self.part_soft.material]
        for soft_material in material.material_soft_palette:
            if not isinstance(soft_material, gs.materials.MPM.Base):
                gs.raise_exception("Every Hybrid material_soft_palette entry must be an MPM material.")
            owned_material = type(soft_material)(**soft_material.model_dump())
            self.part_soft.solver.add_material(owned_material)
            self.soft_materials.append(owned_material)
        self.vertex_material_id, self.vertex_surface_contact = hybrid_skin.load_vertex_attributes(
            surface_world,
            self.skin_options.vertex_material_id,
            self.skin_options.vertex_surface_contact,
            len(self.soft_materials),
        )
        self.particle_material_id = hybrid_skin.transfer_regions_to_particles(
            particles, surface_world, self.vertex_material_id
        )
        self.particle_surface_contact = hybrid_skin.transfer_regions_to_particles(
            particles, surface_world, self.vertex_surface_contact
        ).astype(bool)
        self.mass_options = material.mass or HybridMassOptions()
        self._apply_mass_accounting(particles)
        self._configure_collision_roles()
        self.part_soft.solver.register_hybrid_binding(self)
        self.desc = HybridBindingDescription(
            self.part_rigid.idx,
            self.part_soft.idx,
            self.options,
            self.collision_options,
            self.surface_world,
            self.links_reference_pose,
            self.links_idx,
            self.weights,
            self.active_count,
            self.rest_local_pos,
            self.region_names,
            self.vertex_regions,
            self.particles_region,
            self.soft_materials,
            self.vertex_material_id,
            self.vertex_surface_contact,
            self.particle_material_id,
            self.particle_surface_contact,
            self.particle_mass,
            self.soft_mass_by_link,
        )

    def _apply_mass_accounting(self, particles):
        links = self.part_rigid.links
        masses = np.array([link.desc.mass or 0.0 for link in links], dtype=np.float64)
        densities = np.array([material.rho for material in self.soft_materials])
        particle_mass = self.part_soft.solver.particle_volume_real * densities[self.particle_material_id]
        shares = np.zeros(len(links))
        for i_k in range(self.links_idx.shape[1]):
            is_active = self.links_idx[:, i_k] >= 0
            np.add.at(
                shares,
                self.links_idx[is_active, i_k] - self.part_rigid.link_start,
                self.weights[is_active, i_k] * particle_mass[is_active],
            )
        if self.mass_options.mode == "split_total":
            if masses.sum() <= 0.0:
                gs.raise_exception("Hybrid split_total requires positive authored rigid mass.")
            soft_target = self.mass_options.total_mass * self.mass_options.soft_mass_fraction
            density_ratio = float(soft_target / particle_mass.sum())
            for soft_material in self.soft_materials:
                soft_material.rho *= density_ratio
            particle_mass *= density_ratio
            shares *= density_ratio
            # Every link keeps its authored fraction of total_mass, the skin bound to it included. A single ratio over
            # all links would move the skin mass of heavily covered links onto uncovered ones.
            links_mass = masses * (self.mass_options.total_mass / masses.sum())
            for i_l, link in enumerate(links):
                if shares[i_l] > 0.0 and shares[i_l] >= links_mass[i_l]:
                    gs.raise_exception(
                        f"Hybrid split_total binds {shares[i_l]:.4g} kg of skin to link '{link.name}', at least its "
                        f"{links_mass[i_l]:.4g} kg share of total_mass. Lower soft_mass_fraction or bind less skin to "
                        "this link."
                    )
                if masses[i_l] > 0.0:
                    # Uniform density scaling keeps the link center of mass and principal axes.
                    ratio = (links_mass[i_l] - shares[i_l]) / masses[i_l]
                    link.desc.mass *= ratio
                    link.desc.inertia = np.asarray(link.desc.inertia) * ratio
                    link.desc.invweight[:] = -1.0
        elif self.mass_options.mode == "rigid_total":
            replacements = []
            for i_l, link in enumerate(links):
                if shares[i_l] <= 0.0:
                    continue
                assigned_mass = particle_mass * np.where(self.links_idx == link.idx, self.weights, 0.0).sum(axis=1)
                positions = geom.inv_transform_by_trans_quat(particles, *self.links_reference_pose[link.idx])
                rotation = geom.quat_to_R(link.desc.inertial_quat)
                inertia = rotation @ link.desc.inertia @ rotation.T
                try:
                    mass, com, inertia = hybrid_skin.subtract_particle_inertial(
                        masses[i_l], link.desc.inertial_pos, inertia, positions, assigned_mass
                    )
                except ValueError as error:
                    gs.raise_exception(f"Hybrid rigid_total is invalid for link '{link.name}': {error}")
                replacements.append((link, mass, com, inertia))
            for link, mass, com, inertia in replacements:
                link.desc.mass = mass
                link.desc.inertial_pos = com
                link.desc.inertial_quat = geom.identity_quat()
                link.desc.inertia = inertia
                link.desc.invweight[:] = -1.0
        self.particle_mass = particle_mass
        self.soft_mass_by_link = shares

    def _configure_collision_roles(self):
        overrides = self.collision_options.link_overrides or {}
        names = {link.name for link in self.part_rigid.links}
        if set(overrides) - names:
            gs.raise_exception(f"Unknown Hybrid collision links: {sorted(set(overrides) - names)}.")
        for link in self.part_rigid.links:
            role = overrides.get(link.name, self.collision_options.external_surface)
            if role == "both":
                gs.logger.warning(f"Hybrid link '{link.name}' enables both rigid and soft external contact surfaces.")
            for link_geom in link.geoms:
                link_geom._needs_coup = role in ("rigid", "both")
                if role == "soft":
                    link_geom.desc.contype = 0
                    link_geom.desc.conaffinity = 0

    def build(self):
        solver = self.part_soft.solver
        n_envs = solver._B
        material_idx = np.array([material.idx for material in self.soft_materials], dtype=gs.np_int)
        default_Jp = np.array([material._default_Jp for material in self.soft_materials], dtype=gs.np_float)
        kernel_set_particle_materials(
            self.part_soft.particle_start,
            material_idx[self.particle_material_id],
            default_Jp[self.particle_material_id],
            self.particle_mass.astype(gs.np_float),
            solver,
        )
        self.bind_links_idx = qd.field(gs.qd_int, shape=self.links_idx.shape)
        self.bind_weights = qd.field(gs.qd_float, shape=self.weights.shape)
        self.bind_rest_local_pos = qd.field(gs.qd_vec3, shape=self.weights.shape)
        self.bind_active_count = qd.field(gs.qd_int, shape=(self.n_particles,))
        self.bind_links_idx.from_numpy(self.links_idx.astype(gs.np_int))
        self.bind_weights.from_numpy(self.weights.astype(gs.np_float))
        self.bind_rest_local_pos.from_numpy(self.rest_local_pos.astype(gs.np_float))
        self.bind_active_count.from_numpy(self.active_count.astype(gs.np_int))
        self.omega = qd.field(gs.qd_float, shape=n_envs)
        self.damping_ratio = qd.field(gs.qd_float, shape=n_envs)
        self.omega.fill(1.0 / self.options.timeconst)
        self.damping_ratio.fill(self.options.damping_ratio)
        self.link_impulse = qd.field(gs.qd_vec3, shape=(self.part_rigid.n_links, n_envs))
        self.link_angular_impulse = qd.field(gs.qd_vec3, shape=(self.part_rigid.n_links, n_envs))
        self.position_error_max = qd.field(gs.qd_float, shape=n_envs)
        self.particle_impulse = qd.field(gs.qd_vec3, shape=n_envs)
        self.particle_angular_impulse = qd.field(gs.qd_vec3, shape=n_envs)
        self.damping_work = qd.field(gs.qd_float, shape=n_envs)
        self.spring_potential = qd.field(gs.qd_float, shape=n_envs)
        self.mobility_bound = qd.field(gs.qd_float, shape=n_envs)
        self.stability_index = qd.field(gs.qd_float, shape=n_envs)
        # The stability bound takes the soft mass of every particle, inactive ones included, which keeps it conservative.
        self.link_soft_mass = qd.field(gs.qd_float, shape=(self.part_rigid.n_links,))
        self.link_soft_mass.from_numpy(self.soft_mass_by_link.astype(gs.np_float))
        self.link_weighted_lever = qd.field(gs.qd_float, shape=(self.part_rigid.n_links, n_envs))
        # Seconds of attachment accumulated in the link impulses, per environment since a partial reset or snapshot
        # restore leaves the other environments' impulses in place.
        self.interval = qd.field(gs.qd_float, shape=n_envs)
        self.links_idx_tensor = torch.tensor(self.links_idx, dtype=gs.tc_int, device=gs.device)
        self.weights_tensor = torch.tensor(self.weights, dtype=gs.tc_float, device=gs.device)
        self.active_count_tensor = torch.tensor(self.active_count, dtype=gs.tc_int, device=gs.device)
        self.regions_tensor = torch.tensor(self.particles_region, dtype=gs.tc_int, device=gs.device)
        self.particle_mass_tensor = torch.tensor(self.particle_mass, dtype=gs.tc_float, device=gs.device)
        self.particle_material_id_tensor = torch.tensor(self.particle_material_id, dtype=gs.tc_int, device=gs.device)
        self.particle_surface_contact_tensor = torch.tensor(
            self.particle_surface_contact, dtype=gs.tc_bool, device=gs.device
        )
        self.reset(self.entity.scene._envs_idx)

    @property
    def observation_fields(self) -> tuple[qd.Field, ...]:
        return (
            self.link_impulse,
            self.link_angular_impulse,
            self.position_error_max,
            self.particle_impulse,
            self.particle_angular_impulse,
            self.damping_work,
            self.spring_potential,
            self.stability_index,
            self.interval,
        )

    def get_observation_state(self) -> HybridAttachmentState:
        """Capture owned attachment diagnostics and the interval defining their wrench."""
        return HybridAttachmentState(
            *(qd_to_torch(field, transpose=True, copy=True) for field in self.observation_fields)
        )

    def set_observation_state(self, state: HybridAttachmentState, envs_idx=None) -> None:
        """Restore selected attachment diagnostics, retaining other environments' observations."""
        for field, value in zip(self.observation_fields, state, strict=True):
            restore_contact_field(field, value, envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def reset(self, envs_idx):
        kernel_reset_attachment(envs_idx, self)

    def apply_attachment(self, f, phase):
        if phase != self.options.phase:
            return
        rigid = self.part_rigid.solver
        # The host-side stability test forces a device sync, so it shares the simulator's error-check cadence.
        if self.part_soft.solver.sim._cur_substep_global % gs.engine.simulator.RATE_CHECK_ERRNO == 0:
            kernel_attachment_stability(f, rigid.dyn_state.links, rigid.dyn_info.links, rigid.rigid_config, self)
            stability = qd_to_torch(self.stability_index)
            if torch.any(~torch.isfinite(stability) | (stability > 1.0)):
                gs.raise_exception(
                    "Hybrid attachment exceeds its supported local stability range for the current mass, inertia and "
                    "geometry. Increase binding.timeconst, lower binding.damping_ratio or reduce the simulation substep "
                    "duration."
                )
        kernel_apply_attachment(f, rigid.dyn_state.links, self)

    def get_mass_breakdown(self) -> HybridMassBreakdown:
        rigid = self.part_rigid
        masses = torch.broadcast_to(rigid.get_links_mass(), (rigid.scene._B, rigid.n_links))
        coms = rigid.get_links_pos(ref=gs.link_ref_frame.link_COM, relative=False)
        if rigid.scene.n_envs == 0:
            coms = coms[None]
        particles = self.part_soft.get_particles_pos()
        if rigid.scene.n_envs == 0:
            particles = particles[None]
        rigid_total = masses.sum(dim=-1)
        soft_total = self.particle_mass_tensor.sum().expand_as(rigid_total)
        total = rigid_total + soft_total
        center = (
            (coms * masses[..., None]).sum(dim=1) + (particles * self.particle_mass_tensor[None, :, None]).sum(dim=1)
        ) / total[:, None]
        return HybridMassBreakdown(
            total,
            rigid_total,
            soft_total,
            masses.clone(),
            qd_to_torch(self.link_soft_mass).expand(masses.shape).clone(),
            center,
            tuple(link.name for link in rigid.links),
        )

    def get_binding_diagnostics(self) -> HybridBindingDiagnostics:
        return HybridBindingDiagnostics(
            link_idx=self.links_idx_tensor.clone(),
            weights=self.weights_tensor.clone(),
            active_count=self.active_count_tensor.clone(),
            region_id=self.regions_tensor.clone(),
            region_names=tuple(self.region_names),
            material_id=self.particle_material_id_tensor.clone(),
            surface_contact=self.particle_surface_contact_tensor.clone(),
            particle_mass=self.particle_mass_tensor.clone(),
            attach_pos_err_max=qd_to_torch(self.position_error_max, copy=True),
            attach_impulse_sum=qd_to_torch(self.particle_impulse, copy=True),
            attach_ang_impulse_sum=qd_to_torch(self.particle_angular_impulse, copy=True),
            damping_work=qd_to_torch(self.damping_work, copy=True),
            spring_potential=qd_to_torch(self.spring_potential, copy=True),
            stability_index=qd_to_torch(self.stability_index, copy=True),
        )

    def get_attachment_link_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        interval = qd_to_torch(self.interval).clamp_min(torch.finfo(gs.tc_float).tiny)[:, None, None]
        return (
            qd_to_torch(self.link_impulse, transpose=True, copy=True) / interval,
            qd_to_torch(self.link_angular_impulse, transpose=True, copy=True) / interval,
        )

    def set_attachment_params(self, timeconst=None, damping_ratio=None, envs_idx=None):
        envs_idx = self.entity.scene._sanitize_envs_idx(envs_idx)
        values = []
        for value, field, is_positive in ((timeconst, self.omega, True), (damping_ratio, self.damping_ratio, False)):
            if value is None:
                continue
            value = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)
            if not torch.isfinite(value).all() or (value <= 0.0 if is_positive else value < 0.0).any():
                gs.raise_exception("Attachment timeconst must be finite and positive, and damping_ratio nonnegative.")
            values.append((field, torch.broadcast_to(1.0 / value if is_positive else value, (len(envs_idx),))))
        for field, value in values:
            kernel_set_attachment_parameter(envs_idx, value.contiguous(), field)


@qd.kernel
def kernel_set_particle_materials(
    particle_start: qd.i32,
    material_idx: qd.types.ndarray(),
    default_Jp: qd.types.ndarray(),
    masses: qd.types.ndarray(),
    solver: qd.template(),
):
    for i_p_ in range(masses.shape[0]):
        i_p = particle_start + i_p_
        solver.particles_info.material_idx[i_p] = material_idx[i_p_]
        solver.particles_info.default_Jp[i_p] = default_Jp[i_p_]
        solver.particles_info.mass[i_p] = masses[i_p_] * solver.particle_volume_scale
        for i_b in range(solver._B):
            solver.particles.Jp[0, i_p, i_b] = default_Jp[i_p_]


@qd.kernel
def kernel_set_attachment_parameter(envs_idx: qd.types.ndarray(), values: qd.types.ndarray(), field: qd.template()):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        field[i_b] = values[i_b_]


@qd.kernel
def kernel_reset_attachment(envs_idx: qd.types.ndarray(), binding: qd.template()):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        binding.position_error_max[i_b] = 0.0
        binding.particle_impulse[i_b] = 0.0
        binding.particle_angular_impulse[i_b] = 0.0
        binding.damping_work[i_b] = 0.0
        binding.spring_potential[i_b] = 0.0
        binding.interval[i_b] = 0.0
        for i_l in range(binding.part_rigid.n_links):
            binding.link_impulse[i_l, i_b] = 0.0
            binding.link_angular_impulse[i_l, i_b] = 0.0
