import numpy as np
from scipy.spatial import cKDTree

import quadrants as qd

import genesis as gs
from genesis.utils import hybrid_skin


class GridRegions:
    """MPM contact fields and their scene-entity, semantic-region, surface-contact and seam identities.

    Fields of one semantic region always share grid velocities. Two regions of one skin that meet on a surface triangle
    share them only at grid nodes where both carry particles of their common seam band, so the skin stays one continuum
    at the seam while parts of the two regions that meet elsewhere contact each other.
    """

    def __init__(self, solver):
        bindings = {binding.part_soft.idx: binding for binding in solver._hybrid_bindings}
        self.particles_region = np.empty(solver.n_particles, dtype=gs.np_int)
        self.names = []
        self.entities = []
        region_ids = []
        surface_contact = []
        fields_by_entity = {}
        for entity in solver.entities:
            binding = bindings.get(entity.idx)
            fields = []
            if binding is None:
                self.particles_region[entity.particle_start : entity.particle_end] = len(self.names)
                fields.append((len(self.names), 0))
                self.names.append(entity.name)
                self.entities.append(entity.idx)
                region_ids.append(0)
                surface_contact.append(True)
            else:
                for i_r, name in enumerate(binding.region_names):
                    in_region = binding.particles_region == i_r
                    for flag in np.unique(binding.particle_surface_contact[in_region]):
                        i_field = len(self.names)
                        fields.append((i_field, i_r))
                        selected = in_region & (binding.particle_surface_contact == flag)
                        particles = entity.particle_start + np.flatnonzero(selected)
                        self.particles_region[particles] = i_field
                        self.names.append(name)
                        self.entities.append(entity.idx)
                        region_ids.append(i_r)
                        surface_contact.append(flag)
            fields_by_entity[entity.idx] = fields
        self.n_regions = len(self.names)
        is_bonded = np.eye(self.n_regions, dtype=bool)
        seam_bit = np.zeros((self.n_regions, self.n_regions), dtype=gs.np_int)
        particles_seam_mask = np.zeros(solver.n_particles, dtype=gs.np_int)
        is_collision_enabled = np.ones((self.n_regions, max(1, solver.sim.rigid_solver.n_geoms)), dtype=bool)
        for binding in solver._hybrid_bindings:
            fields = fields_by_entity[binding.part_soft.idx]
            region_overrides = binding.collision_options.region_overrides or {}
            unknown = set(region_overrides) - set(binding.region_names)
            if unknown:
                gs.raise_exception(f"Unknown Hybrid collision regions: {sorted(unknown)}.")
            # Each region numbers the regions it meets on the surface and marks the seam with each of them by one bit.
            # A particle of region i belongs to its seam band with region j when a particle of j lies closer than three
            # cells along every axis in the reference configuration, the widest separation at which two quadratic
            # B-spline stencils still share a node. Every node the two regions share at rest thus carries both bands
            # and bonds as in a single continuum, whereas skin that only meets after deforming contacts instead.
            regions_seam_bit = np.zeros((len(binding.region_names), len(binding.region_names)), dtype=gs.np_int)
            if binding.vertex_regions is not None:
                is_adjacent = hybrid_skin.region_adjacency(
                    binding.vertex_regions, binding.surface_world.faces, len(binding.region_names)
                )
                for i_r in range(len(binding.region_names)):
                    regions_seam_bit[i_r, is_adjacent[i_r]] = 1 << np.arange(is_adjacent[i_r].sum())
                particles = binding.part_soft.init_particles
                for i_r, j_r in np.argwhere(is_adjacent):
                    particles_idx = np.flatnonzero(binding.particles_region == i_r)
                    distance, _ = cKDTree(particles[binding.particles_region == j_r]).query(
                        particles[particles_idx], p=np.inf, distance_upper_bound=3.0 * solver.dx
                    )
                    seam_idx = binding.part_soft.particle_start + particles_idx[np.isfinite(distance)]
                    particles_seam_mask[seam_idx] |= regions_seam_bit[i_r, j_r]
            neighbors = {link.idx: {link.idx} for link in binding.part_rigid.links}
            for link in binding.part_rigid.links:
                if link.parent_idx >= 0:
                    neighbors[link.idx].add(link.parent_idx)
                    neighbors[link.parent_idx].add(link.idx)
            for i_field, i_r in fields:
                role = region_overrides.get(binding.region_names[i_r], binding.collision_options.external_surface)
                if role == "rigid" or not surface_contact[i_field]:
                    is_collision_enabled[i_field] = False
                links = binding.links_idx[binding.particles_region == i_r]
                excluded_links = set()
                for i_l in np.unique(links[links >= 0]):
                    excluded_links.update(neighbors[i_l])
                for link in binding.part_rigid.links:
                    if link.idx in excluded_links:
                        for geom in link.geoms:
                            is_collision_enabled[i_field, geom.idx] = False
                for j_field, j_r in fields:
                    is_bonded[i_field, j_field] = i_r == j_r
                    seam_bit[i_field, j_field] = regions_seam_bit[i_r, j_r]
        self.entity_idx = qd.field(gs.qd_int, shape=self.n_regions)
        self.region_idx = qd.field(gs.qd_int, shape=self.n_regions)
        self.is_surface_contact = qd.field(gs.qd_bool, shape=self.n_regions)
        self.is_bonded = qd.field(gs.qd_bool, shape=(self.n_regions, self.n_regions))
        self.is_collision_enabled = qd.field(gs.qd_bool, shape=is_collision_enabled.shape)
        self.entity_idx.from_numpy(np.array(self.entities, dtype=gs.np_int))
        self.region_idx.from_numpy(np.array(region_ids, dtype=gs.np_int))
        self.is_surface_contact.from_numpy(np.array(surface_contact, dtype=bool))
        self.is_bonded.from_numpy(is_bonded)
        self.is_collision_enabled.from_numpy(is_collision_enabled)
        # Only a skin whose regions meet has seams, and every seam access sits under a static 'has_seams' gate.
        self.has_seams = bool(seam_bit.any())
        self.seam_bit = None
        self.particles_seam_mask = None
        if self.has_seams:
            self.seam_bit = qd.field(gs.qd_int, shape=(self.n_regions, self.n_regions))
            self.particles_seam_mask = qd.field(gs.qd_int, shape=solver.n_particles)
            self.seam_bit.from_numpy(seam_bit)
            self.particles_seam_mask.from_numpy(particles_seam_mask)


@qd.kernel
def kernel_particle_bounds(f: qd.i32, solver: qd.template()):
    # Atomic min/max of floats onto the per-environment bounds would serialize every particle of an environment on the
    # same few words. Each chunk instead reduces a strided subset of the particles in registers, then one thread per
    # environment reduces the chunk partials.
    for i_chunk, i_b in qd.ndrange(solver._n_bounds_chunks, solver._B):
        lower = qd.Vector([np.inf, np.inf, np.inf], dt=gs.qd_float)
        upper = -lower
        n_active = 0
        for i_p_ in range((solver.n_particles - i_chunk + solver._n_bounds_chunks - 1) // solver._n_bounds_chunks):
            i_p = i_p_ * solver._n_bounds_chunks + i_chunk
            if solver.particles_ng[f, i_p, i_b].active:
                position = solver.particles.pos[f, i_p, i_b]
                is_finite = True
                for i_axis in qd.static(range(3)):
                    if qd.math.isnan(position[i_axis]) or qd.math.isinf(position[i_axis]):
                        is_finite = False
                if is_finite:
                    lower = qd.min(lower, position)
                    upper = qd.max(upper, position)
                    n_active += 1
                else:
                    qd.atomic_or(solver.window_errno[i_b], 2)
        solver.particles_bounds_chunks[i_chunk, i_b].lower = lower
        solver.particles_bounds_chunks[i_chunk, i_b].upper = upper
        solver.particles_bounds_chunks[i_chunk, i_b].n_active = n_active
    for i_b in range(solver._B):
        lower = qd.Vector([np.inf, np.inf, np.inf], dt=gs.qd_float)
        upper = -lower
        n_active = 0
        for i_chunk in range(solver._n_bounds_chunks):
            lower = qd.min(lower, solver.particles_bounds_chunks[i_chunk, i_b].lower)
            upper = qd.max(upper, solver.particles_bounds_chunks[i_chunk, i_b].upper)
            n_active += solver.particles_bounds_chunks[i_chunk, i_b].n_active
        solver.particles_lower[i_b] = lower
        solver.particles_upper[i_b] = upper
        solver.active_particles[i_b] = n_active


@qd.kernel
def kernel_update_window(solver: qd.template()):
    for i_b in range(solver._B):
        if solver.active_particles[i_b] > 0:
            lower = solver.particles_lower[i_b] * solver.inv_dx
            upper = solver.particles_upper[i_b] * solver.inv_dx
            for i_axis in qd.static(range(3)):
                feasible_lower = qd.cast(qd.ceil(upper[i_axis]), gs.qd_int) - solver._grid_res[i_axis] + 4
                feasible_upper = qd.cast(qd.floor(lower[i_axis]), gs.qd_int) - 3
                feasible_lower = qd.max(feasible_lower, solver._lower_bound_cell[i_axis])
                feasible_upper = qd.min(feasible_upper, solver._upper_bound_cell[i_axis] - solver._grid_res[i_axis] + 1)
                if feasible_lower > feasible_upper:
                    qd.atomic_or(solver.window_errno[i_b], 1)
                else:
                    centered = (
                        qd.cast(qd.floor((lower[i_axis] + upper[i_axis]) / 2.0), gs.qd_int)
                        - solver._grid_res[i_axis] // 2
                    )
                    solver.grid_origin[i_b][i_axis] = qd.min(qd.max(centered, feasible_lower), feasible_upper)


@qd.kernel
def kernel_contact_grid_fields(f: qd.i32, solver: qd.template()):
    for i_c, i_b in qd.ndrange(solver.grid_dirty_count[0], solver._B):
        flat = solver.grid_dirty_list[i_c]
        k = flat % solver._grid_res[2]
        remaining = flat // solver._grid_res[2]
        j = remaining % solver._grid_res[1]
        i = remaining // solver._grid_res[1]
        cell = qd.Vector([i, j, k])
        # Every node with mass gets its velocity here, including the nodes at or below the mass threshold of the
        # coupler's grid operation, which g2p still reads. A multi-region grid first resolves contact between regions.
        if qd.static(solver.regions.n_regions == 1):
            mass = solver.grid[f, cell, 0, i_b].mass
            if mass > 0.0:
                solver.grid[f, cell, 0, i_b].vel_out = solver.grid[f, cell, 0, i_b].vel_in / mass
        else:
            # Connected components use only regions with mass at this node. An absent intermediate region therefore
            # cannot weld two separated parts of the same continuous skin
            parent = qd.Vector.zero(gs.qd_int, solver.regions.n_regions)
            for i_r in range(solver.regions.n_regions):
                parent[i_r] = i_r
            for i_r in range(solver.regions.n_regions):
                if solver.grid[f, cell, i_r, i_b].mass > 0.0:
                    for j_r in range(i_r + 1, solver.regions.n_regions):
                        if solver.grid[f, cell, j_r, i_b].mass > 0.0:
                            is_joined = solver.regions.is_bonded[i_r, j_r]
                            # Regions of one skin that meet on its surface join only where each carries particles of
                            # their common seam band (see GridRegions).
                            if qd.static(solver.regions.has_seams):
                                seam_bit_ij = solver.regions.seam_bit[i_r, j_r]
                                if seam_bit_ij != 0:
                                    seam_bit_ji = solver.regions.seam_bit[j_r, i_r]
                                    is_joined = (solver.grid_seam_mask[f, cell, i_r, i_b] & seam_bit_ij) != 0 and (
                                        solver.grid_seam_mask[f, cell, j_r, i_b] & seam_bit_ji
                                    ) != 0
                            if is_joined:
                                root_a = i_r
                                root_b = j_r
                                while parent[root_a] != root_a:
                                    root_a = parent[root_a]
                                while parent[root_b] != root_b:
                                    root_b = parent[root_b]
                                parent[qd.max(root_a, root_b)] = qd.min(root_a, root_b)
            masses = qd.Vector.zero(gs.qd_float, solver.regions.n_regions)
            momenta = qd.Matrix.zero(gs.qd_float, solver.regions.n_regions, 3)
            gradients = qd.Matrix.zero(gs.qd_float, solver.regions.n_regions, 3)
            frictions = qd.Vector.zero(gs.qd_float, solver.regions.n_regions)
            surface_contact = qd.Vector.zero(gs.qd_int, solver.regions.n_regions)
            # A region without mass at this node remains its own component, which neither the contact impulses nor the
            # velocity update read, so only the occupied regions are aggregated.
            for i_r in range(solver.regions.n_regions):
                mass = solver.grid[f, cell, i_r, i_b].mass
                if mass > 0.0:
                    root = i_r
                    while parent[root] != root:
                        root = parent[root]
                    parent[i_r] = root
                    masses[root] += mass
                    frictions[root] += solver.grid_contact[f, cell, i_r, i_b].friction_mass
                    if solver.regions.is_surface_contact[i_r]:
                        surface_contact[root] = 1
                    for i_axis in qd.static(range(3)):
                        momenta[root, i_axis] += solver.grid[f, cell, i_r, i_b].vel_in[i_axis]
                        gradients[root, i_axis] += solver.grid_contact[f, cell, i_r, i_b].mass_gradient[i_axis]
            for i_r in range(solver.regions.n_regions):
                if masses[i_r] > 0.0 and surface_contact[i_r]:
                    for j_r in range(i_r + 1, solver.regions.n_regions):
                        if masses[j_r] > 0.0 and surface_contact[j_r]:
                            velocity_a = qd.Vector([momenta[i_r, 0], momenta[i_r, 1], momenta[i_r, 2]]) / masses[i_r]
                            velocity_b = qd.Vector([momenta[j_r, 0], momenta[j_r, 1], momenta[j_r, 2]]) / masses[j_r]
                            normal_a = (
                                qd.Vector([gradients[i_r, 0], gradients[i_r, 1], gradients[i_r, 2]]) / masses[i_r]
                            )
                            normal_b = (
                                qd.Vector([gradients[j_r, 0], gradients[j_r, 1], gradients[j_r, 2]]) / masses[j_r]
                            )
                            normal = normal_a - normal_b
                            norm = normal.norm()
                            if norm > gs.EPS:
                                normal /= norm
                                relative_velocity = velocity_a - velocity_b
                                speed = relative_velocity.dot(normal)
                                if speed < 0.0:
                                    reduced_mass = masses[i_r] * masses[j_r] / (masses[i_r] + masses[j_r])
                                    impulse = -reduced_mass * speed * normal
                                    tangent = relative_velocity - speed * normal
                                    tangent_speed = tangent.norm()
                                    if tangent_speed > gs.EPS:
                                        friction = qd.sqrt(
                                            frictions[i_r] * frictions[j_r] / (masses[i_r] * masses[j_r])
                                        )
                                        impulse -= (
                                            qd.min(reduced_mass * tangent_speed, friction * impulse.norm())
                                            * tangent
                                            / tangent_speed
                                        )
                                    for i_axis in qd.static(range(3)):
                                        momenta[i_r, i_axis] += impulse[i_axis]
                                        momenta[j_r, i_axis] -= impulse[i_axis]
            for i_r in range(solver.regions.n_regions):
                if solver.grid[f, cell, i_r, i_b].mass > 0.0:
                    root = parent[i_r]
                    for i_axis in qd.static(range(3)):
                        solver.grid[f, cell, i_r, i_b].vel_out[i_axis] = momenta[root, i_axis] / masses[root]


@qd.kernel
def kernel_set_grid_origin(envs_idx: qd.types.ndarray(), origin: qd.types.ndarray(), solver: qd.template()):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_axis in qd.static(range(3)):
            solver.grid_origin[i_b][i_axis] = origin[i_b, i_axis]
        solver.window_errno[i_b] = 0
        solver.active_cell_count[i_b] = 0
