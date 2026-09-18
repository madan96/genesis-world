import numpy as np

import quadrants as qd

import genesis as gs


class GridRegions:
    """MPM contact fields and their scene-entity, semantic-region and surface-contact identities."""

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
        is_collision_enabled = np.ones((self.n_regions, max(1, solver.sim.rigid_solver.n_geoms)), dtype=bool)
        for binding in solver._hybrid_bindings:
            fields = fields_by_entity[binding.part_soft.idx]
            region_overrides = binding.collision_options.region_overrides or {}
            unknown = set(region_overrides) - set(binding.region_names)
            if unknown:
                gs.raise_exception(f"Unknown Hybrid collision regions: {sorted(unknown)}.")
            adjacent = np.eye(len(binding.region_names), dtype=bool)
            if binding.vertex_regions is not None:
                triangles = binding.vertex_regions[binding.surface_world.faces]
                for i_corner in range(3):
                    for j_corner in range(i_corner + 1, 3):
                        adjacent[triangles[:, i_corner], triangles[:, j_corner]] = True
                        adjacent[triangles[:, j_corner], triangles[:, i_corner]] = True
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
                    is_bonded[i_field, j_field] = adjacent[i_r, j_r]
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


@qd.kernel
def kernel_particle_bounds(f: qd.i32, solver: qd.template()):
    for i_b in range(solver._B):
        solver.particles_lower[i_b] = np.inf
        solver.particles_upper[i_b] = -np.inf
        solver.active_particles[i_b] = 0
    for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
        if solver.particles_ng[f, i_p, i_b].active:
            position = solver.particles.pos[f, i_p, i_b]
            is_finite = True
            for i_axis in qd.static(range(3)):
                if qd.math.isnan(position[i_axis]) or qd.math.isinf(position[i_axis]):
                    is_finite = False
            if is_finite:
                for i_axis in qd.static(range(3)):
                    qd.atomic_min(solver.particles_lower[i_b][i_axis], position[i_axis])
                    qd.atomic_max(solver.particles_upper[i_b][i_axis], position[i_axis])
                qd.atomic_add(solver.active_particles[i_b], 1)
            else:
                qd.atomic_or(solver.window_errno[i_b], 2)


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
        for i_r in range(solver.regions.n_regions):
            mass = solver.grid[f, cell, i_r, i_b].mass
            if mass > 0.0:
                solver.grid[f, cell, i_r, i_b].vel_out = solver.grid[f, cell, i_r, i_b].vel_in / mass
                solver.active_cell_count[i_b] += 1
        if qd.static(solver.regions.n_regions > 1):
            # Connected components use only regions with mass at this node. An absent intermediate region therefore
            # cannot weld two separated parts of the same continuous skin
            parent = qd.Vector.zero(gs.qd_int, solver.regions.n_regions)
            for i_r in range(solver.regions.n_regions):
                parent[i_r] = i_r
            for i_r in range(solver.regions.n_regions):
                if solver.grid[f, cell, i_r, i_b].mass > 0.0:
                    for j_r in range(i_r + 1, solver.regions.n_regions):
                        if solver.grid[f, cell, j_r, i_b].mass > 0.0 and solver.regions.is_bonded[i_r, j_r]:
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
            for i_r in range(solver.regions.n_regions):
                root = i_r
                while parent[root] != root:
                    root = parent[root]
                parent[i_r] = root
                mass = solver.grid[f, cell, i_r, i_b].mass
                masses[root] += mass
                frictions[root] += solver.grid[f, cell, i_r, i_b].friction_mass
                if mass > 0.0 and solver.regions.is_surface_contact[i_r]:
                    surface_contact[root] = 1
                for i_axis in qd.static(range(3)):
                    momenta[root, i_axis] += solver.grid[f, cell, i_r, i_b].vel_in[i_axis]
                    gradients[root, i_axis] += solver.grid[f, cell, i_r, i_b].mass_gradient[i_axis]
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
                root = parent[i_r]
                if solver.grid[f, cell, i_r, i_b].mass > 0.0:
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
