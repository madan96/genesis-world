import quadrants as qd

import genesis as gs
from genesis.engine.rigid_impulses import add_rigid_impulse
from genesis.utils import array_class, geom


@qd.kernel
def kernel_attachment_stability(
    f: qd.i32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_config: qd.template(),
    binding: qd.template(),
):
    for i_l, i_b in qd.ndrange(binding.part_rigid.n_links, binding.part_soft.solver._B):
        binding.link_soft_mass[i_l, i_b] = 0.0
        binding.link_weighted_radius[i_l, i_b] = 0.0
    for i_b in range(binding.part_soft.solver._B):
        binding.mobility_bound[i_b] = 0.0
    for i_p_, i_b in qd.ndrange(binding.n_particles, binding.part_soft.solver._B):
        i_p = binding.part_soft.particle_start + i_p_
        if binding.part_soft.solver.particles_ng[f, i_p, i_b].active:
            position = binding.part_soft.solver.particles.pos[f, i_p, i_b]
            mass = binding.part_soft.solver.particles_info.mass[i_p] / binding.part_soft.solver.particle_volume_scale
            for i_k in range(binding.bind_active_count[i_p_]):
                i_l = binding.bind_links_idx[i_p_, i_k]
                weight = binding.bind_weights[i_p_, i_k]
                anchor = geom.qd_transform_by_trans_quat(
                    binding.bind_rest_local_pos[i_p_, i_k], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                # Link center of mass in world coordinates: i_pos is relative to root_COM (see forward_kinematics.py).
                # The spring acts at its bound target and objective damping acts at the particle.
                center = links_state.i_pos[i_l, i_b] + links_state.root_COM[i_l, i_b]
                radius = qd.max((position - center).norm(), (anchor - center).norm())
                binding.link_soft_mass[i_l - binding.part_rigid.link_start, i_b] += mass * weight
                binding.link_weighted_radius[i_l - binding.part_rigid.link_start, i_b] += mass * weight * radius
    for i_p_, i_b in qd.ndrange(binding.n_particles, binding.part_soft.solver._B):
        i_p = binding.part_soft.particle_start + i_p_
        if binding.part_soft.solver.particles_ng[f, i_p, i_b].active:
            position = binding.part_soft.solver.particles.pos[f, i_p, i_b]
            bound = gs.qd_float(0.0)
            for i_k in range(binding.bind_active_count[i_p_]):
                i_l = binding.bind_links_idx[i_p_, i_k]
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if not links_info.is_fixed[I_l]:
                    weight = binding.bind_weights[i_p_, i_k]
                    anchor = geom.qd_transform_by_trans_quat(
                        binding.bind_rest_local_pos[i_p_, i_k], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                    )
                    center = links_state.i_pos[i_l, i_b] + links_state.root_COM[i_l, i_b]
                    radius = qd.max((position - center).norm(), (anchor - center).norm())
                    mass = links_info.inertial_mass[I_l]
                    inertia = links_info.inertial_i[I_l]
                    if mass > 0.0 and inertia.determinant() > 0.0:
                        inverse_inertia_norm = qd.sqrt((inertia.inverse() ** 2).sum())
                        i_l_ = i_l - binding.part_rigid.link_start
                        bound += weight * (
                            binding.link_soft_mass[i_l_, i_b] / mass
                            + radius * binding.link_weighted_radius[i_l_, i_b] * inverse_inertia_norm
                        )
                    else:
                        bound = float("inf")
            qd.atomic_max(binding.mobility_bound[i_b], bound)
    for i_b in range(binding.part_soft.solver._B):
        dt_omega = binding.part_soft.solver.substep_dt * binding.omega[i_b]
        # The local linear drift/kick modal limit is dt²*k_eff + 2*dt*c_eff < 4. The supported range reserves half
        # that interval. Free-link mobility bounds the constrained articulation; weighted block-row sums bound both
        # spring and damping operators, including their different lever arms
        binding.stability_index[i_b] = (
            (1.0 + binding.mobility_bound[i_b]) * (dt_omega**2 + 4.0 * binding.damping_ratio[i_b] * dt_omega) / 2.0
        )


@qd.kernel
def kernel_apply_attachment(f: qd.i32, links_state: array_class.LinksState, binding: qd.template()):
    dt = binding.part_soft.solver.substep_dt
    for i_p_, i_b in qd.ndrange(binding.n_particles, binding.part_soft.solver._B):
        i_p = binding.part_soft.particle_start + i_p_
        if binding.part_soft.solver.particles_ng[f, i_p, i_b].active:
            position = binding.part_soft.solver.particles.pos[f, i_p, i_b]
            velocity = binding.part_soft.solver.particles.vel[f, i_p, i_b]
            mass = binding.part_soft.solver.particles_info.mass[i_p] / binding.part_soft.solver.particle_volume_scale
            target_position = qd.Vector.zero(gs.qd_float, 3)
            target_velocity = qd.Vector.zero(gs.qd_float, 3)
            count = binding.bind_active_count[i_p_]
            for i_k in range(count):
                i_l = binding.bind_links_idx[i_p_, i_k]
                weight = binding.bind_weights[i_p_, i_k]
                target_position += weight * geom.qd_transform_by_trans_quat(
                    binding.bind_rest_local_pos[i_p_, i_k], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                target_velocity += weight * (
                    links_state.cd_vel[i_l, i_b]
                    + links_state.cd_ang[i_l, i_b].cross(position - links_state.root_COM[i_l, i_b])
                )
            omega = binding.omega[i_b]
            displacement = target_position - position
            spring_impulse = dt * mass * omega**2 * displacement
            damping_impulse = dt * 2.0 * mass * binding.damping_ratio[i_b] * omega * (target_velocity - velocity)
            if qd.static(binding.is_force_limited):
                impulse_norm = (spring_impulse + damping_impulse).norm()
                limit = binding.options.max_correction_force * dt
                if impulse_norm > limit:
                    spring_impulse *= limit / impulse_norm
                    damping_impulse *= limit / impulse_norm
            impulse = spring_impulse + damping_impulse
            binding.part_soft.solver.particles.vel[f, i_p, i_b] = velocity + impulse / mass
            for i_k in range(count):
                i_l = binding.bind_links_idx[i_p_, i_k]
                weight = binding.bind_weights[i_p_, i_k]
                anchor = geom.qd_transform_by_trans_quat(
                    binding.bind_rest_local_pos[i_p_, i_k], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                spring_reaction = -weight * spring_impulse
                damping_reaction = -weight * damping_impulse
                add_rigid_impulse(
                    i_l, i_b, anchor, spring_reaction, binding.part_soft.solver.sim.coupler.rigid_impulses
                )
                add_rigid_impulse(
                    i_l, i_b, position, damping_reaction, binding.part_soft.solver.sim.coupler.rigid_impulses
                )
                binding.link_impulse[i_l - binding.part_rigid.link_start, i_b] += spring_reaction + damping_reaction
                binding.link_angular_impulse[i_l - binding.part_rigid.link_start, i_b] += anchor.cross(
                    spring_reaction
                ) + position.cross(damping_reaction)
            qd.atomic_max(binding.position_error_max[i_b], displacement.norm())
            binding.particle_impulse[i_b] += impulse
            binding.particle_angular_impulse[i_b] += position.cross(impulse)
            binding.damping_work[i_b] += damping_impulse.dot(velocity - target_velocity)
            binding.spring_potential[i_b] += 0.5 * mass * omega**2 * displacement.norm_sqr()
