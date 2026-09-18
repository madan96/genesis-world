import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom


@qd.func
def func_point_rotational_mobility(arm, inverse_inertia):
    """Return the largest eigenvalue of S^T I^-1 S, with S the cross-product matrix of ``arm``.

    It is the largest velocity change per unit impulse that an impulse at the tip of ``arm`` induces at that tip through
    the rotation of a body with inverse inertia ``inverse_inertia`` about the base of ``arm``, both in the same frame.
    """
    skew = qd.Matrix([[0.0, -arm[2], arm[1]], [arm[2], 0.0, -arm[0]], [-arm[1], arm[0], 0.0]])
    mobility = skew.transpose() @ inverse_inertia @ skew
    mobility_mean = 0.5 * mobility.trace()
    # S has rank two, so the mobility has eigenvalues l1, l2 on the plane normal to the arm and zero along it. Its
    # deviation from their mean on that plane has Frobenius norm |l1 - l2| / sqrt(2), a sum of squares that stays
    # accurate where the invariant form 2 tr(B^2) - tr(B)^2 cancels.
    plane = qd.Matrix.identity(gs.qd_float, 3) - arm.outer_product(arm) / qd.max(arm.norm_sqr(), gs.EPS)
    deviation = mobility - mobility_mean * plane
    return mobility_mean + qd.sqrt(0.5 * (deviation**2).sum())


@qd.func
def func_attachment_lever(
    i_p_, i_k, i_l, i_b, position, inverse_inertia, links_state: array_class.LinksState, binding: qd.template()
):
    """Return the lever gain of influence ``i_k`` of particle ``i_p_`` about the center of mass of link ``i_l``.

    The gain is the square root of the larger func_point_rotational_mobility of the spring arm, which ends at the bound
    target, and of the damping arm, which ends at the particle ``position``. ``inverse_inertia`` is expressed in the
    link inertial frame.
    """
    anchor = geom.qd_transform_by_trans_quat(
        binding.bind_rest_local_pos[i_p_, i_k], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
    )
    # Link center of mass in world coordinates: i_pos is relative to root_COM (see forward_kinematics.py).
    center = links_state.i_pos[i_l, i_b] + links_state.root_COM[i_l, i_b]
    quat = links_state.i_quat[i_l, i_b]
    mobility_spring = func_point_rotational_mobility(
        geom.qd_inv_transform_by_quat(anchor - center, quat), inverse_inertia
    )
    mobility_damping = func_point_rotational_mobility(
        geom.qd_inv_transform_by_quat(position - center, quat), inverse_inertia
    )
    return qd.sqrt(qd.max(mobility_spring, mobility_damping))


@qd.kernel
def kernel_attachment_stability(
    f: qd.i32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_config: qd.template(),
    binding: qd.template(),
):
    for i_l_, i_b in qd.ndrange(binding.part_rigid.n_links, binding.part_soft.solver._B):
        binding.link_weighted_lever[i_l_, i_b] = 0.0
    for i_b in range(binding.part_soft.solver._B):
        binding.mobility_bound[i_b] = 0.0
    for i_p_, i_b in qd.ndrange(binding.n_particles, binding.part_soft.solver._B):
        i_p = binding.part_soft.particle_start + i_p_
        if binding.part_soft.solver.particles_ng[f, i_p, i_b].active:
            position = binding.part_soft.solver.particles.pos[f, i_p, i_b]
            mass = binding.part_soft.solver.particles_info.mass[i_p] / binding.part_soft.solver.particle_volume_scale
            for i_k in range(binding.bind_active_count[i_p_]):
                i_l = binding.bind_links_idx[i_p_, i_k]
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                inertia = links_info.inertial_i[I_l]
                if not links_info.is_fixed[I_l] and inertia.determinant() > 0.0:
                    lever = func_attachment_lever(
                        i_p_, i_k, i_l, i_b, position, inertia.inverse(), links_state, binding
                    )
                    weight = binding.bind_weights[i_p_, i_k]
                    binding.link_weighted_lever[i_l - binding.part_rigid.link_start, i_b] += mass * weight * lever
    for i_p_, i_b in qd.ndrange(binding.n_particles, binding.part_soft.solver._B):
        i_p = binding.part_soft.particle_start + i_p_
        if binding.part_soft.solver.particles_ng[f, i_p, i_b].active:
            position = binding.part_soft.solver.particles.pos[f, i_p, i_b]
            bound = gs.qd_float(0.0)
            for i_k in range(binding.bind_active_count[i_p_]):
                i_l = binding.bind_links_idx[i_p_, i_k]
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if not links_info.is_fixed[I_l]:
                    mass = links_info.inertial_mass[I_l]
                    inertia = links_info.inertial_i[I_l]
                    if mass > 0.0 and inertia.determinant() > 0.0:
                        lever = func_attachment_lever(
                            i_p_, i_k, i_l, i_b, position, inertia.inverse(), links_state, binding
                        )
                        i_l_ = i_l - binding.part_rigid.link_start
                        bound += binding.bind_weights[i_p_, i_k] * (
                            binding.link_soft_mass[i_l_] / mass + lever * binding.link_weighted_lever[i_l_, i_b]
                        )
                    else:
                        bound = float("inf")
            qd.atomic_max(binding.mobility_bound[i_b], bound)
    for i_b in range(binding.part_soft.solver._B):
        dt_omega = binding.part_soft.solver.substep_dt * binding.omega[i_b]
        # The local linear drift/kick modal limit is dt^2*k_eff + 2*dt*c_eff < 4. The supported range reserves half
        # that interval. Free-link mobility bounds the constrained articulation. A free link maps a unit reaction at
        # arm r_q to a velocity at arm r_p through 1/m + S_p^T I^-1 S_q, whose norm is at most 1/m + a_p a_q with the
        # lever gains a of func_attachment_lever, so weighted block-row sums bound both spring and damping operators.
        binding.stability_index[i_b] = (
            (1.0 + binding.mobility_bound[i_b]) * (dt_omega**2 + 4.0 * binding.damping_ratio[i_b] * dt_omega) / 2.0
        )


@qd.kernel
def kernel_apply_attachment(f: qd.i32, links_state: array_class.LinksState, binding: qd.template()):
    dt = binding.part_soft.solver.substep_dt
    # Top-level loops run in order, so the spring energy restarts before the particle loop accumulates it.
    for i_b in range(binding.part_soft.solver._B):
        binding.spring_potential[i_b] = 0.0
        binding.interval[i_b] += dt
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
                reaction = spring_reaction + damping_reaction
                angular_reaction = anchor.cross(spring_reaction) + position.cross(damping_reaction)
                # The spring and damping reactions act on the same link at different points, so they enter the
                # coupler accumulator as one wrench, the sum of two add_rigid_impulse calls with half the atomics.
                binding.part_soft.solver.sim.coupler.rigid_impulses.linear[i_l, i_b] += reaction
                binding.part_soft.solver.sim.coupler.rigid_impulses.angular[i_l, i_b] += angular_reaction
                binding.link_impulse[i_l - binding.part_rigid.link_start, i_b] += reaction
                binding.link_angular_impulse[i_l - binding.part_rigid.link_start, i_b] += angular_reaction
            qd.atomic_max(binding.position_error_max[i_b], displacement.norm())
            binding.particle_impulse[i_b] += impulse
            binding.particle_angular_impulse[i_b] += position.cross(impulse)
            binding.damping_work[i_b] += damping_impulse.dot(velocity - target_velocity)
            binding.spring_potential[i_b] += 0.5 * mass * omega**2 * displacement.norm_sqr()
