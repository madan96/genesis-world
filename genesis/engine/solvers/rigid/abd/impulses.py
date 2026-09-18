import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .forward_dynamics import func_compute_mass_matrix, func_factor_mass, func_solve_mass
from .forward_kinematics import func_forward_velocity_batch
from .misc import func_wakeup_island, func_wakeup_link


@qd.kernel
def kernel_apply_rigid_impulses(
    factor_impulsed_only: bool,
    generalized: qd.Tensor,
    velocity_delta: qd.Tensor,
    impulses: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Apply the accumulated world impulses to the dof velocities and clear the accumulator.

    The velocity change dq of every mass block solves M dq = J^T p, with the mass matrix M assembled at the current
    configuration without implicit damping. M stays in rigid_info.mass_mat for all environments. The environments whose
    dof velocities change refresh their link velocities, except under MuJoCo compatibility, whose link poses predate
    the integration of the substep and get refreshed with the velocities by the next substep.

    Parameters
    ----------
    factor_impulsed_only : bool
        True factors and solves only the mass blocks receiving a nonzero generalized impulse, for a mass matrix that the
        next forward dynamics factors again. False factors every mass block, so that the factor in rigid_info stays the
        one of rigid_info.mass_mat.
    """
    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
            if impulses.linear[i_l, i_b].norm_sqr() + impulses.angular[i_l, i_b].norm_sqr() > 0.0:
                func_wakeup_island(
                    constraint_state.island.links_island_idx[i_l, i_b],
                    i_b,
                    dyn_state,
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )
                j_l = i_l
                while j_l >= 0:
                    func_wakeup_link(j_l, i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
                    J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
                    j_l = dyn_info.links.parent_idx[J_l]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(generalized.shape[0], generalized.shape[1]):
        generalized[i_d, i_b] = 0.0
        velocity_delta[i_d, i_b] = 0.0

    # Every link projects its impulse on the dofs of its ancestors, which several links share.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        linear = impulses.linear[i_l, i_b]
        angular = impulses.angular[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b].cross(linear)
        impulses.linear[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        impulses.angular[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        j_l = i_l
        while j_l >= 0:
            J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
            for i_d in range(dyn_info.links.dof_start[J_l], dyn_info.links.dof_end[J_l]):
                generalized[i_d, i_b] += dyn_state.dofs.cdof_vel[i_d, i_b].dot(linear) + dyn_state.dofs.cdof_ang[
                    i_d, i_b
                ].dot(angular)
            j_l = dyn_info.links.parent_idx[J_l]

    # The factor and the solve of a mass block both gate on the entity rooting it (see mass_mat_mask in array_class.py).
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in qd.ndrange(rigid_info.mass_mat_mask.shape[0], rigid_info.mass_mat_mask.shape[1]):
        has_impulse = False
        for i_d in range(rigid_info.entities_mass_block_dof_start[i_e], rigid_info.entities_mass_block_dof_end[i_e]):
            if generalized[i_d, i_b] != 0.0:
                has_impulse = True
        rigid_info.mass_mat_mask[i_e, i_b] = has_impulse or not factor_impulsed_only

    func_compute_mass_matrix(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False, is_backward=False)
    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)
    func_solve_mass(generalized, velocity_delta, dyn_state, dyn_info, rigid_info, rigid_config)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(dyn_state.dofs.vel.shape[0], dyn_state.dofs.vel.shape[1]):
        dyn_state.dofs.vel[i_d, i_b] += velocity_delta[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(rigid_info.mass_mat_mask.shape[1]):
        for i_e in range(rigid_info.mass_mat_mask.shape[0]):
            rigid_info.mass_mat_mask[i_e, i_b] = True
        if qd.static(not rigid_config.enable_mujoco_compatibility):
            has_velocity_change = False
            for i_d in range(velocity_delta.shape[0]):
                if velocity_delta[i_d, i_b] != 0.0:
                    has_velocity_change = True
            if has_velocity_change:
                func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
