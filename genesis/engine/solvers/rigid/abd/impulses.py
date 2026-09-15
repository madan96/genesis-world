import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .forward_dynamics import func_compute_mass_matrix, func_factor_mass, func_solve_mass
from .forward_kinematics import func_forward_velocity_batch
from .misc import func_wakeup_island, func_wakeup_link


@qd.kernel
def kernel_apply_rigid_impulses(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    generalized: qd.Tensor,
    velocity_delta: qd.Tensor,
    impulses: qd.template(),
):
    if qd.static(rigid_config.use_hibernation):
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
    for i_d, i_b in qd.ndrange(generalized.shape[0], generalized.shape[1]):
        generalized[i_d, i_b] = 0.0
        velocity_delta[i_d, i_b] = 0.0
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        linear = impulses.linear[i_l, i_b]
        angular = impulses.angular[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b].cross(linear)
        j_l = i_l
        while j_l >= 0:
            J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
            for i_d in range(dyn_info.links.dof_start[J_l], dyn_info.links.dof_end[J_l]):
                generalized[i_d, i_b] += dyn_state.dofs.cdof_vel[i_d, i_b].dot(linear) + dyn_state.dofs.cdof_ang[
                    i_d, i_b
                ].dot(angular)
            j_l = dyn_info.links.parent_idx[J_l]
    for i_e, i_b in qd.ndrange(rigid_info.mass_mat_mask.shape[0], rigid_info.mass_mat_mask.shape[1]):
        rigid_info.mass_mat_mask[i_e, i_b] = True
    func_compute_mass_matrix(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False, is_backward=False)
    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)
    func_solve_mass(generalized, velocity_delta, dyn_state, dyn_info, rigid_info, rigid_config)
    for i_d, i_b in qd.ndrange(dyn_state.dofs.vel.shape[0], dyn_state.dofs.vel.shape[1]):
        dyn_state.dofs.vel[i_d, i_b] += velocity_delta[i_d, i_b]
    for i_b in range(dyn_state.links.pos.shape[1]):
        func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
