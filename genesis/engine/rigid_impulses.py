import quadrants as qd

import genesis as gs
from genesis.utils import array_class


class RigidImpulses:
    """Accumulated world impulses consumed synchronously without advancing rigid positions."""

    def __init__(self, solver):
        self.solver = solver
        self.linear = qd.field(gs.qd_vec3, shape=(solver.n_links_, solver._B))
        self.angular = qd.field(gs.qd_vec3, shape=(solver.n_links_, solver._B))
        self.generalized = array_class.V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B))
        self.velocity_delta = array_class.V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B))

    def clear(self):
        self.linear.fill(0.0)
        self.angular.fill(0.0)

    def apply(self):
        """Change the dof velocities by the accumulated impulses through the mass matrix and clear the accumulator.

        The velocity change takes effect at the end of the current substep, before the next substep of any solver.
        The forward dynamics of the next substep of the same step reuses the mass matrix assembled here (see
        RigidSolver._is_mass_mat_updated), unless the link poses predate the integration of this substep or the bodies
        may hibernate.
        """
        from genesis.engine.solvers.rigid.abd.impulses import kernel_apply_rigid_impulses

        if not self.solver.n_dofs:
            self.clear()
            return
        sim = self.solver.sim
        is_mass_mat_reused = (
            self.solver._is_forward_pos_updated
            and not self.solver.rigid_config.use_hibernation
            and (sim.cur_substep_global + 1) % sim.substeps > 0
        )
        kernel_apply_rigid_impulses(
            is_mass_mat_reused,
            self.generalized,
            self.velocity_delta,
            self,
            self.solver.dyn_state,
            self.solver.constraint_solver.constraint_state,
            self.solver.dyn_info,
            self.solver.rigid_info,
            self.solver.rigid_config,
        )
        self.solver._is_mass_mat_updated = is_mass_mat_reused


@qd.func
def add_rigid_impulse(i_l, i_b, position, impulse, impulses: qd.template()):
    impulses.linear[i_l, i_b] += impulse
    impulses.angular[i_l, i_b] += position.cross(impulse)
