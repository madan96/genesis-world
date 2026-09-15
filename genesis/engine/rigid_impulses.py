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
        from genesis.engine.solvers.rigid.abd.impulses import kernel_apply_rigid_impulses

        if self.solver.n_dofs:
            kernel_apply_rigid_impulses(
                self.solver.dyn_state,
                self.solver.constraint_solver.constraint_state,
                self.solver.dyn_info,
                self.solver.rigid_info,
                self.solver.rigid_config,
                self.generalized,
                self.velocity_delta,
                self,
            )
            self.solver._is_forward_vel_updated = True
        self.clear()


@qd.func
def add_rigid_impulse(i_l, i_b, position, impulse, impulses: qd.template()):
    impulses.linear[i_l, i_b] += impulse
    impulses.angular[i_l, i_b] += position.cross(impulse)
