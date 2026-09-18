"""Passive scalar-joint forces and coupled soft limits for articulated entities."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.utils import array_class
from genesis.utils.misc import assign_indexed_tensor, broadcast_tensor, qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity


@dataclass
class CoupledLimitDescription:
    name: str
    kind: str
    dofs_idx_local: np.ndarray
    n_faces: int


@dataclass
class PassiveJointDescription:
    """Joint selection and coupled-limit structure needed to reconstruct a passive model."""

    dofs_idx_local: np.ndarray
    friction_vel_eps: float
    groups: list[CoupledLimitDescription] = field(default_factory=list)


@dataclass(frozen=True)
class PassiveJointInfo:
    kind: ClassVar[array_class.DataKind] = array_class.DataKind.INFO

    dofs_idx: qd.Tensor = array_class.of_kind(array_class.DataKind.CONSTANT)
    friction_vel_eps: qd.Tensor = array_class.of_kind(array_class.DataKind.CONSTANT)
    q_neutral: qd.Tensor
    k_neutral: qd.Tensor
    soft_lo: qd.Tensor
    soft_hi: qd.Tensor
    k_soft: qd.Tensor
    soft_exponent: qd.Tensor
    damping: qd.Tensor
    friction: qd.Tensor
    force_limit: qd.Tensor


@dataclass(frozen=True)
class PassiveJointState:
    kind: ClassVar[array_class.DataKind] = array_class.DataKind.DERIVED

    passive_force: qd.Tensor
    limit_force: qd.Tensor
    coupled_force: qd.Tensor
    applied_force: qd.Tensor
    potential_energy: qd.Tensor


@dataclass(frozen=True)
class CoupledLimitInfo:
    kind: ClassVar[array_class.DataKind] = array_class.DataKind.INFO

    slots: qd.Tensor = array_class.of_kind(array_class.DataKind.CONSTANT)
    center: qd.Tensor
    radii_pos: qd.Tensor
    radii_neg: qd.Tensor
    faces_normal: qd.Tensor
    faces_offset: qd.Tensor
    stiffness: qd.Tensor
    exponent: qd.Tensor


@dataclass(frozen=True)
class CoupledLimitState:
    kind: ClassVar[array_class.DataKind] = array_class.DataKind.DERIVED

    violation: qd.Tensor
    potential_energy: qd.Tensor


class PassiveJointForces(NamedTuple):
    """Forces of a passive joint model at one state, each of shape ``(B, n_dofs)``."""

    passive: torch.Tensor
    limit: torch.Tensor
    applied: torch.Tensor


@dataclass
class CoupledLimitGroup:
    name: str
    kind: str
    slots: np.ndarray
    n_faces: int
    info: CoupledLimitInfo | None = None
    state: CoupledLimitState | None = None


@qd.kernel
def kernel_set_parameter(
    envs_idx: qd.types.ndarray(),
    value: qd.types.ndarray(),
    parameter: qd.Tensor,
):
    for I in qd.grouped(qd.ndrange(*value.shape)):
        J = qd.Vector.zero(qd.i32, qd.static(len(parameter.shape)))
        for k in qd.static(range(len(parameter.shape) - 1)):
            J[k] = I[k + 1]
        J[qd.static(len(parameter.shape) - 1)] = envs_idx[I[0]]
        parameter[J] = value[I]


@qd.kernel
def kernel_coupled_limit(
    dyn_state: array_class.DynState,
    passive_state: PassiveJointState,
    coupled_state: CoupledLimitState,
    passive_info: PassiveJointInfo,
    coupled_info: CoupledLimitInfo,
    is_ellipsoid: qd.template(),
):
    for i_b in range(dyn_state.dofs.pos.shape[1]):
        stiffness = coupled_info.stiffness[i_b]
        exponent = coupled_info.exponent[i_b]
        violation = gs.qd_float(0.0)
        energy = gs.qd_float(0.0)
        if qd.static(is_ellipsoid):
            radius_squared = gs.qd_float(0.0)
            for i_m in range(coupled_info.slots.shape[0]):
                i_d_ = coupled_info.slots[i_m]
                i_d = passive_info.dofs_idx[i_d_]
                delta = dyn_state.dofs.pos[i_d, i_b] - coupled_info.center[i_m, i_b]
                radius = coupled_info.radii_pos[i_m, i_b]
                if delta < 0.0:
                    radius = coupled_info.radii_neg[i_m, i_b]
                radius_squared += (delta / radius) ** 2
            rho = qd.sqrt(radius_squared)
            violation = qd.max(rho - 1.0, 0.0)
            if rho > 1.0:
                magnitude = stiffness * violation**exponent
                energy = magnitude * violation / (exponent + 1.0)
                for i_m in range(coupled_info.slots.shape[0]):
                    i_d_ = coupled_info.slots[i_m]
                    i_d = passive_info.dofs_idx[i_d_]
                    delta = dyn_state.dofs.pos[i_d, i_b] - coupled_info.center[i_m, i_b]
                    radius = coupled_info.radii_pos[i_m, i_b]
                    if delta < 0.0:
                        radius = coupled_info.radii_neg[i_m, i_b]
                    passive_state.coupled_force[i_d_, i_b] += -magnitude * delta / (rho * radius**2)
        else:
            for i_f in range(coupled_info.faces_offset.shape[0]):
                excess = -coupled_info.faces_offset[i_f, i_b]
                for i_m in range(coupled_info.slots.shape[0]):
                    i_d_ = coupled_info.slots[i_m]
                    i_d = passive_info.dofs_idx[i_d_]
                    excess += coupled_info.faces_normal[i_f, i_m, i_b] * (
                        dyn_state.dofs.pos[i_d, i_b] - coupled_info.center[i_m, i_b]
                    )
                if excess > 0.0:
                    violation = qd.max(violation, excess)
                    magnitude = stiffness * excess**exponent
                    energy += magnitude * excess / (exponent + 1.0)
                    for i_m in range(coupled_info.slots.shape[0]):
                        i_d_ = coupled_info.slots[i_m]
                        passive_state.coupled_force[i_d_, i_b] += -magnitude * coupled_info.faces_normal[i_f, i_m, i_b]
        coupled_state.violation[i_b] = violation
        coupled_state.potential_energy[i_b] = energy


@qd.kernel
def kernel_passive_joints(
    dyn_state: array_class.DynState,
    passive_state: PassiveJointState,
    rigid_info: array_class.RigidInfo,
    passive_info: PassiveJointInfo,
    rigid_config: qd.template(),
    is_apply: qd.template(),
    has_coupled_limits: qd.template(),
):
    EPS = rigid_info.EPS[None]
    friction_vel_eps = passive_info.friction_vel_eps[None]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d_, i_b in qd.ndrange(passive_info.dofs_idx.shape[0], dyn_state.dofs.pos.shape[1]):
        i_d = passive_info.dofs_idx[i_d_]
        position = dyn_state.dofs.pos[i_d, i_b]
        velocity = dyn_state.dofs.vel[i_d, i_b]
        delta = position - passive_info.q_neutral[i_d_, i_b]
        stiffness = passive_info.k_neutral[i_d_, i_b]
        damping = passive_info.damping[i_d_, i_b]
        friction = passive_info.friction[i_d_, i_b]
        velocity_normalized = velocity / friction_vel_eps
        tanh_velocity = qd.tanh(velocity_normalized)
        passive_force = -stiffness * delta - damping * velocity - friction * tanh_velocity
        energy = 0.5 * stiffness * delta**2
        limit_force = gs.qd_float(0.0)
        if qd.static(has_coupled_limits):
            limit_force = passive_state.coupled_force[i_d_, i_b]
        exponent = passive_info.soft_exponent[i_d_, i_b]
        stiffness_limit = passive_info.k_soft[i_d_, i_b]
        lower_excess = qd.max(passive_info.soft_lo[i_d_, i_b] - position, 0.0)
        upper_excess = qd.max(position - passive_info.soft_hi[i_d_, i_b], 0.0)
        limit_force += stiffness_limit * (lower_excess**exponent - upper_excess**exponent)
        energy += (
            stiffness_limit * (lower_excess ** (exponent + 1.0) + upper_excess ** (exponent + 1.0)) / (exponent + 1.0)
        )
        force_limit = passive_info.force_limit[i_d_, i_b]
        applied_force = qd.math.clamp(passive_force + limit_force, -force_limit, force_limit)
        passive_state.passive_force[i_d_, i_b] = passive_force
        passive_state.limit_force[i_d_, i_b] = limit_force
        passive_state.applied_force[i_d_, i_b] = applied_force
        passive_state.potential_energy[i_d_, i_b] = energy
        if qd.static(is_apply):
            dyn_state.dofs.qf_passive_model[i_d, i_b] += applied_force
            # The substep integrates the velocity dependence of the applied force implicitly through the slope of its
            # chord between rest and the current velocity: applied(v) = applied(0) - slope * v holds exactly, so the
            # damping and friction can slow a DOF down to rest but never reverse its velocity, whatever the substep and
            # the inertia. The tangent slope vanishes where the friction saturates and leaves that regime explicit.
            # The chord is computed from the unclamped expression where the clamp is inactive at both ends, which is
            # exact near rest, and lies within [0, damping + friction / friction_vel_eps].
            elastic_force = limit_force - stiffness * delta
            slope = gs.qd_float(0.0)
            if qd.abs(elastic_force) < force_limit and qd.abs(passive_force + limit_force) < force_limit:
                tanh_ratio = gs.qd_float(1.0)
                if qd.abs(velocity_normalized) > EPS:
                    tanh_ratio = tanh_velocity / velocity_normalized
                slope = damping + friction * tanh_ratio / friction_vel_eps
            elif qd.abs(velocity) > EPS:
                rest_force = qd.math.clamp(elastic_force, -force_limit, force_limit)
                slope = qd.math.clamp(
                    (rest_force - applied_force) / velocity, 0.0, damping + friction / friction_vel_eps
                )
            dyn_state.dofs.damping_passive_model[i_d, i_b] += slope


class PassiveJointModel:
    """Add elastic, dissipative and soft-limit forces to an entity's scalar joints.

    Register before ``scene.build()``. The model adds forces to native joint springs and damping, alongside actuator
    commands. Actuator effort limits apply to the actuator contribution, and ``force_limit`` bounds the force this
    model applies. Joint coordinates are the values returned by ``get_dofs_position()`` (radians for revolute joints
    and metres for prismatic joints). Free and spherical joints need a rotational potential on their configuration
    manifold and cannot be selected here.

    Damping and friction are integrated implicitly, like native joint damping: every substep adds to the implicit
    mass-matrix diagonal of each DOF ``substep_dt`` times the slope of the chord of the applied force between rest and
    the current velocity. They slow a DOF down to rest without reversing its velocity, whatever its inertia and the
    substep. Under the ``approximate_implicitfast`` integrator ``get_mass_mat()`` includes this term like it includes
    native damping, and under the other integrators the factor from ``get_mass_mat(decompose=True)`` includes it.

    Parameters are zero by default, with infinite soft bounds and force limit, and set after build. Each value can vary
    by environment and survives ``scene.reset()``. Solver checkpoints include these parameters. All force getters
    return owned tensors with shape ``(max(n_envs, 1), n_dofs)`` evaluated at the current state.
    """

    def __init__(
        self,
        entity: "RigidEntity",
        dofs_idx_local=None,
        friction_vel_eps: float = 0.05,
        *,
        _desc: PassiveJointDescription | None = None,
    ):
        if _desc is not None:
            dofs_idx_local, friction_vel_eps = _desc.dofs_idx_local, _desc.friction_vel_eps
        if entity.solver is not entity.scene.rigid_solver:
            gs.raise_exception("Passive joint models require a rigid entity.")
        if entity.is_built or entity.scene.is_built:
            gs.raise_exception("Configure passive joints before scene.build().")
        if entity.sim.requires_grad:
            gs.raise_exception("Passive joint models require forward simulation.")
        if not np.isfinite(friction_vel_eps) or friction_vel_eps <= 0.0:
            gs.raise_exception("friction_vel_eps must be finite and positive.")
        scalar_dofs = np.array(
            [
                joint.dof_start - entity.dof_start
                for joint in entity.joints
                if joint.n_dofs == 1 and joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
            ],
            dtype=gs.np_int,
        )
        dofs_idx = scalar_dofs if dofs_idx_local is None else tensor_to_array(dofs_idx_local)
        if (
            dofs_idx.ndim != 1
            or not dofs_idx.size
            or not np.issubdtype(dofs_idx.dtype, np.integer)
            or np.unique(dofs_idx).size != dofs_idx.size
            or not np.isin(dofs_idx, scalar_dofs).all()
        ):
            gs.raise_exception("Passive joints need distinct entity-local scalar revolute or prismatic DOF indices.")
        self._entity = entity
        self._solver = entity.solver
        self._dofs_idx_local = dofs_idx.astype(gs.np_int)
        self._friction_vel_eps = float(friction_vel_eps)
        self._groups: list[CoupledLimitGroup] = []
        self._info: PassiveJointInfo | None = None
        self._state: PassiveJointState | None = None
        self._desc = PassiveJointDescription(self._dofs_idx_local.copy(), self._friction_vel_eps)
        if _desc is not None:
            for group in _desc.groups:
                self.add_coupled_limit_group(group.dofs_idx_local, group.kind, group.name, group.n_faces or None)
            self._desc = _desc
        else:
            entity.desc.passive_joints.append(self._desc)
        entity._passive_models.append(self)
        self._solver._passive_models.append(self)

    def add_coupled_limit_group(self, dofs_idx_local, kind="ellipsoid", name=None, n_faces=None) -> int:
        """Declare a coupled soft limit before build and return its index.

        Ellipsoids use ``rho = norm((q - center) / radii)`` with sign-selected positive/negative radii and excess
        ``max(rho - 1, 0)``. Polytopes use excess ``max(faces_normal @ (q - center) - faces_offset, 0)`` for each face.
        Every excess contributes potential ``k * excess**(exponent + 1) / (exponent + 1)``. Its negative gradient
        contributes to the model's force before the per-DOF ``force_limit`` clamp. Group values are set after build
        with ``set_coupled_limit_params``. Multiple groups may share members.
        """
        if self._info is not None:
            gs.raise_exception("Declare coupled-limit groups before scene.build().")
        if kind not in ("ellipsoid", "polytope"):
            gs.raise_exception("Coupled-limit kind must be 'ellipsoid' or 'polytope'.")
        members = tensor_to_array(dofs_idx_local)
        if (
            members.ndim != 1
            or members.size < 2
            or not np.issubdtype(members.dtype, np.integer)
            or np.unique(members).size != members.size
            or not np.isin(members, self._dofs_idx_local).all()
        ):
            gs.raise_exception("A coupled limit needs at least two distinct DOFs covered by its passive model.")
        if kind == "polytope":
            if not isinstance(n_faces, (int, np.integer)) or n_faces < 1:
                gs.raise_exception("A polytope needs an integer n_faces >= 1.")
        elif n_faces is not None:
            gs.raise_exception("n_faces applies to polytope groups.")
        name = f"coupled_{len(self._groups)}" if name is None else name
        if not isinstance(name, str) or name in self.coupled_limit_group_names:
            gs.raise_exception("Coupled-limit group names must be distinct strings.")
        slots = np.array([np.flatnonzero(self._dofs_idx_local == i_d)[0] for i_d in members], dtype=gs.np_int)
        self._groups.append(CoupledLimitGroup(name, kind, slots, n_faces or 0))
        self._desc.groups.append(CoupledLimitDescription(name, kind, members.copy(), n_faces or 0))
        return len(self._groups) - 1

    def _build(self):
        shape = (self.n_dofs, self._solver._B)
        self._info = PassiveJointInfo(
            array_class.V(gs.qd_int, (self.n_dofs,)),
            array_class.V_SCALAR_FROM(dtype=gs.qd_float, value=self._friction_vel_eps),
            *(array_class.V(gs.qd_float, shape) for _ in range(9)),
        )
        self._info.dofs_idx.from_numpy(self.dofs_idx)
        self._info.soft_lo.fill(float("-inf"))
        self._info.soft_hi.fill(float("inf"))
        self._info.soft_exponent.fill(1.0)
        self._info.force_limit.fill(float("inf"))
        self._state = PassiveJointState(*(array_class.V(gs.qd_float, shape) for _ in range(5)))
        for group in self._groups:
            shape_members = (len(group.slots), self._solver._B)
            group.info = CoupledLimitInfo(
                array_class.V(gs.qd_int, (len(group.slots),)),
                *(array_class.V(gs.qd_float, shape_members) for _ in range(3)),
                array_class.V(gs.qd_float, (max(group.n_faces, 1), *shape_members)),
                array_class.V(gs.qd_float, (max(group.n_faces, 1), self._solver._B)),
                array_class.V(gs.qd_float, (self._solver._B,)),
                array_class.V(gs.qd_float, (self._solver._B,)),
            )
            group.info.slots.from_numpy(group.slots)
            group.info.radii_pos.fill(1.0)
            group.info.radii_neg.fill(1.0)
            group.info.exponent.fill(1.0)
            group.state = CoupledLimitState(*(array_class.V(gs.qd_float, (self._solver._B,)) for _ in range(2)))

    def _prepare_parameter(self, parameter, value, envs_idx, *, minimum=None, is_infinite=False):
        shape = (len(envs_idx), *parameter.shape[:-1])
        tensor = broadcast_tensor(value, gs.tc_float, shape).contiguous()
        if torch.isnan(tensor).any() or (not is_infinite and not torch.isfinite(tensor).all()):
            gs.raise_exception("Passive joint parameters must be finite (soft bounds may be infinite).")
        if minimum is not None and (tensor < minimum).any():
            gs.raise_exception(f"Passive joint parameter must be >= {minimum}.")
        return tensor

    @staticmethod
    def _write_parameter(parameter, tensor, envs_idx):
        if gs.use_zerocopy:
            view = qd_to_torch(parameter, transpose=True, copy=False)
            assign_indexed_tensor(view, (envs_idx,), tensor)
        else:
            kernel_set_parameter(envs_idx, tensor, parameter)

    def set_params(
        self,
        q_neutral=None,
        k_neutral=None,
        soft_lo=None,
        soft_hi=None,
        k_soft=None,
        soft_exponent=None,
        damping=None,
        friction=None,
        force_limit=None,
        envs_idx=None,
    ):
        """Set parameters broadcast over ``(selected environments, model DOFs)``.

        Neutral torque is ``-k_neutral * (q - q_neutral)``. Soft-limit torque is a restoring power law outside
        ``[soft_lo, soft_hi]``. Damping is ``-damping * velocity`` and friction is
        ``-friction * tanh(velocity / friction_vel_eps)``. The applied force sums these terms and every coupled-limit
        group's force, then clamps the total to ``[-force_limit, force_limit]`` per DOF. The raw terms stay readable
        through their getters. ``get_potential_energy`` reports the unclamped elastic potential. Stiffnesses, damping
        and friction must be nonnegative, ``soft_exponent >= 1`` and ``force_limit >= 0`` (infinite by default). Values of None
        retain the current parameter. Invalid updates leave every parameter unchanged. Coordinates use the units of
        their selected DOF, and forces use N*m or N accordingly.
        """
        if self._info is None:
            gs.raise_exception("Set passive joint parameters after scene.build().")
        envs_idx = self._entity.scene._sanitize_envs_idx(envs_idx)
        updates = []
        for parameter, value, minimum, is_infinite in (
            (self._info.q_neutral, q_neutral, None, False),
            (self._info.k_neutral, k_neutral, 0.0, False),
            (self._info.soft_lo, soft_lo, None, True),
            (self._info.soft_hi, soft_hi, None, True),
            (self._info.k_soft, k_soft, 0.0, False),
            (self._info.soft_exponent, soft_exponent, 1.0, False),
            (self._info.damping, damping, 0.0, False),
            (self._info.friction, friction, 0.0, False),
            (self._info.force_limit, force_limit, 0.0, True),
        ):
            if value is not None:
                updates.append(
                    (
                        parameter,
                        self._prepare_parameter(parameter, value, envs_idx, minimum=minimum, is_infinite=is_infinite),
                    )
                )
        lower = qd_to_torch(self._info.soft_lo, envs_idx, transpose=True)
        upper = qd_to_torch(self._info.soft_hi, envs_idx, transpose=True)
        for parameter, tensor in updates:
            if parameter is self._info.soft_lo:
                lower = tensor
            if parameter is self._info.soft_hi:
                upper = tensor
        if (lower > upper).any() or torch.isposinf(lower).any() or torch.isneginf(upper).any():
            gs.raise_exception("Passive soft bounds must contain a finite coordinate and soft_lo must be <= soft_hi.")
        for parameter, tensor in updates:
            self._write_parameter(parameter, tensor, envs_idx)

    def set_coupled_limit_params(
        self,
        group,
        center=None,
        radii_pos=None,
        radii_neg=None,
        faces_normal=None,
        faces_offset=None,
        k=None,
        exponent=None,
        envs_idx=None,
    ):
        """Set a coupled group's parameters, broadcasting the first dimension over selected environments.

        Centers and radii have trailing shape ``(n_members,)``. Face normals and offsets have trailing shapes
        ``(n_faces, n_members)`` and ``(n_faces,)``. Stiffness k and exponent are scalar per environment. Radii must be
        positive, ``k >= 0`` and ``exponent >= 1``. Face normalization is part of the specified potential: rescaling a
        face changes its stiffness. None retains the value. Invalid updates leave the group unchanged.
        """
        if self._info is None:
            gs.raise_exception("Set coupled-limit parameters after scene.build().")
        if isinstance(group, str):
            if group not in self.coupled_limit_group_names:
                gs.raise_exception(f"Unknown coupled-limit group '{group}'.")
            group = self.coupled_limit_group_names.index(group)
        if not isinstance(group, (int, np.integer)) or not 0 <= group < len(self._groups):
            gs.raise_exception("Coupled-limit group index is out of range.")
        group = self._groups[group]
        if group.kind == "ellipsoid" and (faces_normal is not None or faces_offset is not None):
            gs.raise_exception("Face parameters apply to polytope groups.")
        if group.kind == "polytope" and (radii_pos is not None or radii_neg is not None):
            gs.raise_exception("Radii apply to ellipsoid groups.")
        envs_idx = self._entity.scene._sanitize_envs_idx(envs_idx)
        updates = []
        for parameter, value, minimum in (
            (group.info.center, center, None),
            (group.info.radii_pos, radii_pos, None),
            (group.info.radii_neg, radii_neg, None),
            (group.info.faces_normal, faces_normal, None),
            (group.info.faces_offset, faces_offset, None),
            (group.info.stiffness, k, 0.0),
            (group.info.exponent, exponent, 1.0),
        ):
            if value is not None:
                tensor = self._prepare_parameter(parameter, value, envs_idx, minimum=minimum)
                if parameter is group.info.radii_pos or parameter is group.info.radii_neg:
                    if (tensor <= 0.0).any():
                        gs.raise_exception("Coupled-limit radii must be positive.")
                updates.append((parameter, tensor))
        for parameter, tensor in updates:
            self._write_parameter(parameter, tensor, envs_idx)

    def _evaluate(self, *, is_apply=False):
        if self._info is None:
            gs.raise_exception("Read passive joint forces after scene.build().")
        self._solver.update_forward_pos()
        if gs.device.type == "mps":
            torch.mps.synchronize()
        # Only the group kernels write coupled_force, so a model without groups leaves it at zero and skips reading it.
        if self._groups:
            self._state.coupled_force.fill(0.0)
            for group in self._groups:
                kernel_coupled_limit(
                    self._solver.dyn_state,
                    self._state,
                    group.state,
                    self._info,
                    group.info,
                    is_ellipsoid=group.kind == "ellipsoid",
                )
        kernel_passive_joints(
            self._solver.dyn_state,
            self._state,
            self._solver.rigid_info,
            self._info,
            self._solver.rigid_config,
            is_apply=is_apply,
            has_coupled_limits=bool(self._groups),
        )

    def get_passive_dofs_force(self) -> torch.Tensor:
        """Return current neutral-spring, damping and friction forces, shape ``(B, n_dofs)``."""
        self._evaluate()
        return qd_to_torch(self._state.passive_force, transpose=True, copy=True)

    def get_limit_dofs_force(self) -> torch.Tensor:
        """Return current independent and coupled soft-limit forces, shape ``(B, n_dofs)``."""
        self._evaluate()
        return qd_to_torch(self._state.limit_force, transpose=True, copy=True)

    def get_coupled_limit_dofs_force(self) -> torch.Tensor:
        """Return current coupled soft-limit forces, shape ``(B, n_dofs)``."""
        self._evaluate()
        return qd_to_torch(self._state.coupled_force, transpose=True, copy=True)

    def get_applied_dofs_force(self) -> torch.Tensor:
        """Return the current force the model applies after ``force_limit``, shape ``(B, n_dofs)``."""
        self._evaluate()
        return qd_to_torch(self._state.applied_force, transpose=True, copy=True)

    def get_dofs_forces(self) -> PassiveJointForces:
        """Return the passive, limit and applied forces of the three getters above from one evaluation of the current
        state, cheaper than calling them in turn.
        """
        self._evaluate()
        return PassiveJointForces(
            qd_to_torch(self._state.passive_force, transpose=True, copy=True),
            qd_to_torch(self._state.limit_force, transpose=True, copy=True),
            qd_to_torch(self._state.applied_force, transpose=True, copy=True),
        )

    def get_coupled_limit_violation(self) -> torch.Tensor:
        """Return current excess per group, shape ``(B, n_groups)`` (maximum face excess for polytopes)."""
        self._evaluate()
        if not self._groups:
            return torch.empty((self._solver._B, 0), dtype=gs.tc_float, device=gs.device)
        return torch.stack([qd_to_torch(group.state.violation, transpose=True) for group in self._groups], dim=-1)

    def get_potential_energy(self, dofs_idx=None, envs_idx=None) -> torch.Tensor:
        """Return current elastic energy in Joules, shape ``(B,)``.

        A DOF selection uses solver-global indices and must contain either all or none of each coupled group's
        members. Dissipative forces store no elastic energy.
        """
        self._evaluate()
        if isinstance(dofs_idx, slice):
            dofs_idx = range(*dofs_idx.indices(self._solver.n_dofs))
        selected = (
            np.ones(self.n_dofs, dtype=bool) if dofs_idx is None else np.isin(self.dofs_idx, tensor_to_array(dofs_idx))
        )
        energy = qd_to_torch(self._state.potential_energy, envs_idx, transpose=True)[:, selected].sum(dim=-1)
        for group in self._groups:
            members_selected = selected[group.slots]
            if members_selected.any():
                if not members_selected.all():
                    gs.raise_exception("Coupled-limit energy requires all or none of the group's DOFs.")
                energy += qd_to_torch(group.state.potential_energy, envs_idx, transpose=True)
        return energy

    @property
    def data(self):
        yield from array_class.iter_data(self._info, "info")
        yield from array_class.iter_data(self._state, "state")
        for i_g, group in enumerate(self._groups):
            yield from array_class.iter_data(group.info, f"groups.{i_g}.info")
            yield from array_class.iter_data(group.state, f"groups.{i_g}.state")

    @property
    def n_coupled_limit_groups(self) -> int:
        return len(self._groups)

    @property
    def coupled_limit_group_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self._groups)

    @property
    def n_dofs(self) -> int:
        return len(self._dofs_idx_local)

    @property
    def dofs_idx(self) -> np.ndarray:
        """Owned array of solver-global DOF indices, updated when entities are attached."""
        return self._dofs_idx_local + self._entity.dof_start
