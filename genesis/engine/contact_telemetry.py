"""Device accumulators for the external MPM-to-rigid momentum exchange."""

import quadrants as qd
import torch

import genesis as gs
from genesis.engine.states.contacts import MPMContactLinks, MPMContactRows, MPMContactState, restore_contact_field
from genesis.utils.misc import qd_to_torch


class ContactTelemetry:
    """Bounded contribution storage with unbounded per-link reductions.

    The coupler records only external collision impulses, after resolving its actual
    velocity change. Angular impulses use the fixed world origin, so a moving link
    reference can be selected later without changing the accumulated moment.
    """

    def __init__(self, n_links: int, n_envs: int, capacity: int) -> None:
        if capacity < 1:
            gs.raise_exception("Contact telemetry capacity must be positive.")
        self.capacity = capacity
        self.n_links = n_links
        self.n_envs = max(n_envs, 1)
        self.rows = qd.types.struct(
            entity=gs.qd_int,
            region=gs.qd_int,
            geom=gs.qd_int,
            link=gs.qd_int,
            substep=gs.qd_int,
            position=gs.qd_vec3,
            normal=gs.qd_vec3,
            impulse=gs.qd_vec3,
            relative_velocity=gs.qd_vec3,
            area_time=gs.qd_float,
        ).field(shape=(capacity, self.n_envs), layout=qd.Layout.SOA)
        self.attempted_count = qd.field(gs.qd_int, shape=self.n_envs)
        self.interval = qd.field(gs.qd_float, shape=self.n_envs)
        self.links = qd.types.struct(
            impulse=gs.qd_vec3,
            angular_impulse=gs.qd_vec3,
            normal_impulse=gs.qd_float,
            position_weighted=gs.qd_vec3,
            area_time=gs.qd_float,
            peak_normal_impulse=gs.qd_float,
        ).field(shape=(max(n_links, 1), self.n_envs), layout=qd.Layout.SOA)

    @property
    def row_fields(self) -> tuple[qd.Field, ...]:
        return (
            self.rows.entity,
            self.rows.region,
            self.rows.geom,
            self.rows.link,
            self.rows.substep,
            self.rows.position,
            self.rows.normal,
            self.rows.impulse,
            self.rows.relative_velocity,
            self.rows.area_time,
        )

    @property
    def link_fields(self) -> tuple[qd.Field, ...]:
        return (
            self.links.impulse,
            self.links.angular_impulse,
            self.links.normal_impulse,
            self.links.position_weighted,
            self.links.area_time,
            self.links.peak_normal_impulse,
        )

    def get_state(self) -> MPMContactState:
        """Capture owned observations, compacting rows to the largest retained count across environments."""
        attempted = qd_to_torch(self.attempted_count, transpose=True, copy=True)
        count = min(attempted.max().item(), self.capacity)
        return MPMContactState(
            attempted,
            qd_to_torch(self.interval, transpose=True, copy=True),
            MPMContactRows(
                *(qd_to_torch(field, col_mask=slice(count), transpose=True).clone() for field in self.row_fields)
            ),
            MPMContactLinks(*(qd_to_torch(field, transpose=True, copy=True) for field in self.link_fields)),
        )

    def set_state(self, state: MPMContactState, envs_idx=None) -> None:
        """Restore selected contact observations without changing other environments."""
        for field, value in (
            (self.attempted_count, state.attempted_count),
            (self.interval, state.interval),
            *zip(self.row_fields, state.rows),
            *zip(self.link_fields, state.links),
        ):
            restore_contact_field(field, value, envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Clear selected environments; previously returned snapshots remain owned."""
        if gs.use_zerocopy:
            for field in (self.attempted_count, self.interval):
                view = qd_to_torch(field, transpose=True, copy=False)
                view[envs_idx] = 0
            for field in (
                self.links.impulse,
                self.links.angular_impulse,
                self.links.normal_impulse,
                self.links.position_weighted,
                self.links.area_time,
                self.links.peak_normal_impulse,
            ):
                view = qd_to_torch(field, transpose=True, copy=False)
                view[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_reset_contact_telemetry(envs_idx, self)

    def advance_interval(self, dt: float) -> None:
        """Advance once for every completed MPM collision substep, including empty ones."""
        if gs.use_zerocopy:
            interval = qd_to_torch(self.interval, transpose=True, copy=False)
            interval.add_(dt)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_advance_contact_interval(dt, self)


@qd.kernel
def kernel_reset_contact_telemetry(envs_idx: qd.types.ndarray(), telemetry: qd.template()):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        telemetry.attempted_count[i_b] = 0
        telemetry.interval[i_b] = 0.0
    for i_l, i_b_ in qd.ndrange(telemetry.n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        telemetry.links[i_l, i_b].impulse = 0.0
        telemetry.links[i_l, i_b].angular_impulse = 0.0
        telemetry.links[i_l, i_b].normal_impulse = 0.0
        telemetry.links[i_l, i_b].position_weighted = 0.0
        telemetry.links[i_l, i_b].area_time = 0.0
        telemetry.links[i_l, i_b].peak_normal_impulse = 0.0


@qd.kernel
def kernel_advance_contact_interval(dt: float, telemetry: qd.template()):
    for i_b in range(telemetry.n_envs):
        telemetry.interval[i_b] += dt


@qd.func
def record_contact(
    entity_idx,
    region_idx,
    geom_idx,
    link_idx,
    substep_idx,
    env_idx,
    position,
    outward_normal,
    impulse_on_rigid,
    relative_velocity,
    area,
    dt,
    telemetry: qd.template(),
):
    """Record a resolved impulse on the rigid link; the soft body receives its negative."""
    normal_impulse = qd.max(0.0, -impulse_on_rigid.dot(outward_normal))
    telemetry.links[link_idx, env_idx].impulse += impulse_on_rigid
    telemetry.links[link_idx, env_idx].angular_impulse += position.cross(impulse_on_rigid)
    telemetry.links[link_idx, env_idx].normal_impulse += normal_impulse
    telemetry.links[link_idx, env_idx].position_weighted += position * normal_impulse
    telemetry.links[link_idx, env_idx].area_time += area * dt
    qd.atomic_max(telemetry.links[link_idx, env_idx].peak_normal_impulse, normal_impulse)

    row_idx = qd.atomic_add(telemetry.attempted_count[env_idx], 1)
    if row_idx < telemetry.capacity:
        telemetry.rows[row_idx, env_idx].entity = entity_idx
        telemetry.rows[row_idx, env_idx].region = region_idx
        telemetry.rows[row_idx, env_idx].geom = geom_idx
        telemetry.rows[row_idx, env_idx].link = link_idx
        telemetry.rows[row_idx, env_idx].substep = substep_idx
        telemetry.rows[row_idx, env_idx].position = position
        telemetry.rows[row_idx, env_idx].normal = -outward_normal
        telemetry.rows[row_idx, env_idx].impulse = impulse_on_rigid
        telemetry.rows[row_idx, env_idx].relative_velocity = relative_velocity
        telemetry.rows[row_idx, env_idx].area_time = area * dt
