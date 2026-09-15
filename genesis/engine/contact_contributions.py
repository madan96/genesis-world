"""Owned, padded contact snapshots for rigid and MPM-to-rigid contacts.

All vectors use the world frame and SI units. A row's force and impulse act on
``entity_a/link_a`` at ``position``; B receives their negatives. The unit normal
points from B into A, so compression has positive ``force.dot(normal)``.

Rigid rows describe the last completed rigid substep: impulse = force * substep_dt.
Hybrid rows accumulate external collision impulses over the current scene step:
force = impulse / elapsed MPM interval, including substeps without contact.
Their relative velocity is sampled before the exchange; rigid relative velocity
is sampled at read time. Internal hybrid attachment forces are excluded.
Combining sources sums these explicitly different temporal samples. It does not
turn rigid snapshots into a scene-step average.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

ContactSource = Literal["all", "rigid", "hybrid"]
SOURCE_RIGID = 0
SOURCE_HYBRID = 1


@dataclass(frozen=True)
class ContactContributions:
    """An owned contact snapshot with leading dimensions ``[B, capacity]``.

    B is always max(n_envs, 1), including nonparallel scenes. Invalid rows have
    zero floating values and -1 indices. ``valid`` is authoritative: concatenated
    source blocks need not put all live rows first. ``count`` counts retained
    rows; ``dropped_count`` counts hybrid contributions lost to buffer capacity.
    Filtering preserves this conservative completeness information. Every tensor
    is caller-owned and remains unchanged by later reads, steps, and resets.

    Entity indices are scene indices of the actual rigid/MPM participants (the
    children of a composite HybridEntity); link/geometry indices are rigid-solver
    global indices. Hybrid B uses an MPM entity and a participant-local semantic
    region index (Hybrid ``region_names`` order, or zero for standalone MPM); its link and
    geometry are -1. Hybrid substep indices start at zero within each scene step.
    Rigid rows have region_b and substep -1. ``interval`` is in
    seconds; ``area`` is a time-weighted grid footprint in m^2 (zero for rigid
    point contacts), not a unique surface union. Relative velocity is B minus A.
    """

    source: torch.Tensor
    entity_a: torch.Tensor
    link_a: torch.Tensor
    geom_a: torch.Tensor
    entity_b: torch.Tensor
    link_b: torch.Tensor
    geom_b: torch.Tensor
    region_b: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    force: torch.Tensor
    impulse: torch.Tensor
    relative_velocity: torch.Tensor
    area: torch.Tensor
    interval: torch.Tensor
    substep: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    dropped_count: torch.Tensor

    @property
    def is_complete(self) -> torch.Tensor:
        """Boolean [B]; false means row sums cannot reproduce the exact wrench."""
        return self.dropped_count == 0

    @property
    def normal_force(self) -> torch.Tensor:
        """Signed compression force [B, capacity], in newtons."""
        return (self.force * self.normal).sum(dim=-1)

    @property
    def tangential_force(self) -> torch.Tensor:
        """Tangential force on A [B, capacity, 3], in newtons."""
        return self.force - self.normal_force[..., None] * self.normal

    def select(
        self, *, entity_idx: int | None = None, link_idx: int | None = None, region_idx: int | None = None
    ) -> "ContactContributions":
        """Return an owned padded snapshot matching every supplied filter.

        Entity and link filters match either side. Region matches the MPM side.
        No device-to-host synchronization or row compaction is performed.
        """
        valid = self.valid.clone()
        for idx in (entity_idx, link_idx, region_idx):
            if idx is not None and idx < 0:
                gs.raise_exception("Contact filter indices must be nonnegative.")
        if entity_idx is not None:
            valid &= (self.entity_a == entity_idx) | (self.entity_b == entity_idx)
        if link_idx is not None:
            valid &= (self.link_a == link_idx) | (self.link_b == link_idx)
        if region_idx is not None:
            valid &= self.region_b == region_idx
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["count"] = valid.sum(dim=1, dtype=gs.tc_int)
        data["dropped_count"] = self.dropped_count.clone()
        return _masked_contributions(data, valid)


@dataclass(frozen=True)
class ContactLinkWrenches:
    """Exact owned external contact reductions over rigid links, including overflow.

    Vector fields have shape [B, n_links, 3]; scalar link fields [B, n_links].
    Force is in N, torque in N*m about ``reference`` (world coordinates, m).
    Impulse/angular_impulse use N*s and N*m*s about the same reference.
    ``rigid_interval`` and ``hybrid_interval`` are [B] seconds and retain the
    distinct temporal windows described in this module. There is no single
    interval relating a mixed-source force to its summed impulse.

    Center of pressure uses positive normal-force weights and is zero where
    ``has_pressure`` is false. Area is the time-weighted grid footprint summed
    over contributions. Peak normal force is the largest single contribution's
    impulse divided by its source interval, not a peak instantaneous load.
    """

    force: torch.Tensor
    torque: torch.Tensor
    impulse: torch.Tensor
    angular_impulse: torch.Tensor
    normal_force: torch.Tensor
    center_of_pressure: torch.Tensor
    area: torch.Tensor
    peak_normal_force: torch.Tensor
    reference: torch.Tensor
    rigid_interval: torch.Tensor
    hybrid_interval: torch.Tensor

    @property
    def has_pressure(self) -> torch.Tensor:
        return self.normal_force > 0


def _masked_contributions(data: dict[str, torch.Tensor], valid: torch.Tensor) -> ContactContributions:
    result = {}
    for name, value in data.items():
        if name in ("count", "dropped_count"):
            result[name] = value
        elif name != "valid":
            mask = valid[..., None] if value.ndim == 3 else valid
            result[name] = torch.where(mask, value, 0.0 if value.is_floating_point() else -1)
    return ContactContributions(**result, valid=valid)


class ContactReader:
    """Scene-owned contact access, with participant mappings resolved at build time."""

    def __init__(self, scene: "Scene") -> None:
        self.scene = scene
        self.link_entities = torch.tensor(
            [link.entity.idx for link in scene.rigid_solver.links], dtype=gs.tc_int, device=gs.device
        )

    def _telemetry(self, source: ContactSource):
        from genesis.engine.couplers.legacy_coupler import LegacyCoupler

        if source not in ("all", "rigid", "hybrid"):
            gs.raise_exception(f"Unknown contact source {source!r}; expected 'all', 'rigid', or 'hybrid'.")
        if source == "rigid":
            return None
        coupler = self.scene.sim.coupler
        if isinstance(coupler, LegacyCoupler):
            if coupler._rigid_mpm and coupler.contact_telemetry is None:
                if self.scene.mpm_solver.enable_CPIC:
                    gs.raise_exception("MPM contact telemetry requires CPIC disabled.")
                gs.raise_exception("MPM contact telemetry is disabled; enable it in LegacyCouplerOptions.")
            return coupler.contact_telemetry
        if self.scene.rigid_solver.is_active and self.scene.mpm_solver.is_active:
            gs.raise_exception("MPM contact telemetry requires the LegacyCoupler.")
        return None

    def _empty(self) -> ContactContributions:
        shape = (self.scene.sim._B, 0)
        vectors = {"position", "normal", "force", "impulse", "relative_velocity"}
        scalars = {"area", "interval"}
        data = {}
        for field in fields(ContactContributions):
            if field.name in ("count", "dropped_count"):
                data[field.name] = torch.zeros(shape[0], dtype=gs.tc_int, device=gs.device)
            elif field.name == "valid":
                data[field.name] = torch.zeros(shape, dtype=torch.bool, device=gs.device)
            else:
                dtype = gs.tc_float if field.name in vectors | scalars else gs.tc_int
                data[field.name] = torch.zeros(
                    shape + ((3,) if field.name in vectors else ()), dtype=dtype, device=gs.device
                )
        return ContactContributions(**data)

    def _rigid(self) -> ContactContributions:
        solver = self.scene.rigid_solver
        if not solver.is_active:
            return self._empty()
        raw = solver.collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True, is_padded=True)
        count = raw["n_contacts"].clone()
        valid = torch.arange(raw["link_a"].shape[1], device=gs.device)[None] < count[:, None]
        link_a = raw["link_a"].long().clamp(0, solver.n_links - 1)
        link_b = raw["link_b"].long().clamp(0, solver.n_links - 1)
        position = raw["position"]
        velocity = qd_to_torch(solver.dyn_state.links.cd_vel, transpose=True)
        angular = qd_to_torch(solver.dyn_state.links.cd_ang, transpose=True)
        root_com = qd_to_torch(solver.dyn_state.links.root_COM, transpose=True)

        def velocity_at(link_idx):
            idx = link_idx[..., None].expand(-1, -1, 3)
            return velocity.gather(1, idx) + angular.gather(1, idx).cross(position - root_com.gather(1, idx), dim=-1)

        force = -raw["force"]
        interval = torch.full_like(force[..., 0], solver.substep_dt)
        data = {
            "source": torch.full_like(raw["link_a"], SOURCE_RIGID),
            "entity_a": self.link_entities[link_a],
            "link_a": raw["link_a"],
            "geom_a": raw["geom_a"],
            "entity_b": self.link_entities[link_b],
            "link_b": raw["link_b"],
            "geom_b": raw["geom_b"],
            "region_b": torch.full_like(raw["link_a"], -1),
            "position": position,
            "normal": raw["normal"],
            "force": force,
            "impulse": force * interval[..., None],
            "relative_velocity": velocity_at(link_b) - velocity_at(link_a),
            "area": torch.zeros_like(interval),
            "interval": interval,
            "substep": torch.full_like(raw["link_a"], -1),
            "count": count,
            "dropped_count": torch.zeros_like(count),
        }
        return _masked_contributions(data, valid)

    def _hybrid(self, telemetry) -> ContactContributions:
        rows = telemetry.rows
        attempted = qd_to_torch(telemetry.attempted_count, transpose=True)
        count = attempted.clamp(max=telemetry.capacity)
        valid = torch.arange(telemetry.capacity, device=gs.device)[None] < count[:, None]
        interval = qd_to_torch(telemetry.interval, transpose=True)[:, None].expand_as(valid)
        inverse_interval = torch.where(interval > 0, interval.reciprocal(), 0.0)
        link = qd_to_torch(rows.link, transpose=True)
        impulse = qd_to_torch(rows.impulse, transpose=True)
        data = {
            "source": torch.full_like(link, SOURCE_HYBRID),
            "entity_a": self.link_entities[link.long().clamp(0, self.link_entities.numel() - 1)],
            "link_a": link,
            "geom_a": qd_to_torch(rows.geom, transpose=True),
            "entity_b": qd_to_torch(rows.entity, transpose=True),
            "link_b": torch.full_like(link, -1),
            "geom_b": torch.full_like(link, -1),
            "region_b": qd_to_torch(rows.region, transpose=True),
            "position": qd_to_torch(rows.position, transpose=True),
            "normal": qd_to_torch(rows.normal, transpose=True),
            "force": impulse * inverse_interval[..., None],
            "impulse": impulse,
            "relative_velocity": qd_to_torch(rows.relative_velocity, transpose=True),
            "area": qd_to_torch(rows.area_time, transpose=True) * inverse_interval,
            "interval": interval,
            "substep": qd_to_torch(rows.substep, transpose=True),
            "count": count,
            "dropped_count": (attempted - telemetry.capacity).clamp(min=0),
        }
        return _masked_contributions(data, valid)

    def contributions(self, source: ContactSource = "all") -> ContactContributions:
        telemetry = self._telemetry(source)
        blocks = []
        if source != "hybrid":
            blocks.append(self._rigid())
        if telemetry is not None:
            blocks.append(self._hybrid(telemetry))
        if not blocks:
            return self._empty()
        if len(blocks) == 1:
            return blocks[0]
        data = {}
        for field in fields(ContactContributions):
            values = [getattr(block, field.name) for block in blocks]
            data[field.name] = sum(values) if field.name in ("count", "dropped_count") else torch.cat(values, dim=1)
        return ContactContributions(**data)

    def wrenches(self, source: ContactSource = "all", *, reference=(0.0, 0.0, 0.0)) -> ContactLinkWrenches:
        telemetry = self._telemetry(source)
        shape = (self.scene.sim._B, self.link_entities.numel())
        vector = torch.zeros((*shape, 3), dtype=gs.tc_float, device=gs.device)
        force, torque, impulse, angular_impulse, position_weighted = (vector.clone() for _ in range(5))
        normal_force, area, peak = (vector[..., 0].clone() for _ in range(3))
        rigid_interval = torch.zeros(shape[0], dtype=gs.tc_float, device=gs.device)
        hybrid_interval = torch.zeros_like(rigid_interval)
        if source != "hybrid" and self.scene.rigid_solver.is_active:
            rows = self._rigid()
            rigid_interval.fill_(self.scene.rigid_solver.substep_dt)
            pressure = rows.normal_force.clamp(min=0)
            for link, sign in ((rows.link_a, 1.0), (rows.link_b, -1.0)):
                idx = link.long().clamp(min=0)
                idx_vector = idx[..., None].expand(-1, -1, 3)
                force.scatter_add_(1, idx_vector, sign * rows.force)
                torque.scatter_add_(1, idx_vector, sign * rows.position.cross(rows.force, dim=-1))
                impulse.scatter_add_(1, idx_vector, sign * rows.impulse)
                angular_impulse.scatter_add_(1, idx_vector, sign * rows.position.cross(rows.impulse, dim=-1))
                normal_force.scatter_add_(1, idx, pressure)
                position_weighted.scatter_add_(1, idx_vector, rows.position * pressure[..., None])
                peak.scatter_reduce_(1, idx, pressure, reduce="amax", include_self=True)
        if telemetry is not None:
            links = telemetry.links
            hybrid_interval = qd_to_torch(telemetry.interval, transpose=True, copy=True)
            inverse = torch.where(hybrid_interval > 0, hybrid_interval.reciprocal(), 0.0)[:, None]
            hybrid_impulse = qd_to_torch(links.impulse, transpose=True)
            hybrid_angular = qd_to_torch(links.angular_impulse, transpose=True)
            impulse += hybrid_impulse
            angular_impulse += hybrid_angular
            force += hybrid_impulse * inverse[..., None]
            torque += hybrid_angular * inverse[..., None]
            normal_force += qd_to_torch(links.normal_impulse, transpose=True) * inverse
            position_weighted += qd_to_torch(links.position_weighted, transpose=True) * inverse[..., None]
            area += qd_to_torch(links.area_time, transpose=True) * inverse
            peak = torch.maximum(peak, qd_to_torch(links.peak_normal_impulse, transpose=True) * inverse)
        reference = torch.broadcast_to(
            torch.as_tensor(reference, dtype=gs.tc_float, device=gs.device), vector.shape
        ).clone()
        torque -= reference.cross(force, dim=-1)
        angular_impulse -= reference.cross(impulse, dim=-1)
        denominator = torch.where(normal_force > 0, normal_force, 1.0)
        return ContactLinkWrenches(
            force,
            torque,
            impulse,
            angular_impulse,
            normal_force,
            position_weighted / denominator[..., None],
            area,
            peak,
            reference,
            rigid_interval,
            hybrid_interval,
        )
