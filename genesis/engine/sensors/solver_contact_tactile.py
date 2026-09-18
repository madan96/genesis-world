"""Conservative taxel transport of solver contacts, with measured-only hardware response."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.contact_contributions import SOURCE_HYBRID, SOURCE_RIGID, ContactContributions
from genesis.options.sensors import SolverContactTaxel as SolverContactTaxelOptions
from genesis.options.sensors.tactile_profile import TactileHardwareProfile
from genesis.utils.misc import qd_to_torch

from .base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    SharedSensorContext,
    SimpleSensor,
    SimpleSensorMetadata,
)
from .tactile_transfer import TactileTransferEvaluator


@qd.kernel
def _kernel_splat_contact_wrenches(
    valid: qd.types.ndarray(),
    position: qd.types.ndarray(),
    force: qd.types.ndarray(),
    sign: qd.types.ndarray(),
    taxels_world: qd.types.ndarray(),
    probe_radius: qd.types.ndarray(),
    result: qd.types.ndarray(),
):
    for i_b, i_c in qd.ndrange(valid.shape[0], valid.shape[1]):
        if valid[i_b, i_c]:
            contact_pos = qd.Vector([position[i_b, i_c, i] for i in qd.static(range(3))])
            contact_force = sign[i_b, i_c] * qd.Vector([force[i_b, i_c, i] for i in qd.static(range(3))])
            weight_sum = gs.qd_float(0.0)
            nearest = -1
            nearest_distance = gs.qd_float(0.0)
            for i_p in range(probe_radius.shape[0]):
                radius = probe_radius[i_p]
                if radius > 0.0:
                    offset = contact_pos - qd.Vector([taxels_world[i_b, 0, i_p, i] for i in qd.static(range(3))])
                    probe_normal = qd.Vector([taxels_world[i_b, 1, i_p, i] for i in qd.static(range(3))])
                    distance_sq = offset.norm_sqr()
                    # The support is a disc on the taxel plane, so a hybrid grid node below the surface keeps the
                    # lateral weights of the surface point above it.
                    plane_distance_sq = distance_sq - offset.dot(probe_normal) ** 2
                    weight_sum += qd.max(0.0, 1.0 - plane_distance_sq / (radius * radius)) ** 2
                    if nearest < 0 or distance_sq < nearest_distance:
                        nearest = i_p
                        nearest_distance = distance_sq
            for i_p in range(probe_radius.shape[0]):
                radius = probe_radius[i_p]
                if radius > 0.0:
                    offset = contact_pos - qd.Vector([taxels_world[i_b, 0, i_p, i] for i in qd.static(range(3))])
                    weight = gs.qd_float(0.0)
                    if weight_sum > 0.0:
                        probe_normal = qd.Vector([taxels_world[i_b, 1, i_p, i] for i in qd.static(range(3))])
                        plane_distance_sq = offset.norm_sqr() - offset.dot(probe_normal) ** 2
                        weight = qd.max(0.0, 1.0 - plane_distance_sq / (radius * radius)) ** 2 / weight_sum
                    elif i_p == nearest:
                        weight = 1.0
                    # A contact lies inside few supports, so most taxels take a zero share and skip the atomics.
                    if weight > 0.0:
                        force_share = weight * contact_force
                        couple_share = offset.cross(force_share)
                        i_s = i_c % result.shape[1]
                        for i in qd.static(range(3)):
                            qd.atomic_add(result[i_b, i_s, 0, i_p, i], force_share[i])
                            qd.atomic_add(result[i_b, i_s, 1, i_p, i], couple_share[i])


class SolverContactTaxelReturnType(NamedTuple):
    """Owned force [N] and free couple [N*m] at each taxel center.

    Shape is ([B,] [history,] *probe_layout, 3). The configured link/world frame applies to both fields.
    A missing contact contribution makes all channels in the corresponding sample NaN. Hardware effects may
    alter the measured wrench; the ground-truth wrench closes to the selected complete contact contributions.
    """

    force: torch.Tensor
    couple: torch.Tensor


class SolverContactContext(SharedSensorContext):
    def __init__(self, sim):
        super().__init__(sim)
        self.sources: set[str] = set()
        self.contacts: ContactContributions | None = None

    def activate(self) -> None:
        self._active = True

    def update(self) -> None:
        if self.is_active:
            source = "all" if len(self.sources) == 2 else next(iter(self.sources))
            self.contacts = self._sim.scene.get_contact_contributions(source=source)

    def reset(self, envs_idx) -> None:
        if self.is_active:
            self.contacts = None

    def destroy(self) -> None:
        self.contacts = None
        self.sources.clear()
        self._active = False


@dataclass
class TactileReadoutState:
    """One sensor's configured geometry and independent per-environment electronics state."""

    options: SolverContactTaxelOptions
    profile: TactileHardwareProfile
    normal_curve: TactileTransferEvaluator
    shear_curve: TactileTransferEvaluator
    cache_slice: slice
    link_idx: int
    owner_entities: torch.Tensor
    filter_entities: torch.Tensor | None
    filter_links: torch.Tensor | None
    filter_regions: torch.Tensor | None
    positions: torch.Tensor
    normals: torch.Tensor
    radii: torch.Tensor
    is_inactive: torch.Tensor
    has_inactive_taxels: bool
    areas: torch.Tensor
    gains: torch.Tensor
    nominal_gains: torch.Tensor
    offset: torch.Tensor
    is_invalid: torch.Tensor
    has_invalid_taxels: bool
    dead: torch.Tensor
    dead_value: torch.Tensor
    temperature: torch.Tensor
    saturation_scale: torch.Tensor
    hysteresis_scale: torch.Tensor
    crosstalk: torch.Tensor | None
    # Whether gain, offset, temperature, transfer curves or couple saturation can change the measured wrench. Set at
    # build and latched True by the gain setter or a restored state, following the `has_any_*` flags of SimpleSensor.
    has_hardware_response: bool
    raw: torch.Tensor
    partial_wrenches: torch.Tensor
    # Taxel centers ([:, 0]) and outward normals ([:, 1]) in the link frame, which change only with the sensor offsets.
    taxels_link: torch.Tensor
    # The same taxel centers and normals in the world frame, which follow the link pose every step.
    taxels_world: torch.Tensor
    link_quat: torch.Tensor
    is_complete: torch.Tensor
    filter_state: torch.Tensor
    hysteresis_state: torch.Tensor
    previous_input: torch.Tensor
    held_sample: torch.Tensor
    tick: torch.Tensor
    sample_index: torch.Tensor
    dt: float

    @property
    def positions_link(self) -> torch.Tensor:
        return self.taxels_link[:, 0]

    @property
    def normals_link(self) -> torch.Tensor:
        return self.taxels_link[:, 1]

    def has_link_frame_response(self, has_any_electronics: bool) -> bool:
        """Whether the measured branch applies an effect defined on the link axes of the taxels.

        ``has_any_electronics`` tells whether the class applies drift, noise, bias or quantization to its channels.
        """
        return (
            self.has_hardware_response
            or self.crosstalk is not None
            or self.options.hysteresis_strength > 0.0
            or self.profile.filter_tau > 0.0
            or self.options.dead_taxel_probability > 0.0
            or has_any_electronics
        )

    def update_offsets(self, offset_pos, offset_quat) -> None:
        self.taxels_link.copy_(
            gu.transform_by_quat(torch.stack((self.positions, self.normals))[None], offset_quat[:, None, None])
        )
        self.taxels_link[:, 0] += offset_pos[:, None]

    def update_pose(self, link_pos, link_quat) -> None:
        self.link_quat.copy_(link_quat)
        self.taxels_world.copy_(gu.transform_by_quat(self.taxels_link, link_quat[:, None, None]))
        self.taxels_world[:, 0] += link_pos[:, None]

    def reset(self, envs_idx) -> None:
        for values in (self.filter_state, self.hysteresis_state, self.previous_input, self.held_sample, self.tick):
            values[envs_idx] = 0
        self.sample_index[envs_idx] = -1
        self.is_complete[envs_idx] = True
        if self.options.probe_gain_resample_range is not None:
            low, high = self.options.probe_gain_resample_range
            shape = self.gains[envs_idx].shape
            factor = low + (high - low) * torch.rand(shape, dtype=gs.tc_float, device=gs.device)
            self.gains[envs_idx] = self.nominal_gains * factor
        if self.options.dead_taxel_probability > 0.0:
            shape = self.dead[envs_idx].shape
            self.dead[envs_idx] = torch.rand(shape, device=gs.device) < self.options.dead_taxel_probability
            low, high = self.options.dead_taxel_value_range
            self.dead_value[envs_idx] = low + (high - low) * torch.rand(shape, dtype=gs.tc_float, device=gs.device)


@dataclass
class SolverContactTaxelMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    readouts: list[TactileReadoutState] = field(default_factory=list)

    @property
    def has_any_electronics(self) -> bool:
        return self.has_any_random_walk or self.has_any_noise or self.has_any_bias or self.has_any_resolution


def _entity_family(scene, entity_idx: int) -> set[int]:
    from genesis.engine.entities.hybrid_entity import HybridEntity

    for entity in scene.entities:
        if isinstance(entity, HybridEntity):
            members = {entity.idx, entity.part_rigid.idx, entity.part_soft.idx}
            if entity_idx in members:
                return members
    return {entity_idx}


class SolverContactTaxelSensor(
    RigidSensorMixin[SolverContactTaxelMetadata],
    SimpleSensor[
        SolverContactTaxelOptions, SolverContactContext, SolverContactTaxelMetadata, SolverContactTaxelReturnType
    ],
):
    """Solver-contact tactile sensing with owned reads, batching, history, and selective reset."""

    def __init__(self, options, idx, shared_context, shared_metadata, manager):
        self._layout = np.asarray(options.probe_local_pos).shape[:-1]
        self._n_taxels = int(np.prod(self._layout))
        self._debug_objects: list = []
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    def _get_return_format(self):
        shape = (*self._layout, 3)
        return shape, shape

    @classmethod
    def state_tensors(cls, shared_metadata):
        yield from super().state_tensors(shared_metadata)
        for index, readout in enumerate(shared_metadata.readouts):
            for name, value in (
                ("gains", readout.gains),
                ("dead", readout.dead),
                ("dead_value", readout.dead_value),
                ("temperature", readout.temperature),
                ("saturation_scale", readout.saturation_scale),
                ("hysteresis_scale", readout.hysteresis_scale),
                ("is_complete", readout.is_complete),
                ("filter", readout.filter_state),
                ("hysteresis", readout.hysteresis_state),
                ("previous_input", readout.previous_input),
                ("held_sample", readout.held_sample),
                ("tick", readout.tick),
                ("sample_index", readout.sample_index),
            ):
                yield f"taxel/{index}/{name}", value

    @classmethod
    def state_restored(cls, shared_metadata) -> None:
        super().state_restored(shared_metadata)
        for readout in shared_metadata.readouts:
            readout.has_hardware_response |= bool(((readout.gains - 1.0).abs() > gs.EPS).any())
        cls._update_geometry(shared_metadata, update_offsets=True)

    @classmethod
    def _get_intermediate_dtype(cls):
        return gs.tc_float

    @classmethod
    def _get_cache_dtype(cls):
        return gs.tc_float

    def build(self) -> None:
        super().build()
        self._shared_context.sources.update(self._options.contact_sources)
        self._shared_context.activate()
        B, P = self._manager._sim._B, self._n_taxels
        scene = self._manager._sim.scene
        options = self._options
        profile = options.hardware_profile or TactileHardwareProfile()

        def per_taxel(value):
            return torch.as_tensor(value, dtype=gs.tc_float, device=gs.device).reshape(-1).broadcast_to((P,)).clone()

        def indices(values):
            return None if values is None else torch.tensor(sorted(values), dtype=gs.tc_int, device=gs.device)

        family = _entity_family(scene, options.entity_idx)
        allowed_entities = None
        if options.filter_entities is not None:
            allowed_entities = set()
            for entity_idx in options.filter_entities:
                allowed_entities.update(_entity_family(scene, entity_idx))
        positions = torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device).reshape(P, 3)
        normals = torch.tensor(options.probe_local_normal, dtype=gs.tc_float, device=gs.device).reshape(-1, 3)
        normals = normals.broadcast_to((P, 3)).clone()
        radii = per_taxel(options.probe_radius)
        areas = math.pi * radii.square() if options.support_area is None else per_taxel(options.support_area)
        gains = per_taxel(options.probe_gain)
        crosstalk = None
        if options.crosstalk_strength > 0.0:
            distances = torch.cdist(positions, positions)
            crosstalk = torch.exp(-0.5 * (distances / options.crosstalk_sigma).square())
            active = radii > 0.0
            crosstalk *= active[:, None] & active[None, :]
            crosstalk /= crosstalk.sum(dim=0, keepdim=True).clamp_min(torch.finfo(gs.tc_float).tiny)
            crosstalk *= options.crosstalk_strength
            crosstalk += (1.0 - options.crosstalk_strength) * torch.diag(active.to(gs.tc_float))
        is_invalid = torch.zeros((P, 1), dtype=torch.bool, device=gs.device)
        if profile.validity is not None:
            is_invalid = ~torch.tensor(profile.validity, dtype=torch.bool, device=gs.device)[:, None]
        offset = per_taxel(profile.offset)
        has_hardware_response = (
            not profile.normal_curve.is_identity
            or not profile.shear_curve.is_identity
            or profile.couple_saturation is not None
            or abs(profile.gain_temp_coeff) > gs.EPS
            or abs(profile.offset_temp_coeff) > gs.EPS
            or options.probe_gain_resample_range is not None
            or bool((offset.abs() > gs.EPS).any())
            or bool(((gains - 1.0).abs() > gs.EPS).any())
        )
        readout = TactileReadoutState(
            options=options,
            profile=profile,
            normal_curve=profile.normal_curve.build(device=gs.device, dtype=gs.tc_float),
            shear_curve=profile.shear_curve.build(device=gs.device, dtype=gs.tc_float),
            cache_slice=slice(self._cache_offset, self._cache_offset + self._cache_size),
            link_idx=self._link.idx,
            owner_entities=indices(family),
            filter_entities=indices(allowed_entities),
            filter_links=indices(options.filter_links),
            filter_regions=indices(options.filter_regions),
            positions=positions,
            normals=normals,
            radii=radii,
            is_inactive=(radii <= 0.0)[:, None],
            has_inactive_taxels=bool((radii <= 0.0).any()),
            areas=areas,
            gains=gains.expand(B, P).clone(),
            nominal_gains=gains,
            offset=offset,
            is_invalid=is_invalid,
            has_invalid_taxels=bool(is_invalid.any()),
            dead=torch.zeros((B, P), dtype=torch.bool, device=gs.device),
            dead_value=torch.zeros((B, P), dtype=gs.tc_float, device=gs.device),
            temperature=torch.full((B, 1), profile.temperature_c, dtype=gs.tc_float, device=gs.device),
            saturation_scale=torch.ones((B, 1), dtype=gs.tc_float, device=gs.device),
            hysteresis_scale=torch.ones((B, 1, 1, 1), dtype=gs.tc_float, device=gs.device),
            crosstalk=crosstalk,
            has_hardware_response=has_hardware_response,
            raw=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            # Partial sums limit float32 roundoff and atomic contention when many contacts share one taxel.
            partial_wrenches=torch.zeros((B, 32, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            taxels_link=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            taxels_world=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            link_quat=torch.zeros((B, 4), dtype=gs.tc_float, device=gs.device),
            is_complete=torch.ones(B, dtype=torch.bool, device=gs.device),
            filter_state=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            hysteresis_state=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            previous_input=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            held_sample=torch.zeros((B, 2, P, 3), dtype=gs.tc_float, device=gs.device),
            tick=torch.zeros(B, dtype=torch.int64, device=gs.device),
            sample_index=torch.full((B,), -1, dtype=torch.int64, device=gs.device),
            dt=self._dt,
        )
        self._shared_metadata.readouts.append(readout)
        readout.reset(torch.arange(B, device=gs.device))
        self._update_geometry(self._shared_metadata, update_offsets=True)

    @classmethod
    def _update_geometry(cls, shared_metadata, *, update_offsets: bool = False):
        """Place the taxels at the current link poses, first recomputing their link-frame layout if requested.

        The link-frame layout depends only on the sensor offsets, so it is recomputed at build, after an offset setter
        and after a state restore.
        """
        solver = shared_metadata.solver
        positions = qd_to_torch(solver.dyn_state.links.pos, transpose=True)
        quaternions = qd_to_torch(solver.dyn_state.links.quat, transpose=True)
        for i_s, readout in enumerate(shared_metadata.readouts):
            if update_offsets:
                readout.update_offsets(shared_metadata.offsets_pos[:, i_s], shared_metadata.offsets_quat[:, i_s])
            readout.update_pose(positions[:, readout.link_idx], quaternions[:, readout.link_idx])

    @classmethod
    def reset(cls, shared_metadata, shared_ground_truth_cache, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        for readout in shared_metadata.readouts:
            readout.reset(envs_idx)
        cls._update_geometry(shared_metadata)

    @gs.assert_built
    def set_pos_offset(self, pos_offset, envs_idx=None):
        super().set_pos_offset(pos_offset, envs_idx)
        self._update_geometry(self._shared_metadata, update_offsets=True)

    @gs.assert_built
    def set_quat_offset(self, quat_offset, envs_idx=None):
        super().set_quat_offset(quat_offset, envs_idx)
        self._update_geometry(self._shared_metadata, update_offsets=True)

    @classmethod
    def _update_raw_data(cls, shared_context, shared_metadata, raw_data_T):
        contacts = shared_context.contacts
        if contacts is None:
            gs.raise_exception("SolverContactTaxel requires an updated contact context.")
        cls._update_geometry(shared_metadata)
        B = raw_data_T.shape[1]
        for readout in shared_metadata.readouts:
            options = readout.options
            is_a = contacts.link_a == readout.link_idx
            is_b = contacts.link_b == readout.link_idx
            counterpart = torch.where(is_a, contacts.entity_b, contacts.entity_a)
            counterpart_link = torch.where(is_a, contacts.link_b, contacts.link_a)
            valid = contacts.valid & (is_a | is_b)
            valid &= ~(counterpart[..., None] == readout.owner_entities).any(dim=-1)
            for values, allowed in (
                (counterpart, readout.filter_entities),
                (counterpart_link, readout.filter_links),
                (contacts.region_b, readout.filter_regions),
            ):
                if allowed is not None:
                    valid &= (values[..., None] == allowed).any(dim=-1)
            if "hybrid" not in options.contact_sources:
                valid &= contacts.source == SOURCE_RIGID
                readout.is_complete.fill_(True)
            else:
                readout.is_complete.copy_(contacts.is_complete)
                if "rigid" not in options.contact_sources:
                    valid &= contacts.source == SOURCE_HYBRID
            sign = torch.where(is_a, 1.0, -1.0).to(gs.tc_float)
            readout.partial_wrenches.zero_()
            if contacts.valid.shape[1] > 0:
                _kernel_splat_contact_wrenches(
                    valid,
                    contacts.position,
                    contacts.force,
                    sign,
                    readout.taxels_world,
                    readout.radii,
                    readout.partial_wrenches,
                )
            torch.sum(readout.partial_wrenches, dim=1, out=readout.raw)
            # The raw wrench stays in the output frame. The measured branch rotates into the link frame only when one of
            # its effects needs it (see `has_link_frame_response`).
            if options.frame == "link":
                readout.raw.copy_(gu.inv_transform_by_quat(readout.raw, readout.link_quat[:, None, None, :]))
            raw_data_T[readout.cache_slice] = readout.raw.reshape(B, -1).T

    @classmethod
    def _apply_transform(cls, shared_metadata, data, timeline, *, is_measured):
        if not is_measured:
            return
        B = data.shape[0]
        has_any_electronics = shared_metadata.has_any_electronics
        for readout in shared_metadata.readouts:
            values = data[:, readout.cache_slice].view(B, 2, -1, 3)
            if readout.options.frame == "world" and readout.has_link_frame_response(has_any_electronics):
                values.copy_(gu.inv_transform_by_quat(values, readout.link_quat[:, None, None, :]))
            # An incomplete step holds the filter and hysteresis states. `_post_process` marks its samples NaN.
            complete = readout.is_complete[:, None, None, None]
            if readout.crosstalk is not None:
                forces, couples = values[:, 0], values[:, 1]
                origin_moments = couples + readout.positions_link.cross(forces, dim=-1)
                forces = readout.crosstalk @ forces
                moments = readout.crosstalk @ origin_moments
                values[:, 0] = forces
                values[:, 1] = moments - readout.positions_link.cross(forces, dim=-1)
            if readout.options.hysteresis_strength > 0.0:
                alpha = math.exp(-readout.dt / readout.options.hysteresis_tau)
                hysteresis = readout.hysteresis_state * alpha + values - readout.previous_input
                readout.hysteresis_state.copy_(torch.where(complete, hysteresis, readout.hysteresis_state))
                readout.previous_input.copy_(torch.where(complete, values, readout.previous_input))
                values.add_(readout.options.hysteresis_strength * readout.hysteresis_scale * readout.hysteresis_state)
            if readout.profile.filter_tau > 0.0:
                alpha = -math.expm1(-readout.dt / readout.profile.filter_tau)
                filtered = readout.filter_state.lerp(values, alpha)
                readout.filter_state.copy_(torch.where(complete, filtered, readout.filter_state))
                values.copy_(readout.filter_state)

    @classmethod
    def _apply_hardware_imperfections(cls, shared_metadata, data):
        B = data.shape[0]
        for readout in shared_metadata.readouts:
            if not readout.has_hardware_response:
                continue
            values = data[:, readout.cache_slice].view(B, 2, -1, 3)
            profile = readout.profile
            delta_temperature = readout.temperature - profile.temperature_c
            gain = readout.gains * (1.0 + profile.gain_temp_coeff * delta_temperature)
            values.mul_(gain[:, None, :, None])
            force = values[:, 0]
            normals = readout.normals_link
            normal_component = (force * normals).sum(dim=-1)
            shear = force - normal_component[..., None] * normals
            shear_magnitude = torch.linalg.vector_norm(shear, dim=-1)
            normal_reading = readout.normal_curve.apply_magnitude(normal_component.abs(), readout.saturation_scale)
            shear_reading = readout.shear_curve.apply_magnitude(shear_magnitude, readout.saturation_scale)
            shear_ratio = shear_reading / shear_magnitude.clamp_min(torch.finfo(gs.tc_float).tiny)
            offset = readout.offset + profile.offset_temp_coeff * delta_temperature
            force.copy_(
                (normal_component.sign() * normal_reading - offset)[..., None] * normals
                + shear_ratio[..., None] * shear
            )
            if profile.couple_saturation is not None:
                bound = profile.couple_saturation * readout.saturation_scale[:, :, None]
                values[:, 1] = torch.maximum(torch.minimum(values[:, 1], bound), -bound)
        super()._apply_hardware_imperfections(shared_metadata, data)

    @classmethod
    def _post_process(cls, shared_metadata, tensor, timeline, *, is_measured):
        # The manager keeps `tensor` as the pre-projection intermediate cache captured by scene snapshots.
        result = tensor.clone()
        B = result.shape[0]
        has_any_electronics = shared_metadata.has_any_electronics
        for readout in shared_metadata.readouts:
            values = result[:, readout.cache_slice].view(B, 2, -1, 3)
            if is_measured:
                if readout.has_invalid_taxels:
                    values.masked_fill_(readout.is_invalid, 0.0)
                if readout.options.dead_taxel_probability > 0.0:
                    dead_value = readout.dead_value[:, None, :, None]
                    values.copy_(torch.where(readout.dead[:, None, :, None], dead_value, values))
            if readout.has_inactive_taxels:
                values.masked_fill_(readout.is_inactive, 0.0)
            if (
                is_measured
                and readout.options.frame == "world"
                and readout.has_link_frame_response(has_any_electronics)
            ):
                values.copy_(gu.transform_by_quat(values, readout.link_quat[:, None, None, :]))
            values.masked_fill_(~readout.is_complete[:, None, None, None], float("nan"))
            if is_measured and readout.profile.sample_rate is not None:
                readout.tick.add_(1)
                phase = readout.tick.to(torch.float64) * (readout.dt * readout.profile.sample_rate)
                index = torch.floor(phase + 8 * torch.finfo(torch.float64).eps * phase.abs()).to(torch.int64)
                capture = index > readout.sample_index
                readout.held_sample.copy_(torch.where(capture[:, None, None, None], values, readout.held_sample))
                readout.sample_index.copy_(index)
                values.copy_(readout.held_sample)
        return result

    @gs.assert_built
    def set_probe_gain(self, value, envs_idx=None) -> None:
        """Set positive nominal gain for selected environments/taxels; reset resampling uses the configured profile."""
        readout = self._shared_metadata.readouts[self._idx]
        indices = self._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)
        if not torch.isfinite(values).all() or not (values > 0.0).all():
            gs.raise_exception("SolverContactTaxel gain must be finite and positive.")
        readout.gains[indices] = values.broadcast_to((len(indices), self._n_taxels))
        readout.has_hardware_response |= bool(((values - 1.0).abs() > gs.EPS).any())

    @gs.assert_built
    def set_saturation_scale(self, value, envs_idx=None) -> None:
        """Scale configured force and couple saturation for selected environments by a finite positive factor."""
        readout = self._shared_metadata.readouts[self._idx]
        indices = self._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device).reshape(-1, 1)
        if not torch.isfinite(values).all() or not (values > 0.0).all():
            gs.raise_exception("SolverContactTaxel saturation scale must be finite and positive.")
        readout.saturation_scale[indices] = values.broadcast_to((len(indices), 1))

    @gs.assert_built
    def set_hysteresis_scale(self, value, envs_idx=None) -> None:
        """Scale this sensor's configured hysteresis strength for selected environments."""
        readout = self._shared_metadata.readouts[self._idx]
        indices = self._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device).reshape(-1, 1, 1, 1)
        if not torch.isfinite(values).all() or not (values >= 0.0).all():
            gs.raise_exception("SolverContactTaxel hysteresis scale must be finite and nonnegative.")
        readout.hysteresis_scale[indices] = values.broadcast_to((len(indices), 1, 1, 1))

    @gs.assert_built
    def set_temperature(self, value_c, envs_idx=None) -> None:
        """Set sensor temperature in degrees C; reject temperatures making the configured gain nonpositive."""
        readout = self._shared_metadata.readouts[self._idx]
        indices = self._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(value_c, dtype=gs.tc_float, device=gs.device).reshape(-1, 1)
        gain_scale = 1.0 + readout.profile.gain_temp_coeff * (values - readout.profile.temperature_c)
        if not torch.isfinite(values).all() or not torch.isfinite(gain_scale).all() or not (gain_scale > 0.0).all():
            gs.raise_exception("SolverContactTaxel temperature must be finite and preserve positive gain.")
        readout.temperature[indices] = values.broadcast_to((len(indices), 1))

    @property
    @gs.assert_built
    def taxel_normals(self) -> torch.Tensor:
        """Owned outward normals in the configured output frame, shape ([B,] *probe_layout, 3)."""
        readout = self._shared_metadata.readouts[self._idx]
        taxels = readout.taxels_world if readout.options.frame == "world" else readout.taxels_link
        normals = taxels[:, 1].reshape(-1, *self._layout, 3).clone()
        return normals[0] if self._manager._sim.n_envs == 0 else normals

    @property
    @gs.assert_built
    def taxel_support_areas(self) -> torch.Tensor:
        """Owned support areas in m^2, shape probe_layout; inactive filler areas are zero."""
        readout = self._shared_metadata.readouts[self._idx]
        return torch.where(readout.radii > 0.0, readout.areas, 0.0).reshape(self._layout)

    @property
    @gs.assert_built
    def is_complete(self) -> torch.Tensor:
        """Whether each measured sample contains finite force and couple channels, including delayed/history reads."""
        data = self.read()
        dimensions = tuple(range(-len(self._layout) - 1, 0))
        return torch.isfinite(data.force).all(dim=dimensions) & torch.isfinite(data.couple).all(dim=dimensions)

    @gs.assert_built
    def normal_force(self, *, ground_truth: bool = False) -> torch.Tensor:
        """Read compressive normal force per taxel in N; tensile readings clamp to zero."""
        readout = self._shared_metadata.readouts[self._idx]
        if self._options.frame == "world" and (
            self._options.delay > 0.0 or self._options.history_length > 0 or readout.profile.sample_rate is not None
        ):
            gs.raise_exception(
                "Normal reductions with delay, history, or sample hold require frame='link' "
                "to preserve captured normals."
            )
        data = self.read_ground_truth() if ground_truth else self.read()
        normals = self.taxel_normals
        if self._options.history_length > 0:
            normals = normals.unsqueeze(1 if self._manager._sim.n_envs > 0 else 0)
        return -(data.force * normals).sum(dim=-1).clamp(max=0.0)

    @gs.assert_built
    def pressure(self, *, ground_truth: bool = False) -> torch.Tensor:
        """Read compressive force / declared support area in Pa; inactive fillers return zero."""
        force = self.normal_force(ground_truth=ground_truth)
        area = self.taxel_support_areas
        return torch.where(area > 0.0, force / area.clamp_min(torch.finfo(gs.tc_float).tiny), 0.0)

    def _draw_debug(self, context):
        for item in self._debug_objects:
            context.clear_debug_object(item)
        self._debug_objects.clear()
        readout = self._shared_metadata.readouts[self._idx]
        self._debug_objects.append(
            context.draw_debug_spheres(
                poss=readout.taxels_world[:, 0].reshape(-1, 3),
                radius=self._options.debug_probe_center_radius,
                color=(*self._options.debug_probe_color, 1.0),
            )
        )
