from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import model_validator

import genesis as gs
from genesis.typing import (
    FArrayType,
    FGridType,
    IArrayType,
    LaxPositiveFArrayType,
    NonNegativeFloat,
    PositiveFloat,
    UnitInterval,
    Vec2FType,
)

from .options import ProbesWithNormalSensorOptionsMixin, RigidEntitySensorOptionsMixin, SimpleSensorOptions
from .tactile_profile import TactileHardwareProfile

if TYPE_CHECKING:
    from genesis.engine.sensors.solver_contact_tactile import SolverContactTaxelSensor


class SolverContactTaxel(
    RigidEntitySensorOptionsMixin["SolverContactTaxelSensor"],
    ProbesWithNormalSensorOptionsMixin["SolverContactTaxelSensor"],
    SimpleSensorOptions["SolverContactTaxelSensor"],
):
    """Distribute solved contact forces and moments over a rigid link's taxels.

    Each contribution uses normalized compact weights ``(1 - (distance / radius)^2)^2`` inside the support
    radius. A contact outside every active support goes to the nearest active taxel. Zero-radius entries are
    inactive padding. Force and its moment about the taxel center are transported together, preserving the
    selected contact wrench. ``support_area`` is in m^2 and defaults to pi * radius^2 for pressure reductions.

    Positions/normals use the sensor frame given by the link's ``pos_offset`` and ``euler_offset``. Force/couple
    output uses ``frame``; profile response and stateful filters always operate in the link frame. Rigid input
    uses the last rigid substep and hybrid input uses the current scene-step average; mixed sources retain these
    distinct temporal conventions. Hybrid input requires enabled LegacyCoupler contact telemetry.

    ``filter_entities``, ``filter_links``, and ``filter_regions`` include only matching counterparts (scene entity,
    global rigid link, and participant-local semantic region indices). None includes all. A hybrid composite entity includes
    its rigid and soft children. Contacts within the sensor's own entity are excluded. Region filters select
    hybrid contributions only. An overflow makes ground truth NaN and invalidates measured samples captured at
    that step. No incomplete force sum is exposed as a valid measurement.

    ``hardware_profile`` supplies measured-only electronics. Explicit options override matching profile fields.
    The measured response applies crosstalk, Maxwell hysteresis, low-pass filtering, gain and temperature response,
    force transfer curves and normal offset, then drift, noise, bias, and quantization. Taxel faults and the output
    frame are applied before sample-and-hold and delay. Ground truth contains the transported contact wrench.
    Electronics parameters must be finite; noise, random walk, and resolution must also be nonnegative.
    ``probe_gain_resample_range`` multiplies nominal gains independently per taxel/environment on reset.
    Crosstalk shares signal between arbitrary taxel positions with a column-normalized Gaussian matrix; total
    force and transported moment are conserved before gain, transfer curves, and defective-taxel effects.
    ``probe_radius_noise`` is unsupported: solver-contact support is fixed by the taxel geometry.
    """

    probe_radius: PositiveFloat | FArrayType | FGridType = 0.02
    support_area: LaxPositiveFArrayType | None = None
    contact_sources: tuple[Literal["rigid", "hybrid"], ...] = ("rigid", "hybrid")
    filter_entities: IArrayType | None = None
    filter_links: IArrayType | None = None
    filter_regions: IArrayType | None = None
    frame: Literal["link", "world"] = "link"
    hardware_profile: TactileHardwareProfile | None = None
    probe_gain: LaxPositiveFArrayType = 1.0
    probe_gain_resample_range: Vec2FType | None = None
    dead_taxel_probability: UnitInterval = 0.0
    dead_taxel_value_range: Vec2FType = (0.0, 0.0)
    hysteresis_strength: NonNegativeFloat = 0.0
    hysteresis_tau: NonNegativeFloat = 0.0
    crosstalk_strength: UnitInterval = 0.0
    crosstalk_sigma: NonNegativeFloat = 0.0

    @model_validator(mode="before")
    @classmethod
    def _resolve_profile(cls, data):
        if not isinstance(data, dict) or data.get("hardware_profile") is None:
            return data
        data = data.copy()
        profile = TactileHardwareProfile.resolve(data["hardware_profile"])
        data["hardware_profile"] = profile
        values = profile.to_dict()
        for source, target in (
            ("resolution", "resolution"),
            ("noise", "noise"),
            ("bias", "bias"),
            ("drift", "random_walk"),
            ("delay", "delay"),
            ("jitter", "jitter"),
            ("gain", "probe_gain"),
            ("gain_variation", "probe_gain_resample_range"),
            ("dead_taxel_probability", "dead_taxel_probability"),
            ("dead_taxel_value_range", "dead_taxel_value_range"),
            ("hysteresis_strength", "hysteresis_strength"),
            ("hysteresis_tau", "hysteresis_tau"),
            ("crosstalk_strength", "crosstalk_strength"),
            ("crosstalk_sigma", "crosstalk_sigma"),
            ("support_area", "support_area"),
        ):
            if values[source] is not None:
                data.setdefault(target, values[source])
        return data

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        n_taxels = int(np.prod(np.asarray(self.probe_local_pos).shape[:-1]))
        radii = np.asarray(self.probe_radius)
        if not np.isfinite(radii).all() or np.any(radii < 0.0) or not np.any(radii > 0.0):
            gs.raise_exception("SolverContactTaxel requires finite nonnegative radii and at least one active taxel.")
        for name, values in (
            ("resolution", self.resolution),
            ("noise", self.noise),
            ("random_walk", self.random_walk),
            ("bias", self.bias),
        ):
            values = np.asarray(values)
            if not np.isfinite(values).all() or (name != "bias" and np.any(values < 0.0)):
                requirement = "finite" if name == "bias" else "finite and nonnegative"
                gs.raise_exception(f"SolverContactTaxel {name} must be {requirement}.")
        for name, values in (("probe_gain", self.probe_gain), ("support_area", self.support_area)):
            if values is not None and len(values) not in (1, n_taxels):
                gs.raise_exception(f"SolverContactTaxel {name} must be scalar or match the taxel count.")
        if not self.contact_sources or len(set(self.contact_sources)) != len(self.contact_sources):
            gs.raise_exception("SolverContactTaxel requires one or two distinct contact sources.")
        if self.probe_radius_noise != 0.0:
            gs.raise_exception("SolverContactTaxel uses fixed support radii; probe_radius_noise must be zero.")
        if self.probe_gain_resample_range is not None and not (
            0.0 < self.probe_gain_resample_range[0] <= self.probe_gain_resample_range[1]
        ):
            gs.raise_exception("SolverContactTaxel gain resampling requires 0 < lower <= upper.")
        if self.dead_taxel_value_range[0] > self.dead_taxel_value_range[1]:
            gs.raise_exception("SolverContactTaxel dead-taxel range requires lower <= upper.")
        if self.hysteresis_strength > 0.0 and self.hysteresis_tau == 0.0:
            gs.raise_exception("SolverContactTaxel hysteresis requires positive hysteresis_tau.")
        if self.crosstalk_strength > 0.0 and self.crosstalk_sigma == 0.0:
            gs.raise_exception("SolverContactTaxel crosstalk requires positive crosstalk_sigma.")
        if self.hardware_profile is not None and self.hardware_profile.n_taxels not in (None, n_taxels):
            gs.raise_exception("SolverContactTaxel hardware profile must match the taxel count.")
        for values in (self.filter_entities, self.filter_links, self.filter_regions):
            if values is not None and (not values or min(values) < 0):
                gs.raise_exception("SolverContactTaxel filters must contain nonnegative indices.")

    def validate_scene(self, scene) -> None:
        super().validate_scene(scene)
        if self.hardware_profile is not None:
            rate = self.hardware_profile.sample_rate
            if rate is not None and rate > 1.0 / scene.sim.dt:
                gs.raise_exception("SolverContactTaxel sample rate cannot exceed the simulation rate.")
        if self.filter_entities is not None and max(self.filter_entities) >= len(scene.entities):
            gs.raise_exception("SolverContactTaxel filter_entities contains an unknown scene entity index.")
        if self.filter_links is not None and max(self.filter_links) >= scene.sim.rigid_solver.n_links:
            gs.raise_exception("SolverContactTaxel filter_links contains an unknown rigid link index.")

    @classmethod
    def from_profile(cls, profile, *, entity_idx: int, link_idx_local: int = 0, **overrides) -> "SolverContactTaxel":
        """Construct options from profile geometry and electronics; explicit overrides take precedence."""
        profile = TactileHardwareProfile.resolve(profile)
        if profile.taxel_pos is None:
            gs.raise_exception("SolverContactTaxel.from_profile requires taxel_pos.")
        values = {
            "entity_idx": entity_idx,
            "link_idx_local": link_idx_local,
            "probe_local_pos": profile.taxel_pos,
            "hardware_profile": profile,
        }
        if profile.taxel_normal is not None:
            normals = profile.taxel_normal
            values["probe_local_normal"] = normals[0] if len(normals) == 1 else normals
        if profile.support_radius is not None:
            radii = profile.support_radius
            values["probe_radius"] = radii[0] if len(radii) == 1 else radii
        values.update(overrides)
        return cls(**values)
