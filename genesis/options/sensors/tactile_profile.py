"""Serializable calibration and electronics parameters for solver-contact tactile sensors."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, StrictBool

import genesis as gs
from genesis.options.options import Options
from genesis.typing import (
    FArrayType,
    LaxFArrayType,
    LaxPositiveFArrayType,
    NonNegativeFloat,
    PathType,
    PositiveFloat,
    UnitInterval,
    UnitVec3FLaxArrayType,
    ValidFloat,
    Vec2FType,
    Vec3FArrayType,
)

if TYPE_CHECKING:
    from genesis.engine.sensors.tactile_transfer import TactileTransferEvaluator


class TactileTransferCurve(Options):
    """Monotone conversion from nonnegative force magnitude to a sensor reading in newtons.

    ``linear`` uses ``scale * force``. ``polynomial`` uses coefficients of powers one through N and requires a
    finite ``input_max`` defining the calibrated interval. Values above this interval clamp to its endpoint.
    ``lookup`` interpolates nondecreasing samples beginning at (0, 0) and clamps beyond the last sample.
    ``dead_band`` zeros input magnitudes below its threshold. ``saturation`` caps the converted reading.
    The same operation applies independently to normal force and shear magnitude.
    """

    kind: Literal["linear", "polynomial", "lookup"] = "linear"
    scale: PositiveFloat = 1.0
    coeffs: FArrayType | None = None
    input_max: PositiveFloat | None = None
    input_points: FArrayType | None = None
    output_points: FArrayType | None = None
    saturation: PositiveFloat | None = None
    dead_band: NonNegativeFloat = 0.0

    def model_post_init(self, context) -> None:
        if self.kind == "polynomial":
            if self.coeffs is None or self.input_max is None:
                gs.raise_exception("Polynomial tactile curves require coeffs and a calibrated input_max in newtons.")
            derivative = np.arange(1, len(self.coeffs) + 1) * self.coeffs
            extrema = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(derivative))
            interior = extrema.real[(extrema.imag == 0.0) & (extrema.real > 0.0) & (extrema.real < self.input_max)]
            sample = np.concatenate(([0.0, self.input_max], interior))
            if np.any(np.polynomial.polynomial.polyval(sample, derivative) < 0.0):
                gs.raise_exception("Polynomial tactile curves must be nondecreasing throughout [0, input_max].")
        elif self.kind == "lookup":
            if self.input_points is None or self.output_points is None:
                gs.raise_exception("Lookup tactile curves require input_points and output_points.")
            if len(self.input_points) < 2 or len(self.input_points) != len(self.output_points):
                gs.raise_exception(
                    "Lookup tactile curves require matching input/output arrays with at least two samples."
                )
            if self.input_points[0] != 0.0 or self.output_points[0] != 0.0:
                gs.raise_exception("Lookup tactile curves must begin at zero input and zero output.")
            if np.any(np.diff(self.input_points) <= 0.0) or np.any(np.diff(self.output_points) < 0.0):
                gs.raise_exception("Lookup tactile inputs must increase strictly and outputs must be nondecreasing.")

    @property
    def is_identity(self) -> bool:
        return self.kind == "linear" and self.scale == 1.0 and self.saturation is None and self.dead_band == 0.0

    def apply_magnitude(self, magnitude: torch.Tensor, saturation_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Convert nonnegative finite magnitudes on their current device, returning a fresh tensor.

        This convenience read builds the device constants on each call. Repeated
        simulation reads should reuse the evaluator returned by ``build``.
        """
        return self.build(device=magnitude.device, dtype=magnitude.dtype).apply_magnitude(magnitude, saturation_scale)

    def build(self, *, device: torch.device, dtype: torch.dtype) -> "TactileTransferEvaluator":
        """Snapshot the curve and allocate reusable device constants for its evaluator."""
        from genesis.engine.sensors.tactile_transfer import TactileTransferEvaluator

        return TactileTransferEvaluator(self, device=device, dtype=dtype)


class TactileHardwareProfile(Options):
    """Versioned hardware model for the measured branch of a tactile sensor.

    Taxel positions are link-local meters, normals are outward unit vectors, and support radius/area are in m/m^2.
    Scalar sensitivity values broadcast to every taxel. Arrays use the same order as ``taxel_pos``. ``validity``
    permanently masks defective taxels. Geometry may be omitted when the sensor supplies it separately.

    ``gain`` is dimensionless and ``offset`` is a normal-force offset in N. Transfer curves act on normal/shear
    force magnitude. ``couple_saturation`` limits each moment component in N*m. ``resolution``, ``noise``, ``bias``,
    and ``drift`` use N for force and N*m for moment channels, with drift specified per simulation step.

    Delay/jitter/filter and hysteresis time constants are seconds. ``sample_rate=None`` samples every simulation
    step. A finite rate samples at simulation instants reaching its requested sampling times. ``gain_variation``
    is a positive multiplicative per-reset gain range. Dead-taxel probability and stuck-value range apply per reset.
    Hysteresis strength and crosstalk strength are dimensionless. Crosstalk sigma is a distance in meters.

    Temperature coefficients modify gain (1/degree C) and normal offset (N/degree C) relative to ``temperature_c``.
    These parameters describe a simulation model. Calibration provenance belongs in ``notes``.
    """

    schema_version: Literal[1] = 1
    name: str = "unnamed_profile"
    taxel_pos: Vec3FArrayType | None = None
    taxel_normal: UnitVec3FLaxArrayType | None = None
    support_radius: LaxPositiveFArrayType | None = None
    support_area: LaxPositiveFArrayType | None = None
    gain: LaxPositiveFArrayType = 1.0
    offset: LaxFArrayType = 0.0
    validity: Annotated[tuple[StrictBool, ...], Field(min_length=1, strict=False)] | None = None
    normal_curve: TactileTransferCurve = Field(default_factory=TactileTransferCurve)
    shear_curve: TactileTransferCurve = Field(default_factory=TactileTransferCurve)
    couple_saturation: PositiveFloat | None = None
    resolution: NonNegativeFloat = 0.0
    noise: NonNegativeFloat = 0.0
    bias: ValidFloat = 0.0
    drift: NonNegativeFloat = 0.0
    gain_variation: Vec2FType | None = None
    delay: NonNegativeFloat = 0.0
    jitter: NonNegativeFloat = 0.0
    sample_rate: PositiveFloat | None = None
    filter_tau: NonNegativeFloat = 0.0
    hysteresis_strength: NonNegativeFloat = 0.0
    hysteresis_tau: NonNegativeFloat = 0.0
    crosstalk_strength: UnitInterval = 0.0
    crosstalk_sigma: NonNegativeFloat = 0.0
    dead_taxel_probability: UnitInterval = 0.0
    dead_taxel_value_range: Vec2FType = (0.0, 0.0)
    temperature_c: ValidFloat = 25.0
    gain_temp_coeff: ValidFloat = 0.0
    offset_temp_coeff: ValidFloat = 0.0
    notes: str = ""

    def model_post_init(self, context) -> None:
        if self.jitter > self.delay:
            gs.raise_exception("Tactile profile jitter must be no larger than delay.")
        if self.gain_variation is not None and not 0.0 < self.gain_variation[0] <= self.gain_variation[1]:
            gs.raise_exception("Tactile profile gain_variation must satisfy 0 < lower <= upper.")
        if self.dead_taxel_value_range[0] > self.dead_taxel_value_range[1]:
            gs.raise_exception("Tactile profile dead_taxel_value_range must satisfy lower <= upper.")
        if self.hysteresis_strength > 0.0 and self.hysteresis_tau == 0.0:
            gs.raise_exception("Tactile profile hysteresis requires a positive hysteresis_tau.")
        if self.crosstalk_strength > 0.0 and self.crosstalk_sigma == 0.0:
            gs.raise_exception("Tactile profile crosstalk requires a positive crosstalk_sigma.")
        lengths = []
        if self.taxel_pos is not None:
            lengths.append(len(self.taxel_pos))
        if self.validity is not None:
            lengths.append(len(self.validity))
        for values in (self.taxel_normal, self.support_radius, self.support_area, self.gain, self.offset):
            if values is not None and len(values) > 1:
                lengths.append(len(values))
        if lengths and min(lengths) != max(lengths):
            gs.raise_exception("Tactile profile geometry and per-taxel arrays must have matching lengths.")

    @property
    def n_taxels(self) -> int | None:
        if self.taxel_pos is not None:
            return len(self.taxel_pos)
        if self.validity is not None:
            return len(self.validity)
        for values in (self.taxel_normal, self.support_radius, self.support_area, self.gain, self.offset):
            if values is not None and len(values) > 1:
                return len(values)
        return None

    def to_dict(self) -> dict:
        """Return an owned JSON-compatible record including the profile schema version."""
        return self.model_dump(mode="json")

    def save(self, path: PathType) -> None:
        """Write JSON or YAML selected by suffix, raising before opening an unsupported destination."""
        destination = Path(path)
        if destination.suffix.lower() not in (".json", ".yaml", ".yml"):
            gs.raise_exception("Tactile profile destination must end in .json, .yaml, or .yml.")
        data = self.to_dict()
        text = (
            json.dumps(data, indent=2)
            if destination.suffix.lower() == ".json"
            else yaml.safe_dump(data, sort_keys=False)
        )
        destination.write_text(text + "\n")

    @classmethod
    def load(cls, path: PathType) -> "TactileHardwareProfile":
        """Read and validate JSON/YAML, accepting unversioned linear profiles.

        Unversioned nonlinear profiles require explicit migration: review the
        calibrated polynomial input_max and the lookup endpoint-clamping policy,
        then mark the reviewed record with schema_version 1. Unversioned linear
        offsets and temperature-offset coefficients are converted from outward
        normal-vector offsets to positive-compression offsets.
        """
        source = Path(path)
        if source.suffix.lower() not in (".json", ".yaml", ".yml"):
            gs.raise_exception("Tactile profile source must end in .json, .yaml, or .yml.")
        data = (
            json.loads(source.read_text()) if source.suffix.lower() == ".json" else yaml.safe_load(source.read_text())
        )
        if not isinstance(data, dict):
            gs.raise_exception("Tactile profile file must contain a mapping of profile fields.")
        return cls._from_record(data)

    @classmethod
    def _from_record(cls, data: dict) -> "TactileHardwareProfile":
        if "schema_version" not in data:
            for name in ("normal_curve", "shear_curve"):
                curve = data.get(name, {})
                if isinstance(curve, dict):
                    kind = curve.get("kind", "linear")
                elif isinstance(curve, TactileTransferCurve):
                    kind = curve.kind
                else:
                    continue  # Field validation below reports malformed curve records.
                if kind != "linear":
                    gs.raise_exception(
                        "Unversioned nonlinear tactile profiles require migration: review polynomial input_max and "
                        "lookup endpoint clamping, then explicitly set schema_version=1 in the reviewed record."
                    )
            profile = cls(**data)
            converted = profile.to_dict()
            converted["offset"] = [-value for value in profile.offset]
            converted["offset_temp_coeff"] = -profile.offset_temp_coeff
            return cls(**converted)
        return cls(**data)

    @classmethod
    def resolve(cls, value: "TactileHardwareProfile | dict | PathType") -> "TactileHardwareProfile":
        """Return a profile from an existing record, a field mapping, or a JSON/YAML path."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls._from_record(value)
        if isinstance(value, (str, Path)):
            return cls.load(value)
        gs.raise_exception("hardware_profile must be a TactileHardwareProfile, a mapping, or a JSON/YAML path.")
