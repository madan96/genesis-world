from typing import Any, Literal

from pydantic import StrictBool, model_validator
from typing_extensions import Self

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, PositiveInt, UnitVec4FType, Vec3FType

from .options import Options


class HybridSkinOptions(Options):
    """Continuous solid skin and its reference data.

    ``surface`` accepts a mesh path or ``trimesh.Trimesh`` in the entity frame. Its vertices must form a watertight,
    consistently wound solid. ``reference_pos`` and ``reference_quat`` place that mesh within the entity before the
    morph transform. ``from_urdf_geoms`` instead samples the signed-distance union of every collision geometry.

    ``skinning`` accepts an NPZ path or a mapping with ``link_names``, integer ``indices[V, K]`` and nonnegative
    ``weights[V, K]``. Index -1 denotes an unused slot. Closest-triangle barycentric interpolation transfers weights
    to volume particles. ``regions`` contains unique ``region_names`` and integer ``vertex_region[V]``. Its labels
    transfer from the barycentric-dominant corner of the closest triangle.

    ``vertex_material_id[V]`` selects material 0 (``material_soft``) or one of ``material_soft_palette`` (IDs 1 onward).
    ``vertex_surface_contact[V]`` is a boolean mask of surface patches that exchange external contact impulses.
    Both arrays transfer by the same closest-triangle dominant-corner rule. Omitted arrays select material 0 and
    contact on every patch. Distinct contact masks increase simulation memory at semantic-region boundaries.

    Optional ``link_reference_poses`` maps link names to ``(x, y, z, qw, qx, qy, qz)`` in world coordinates. Omitted
    links use the authored initial link frame. Distances and translations are in meters.
    """

    surface: Any = None
    from_urdf_geoms: StrictBool = False
    volume_mode: Literal["solid", "shell"] = "solid"
    shell_thickness: PositiveFloat = 0.02
    skinning: Any = None
    regions: Any = None
    vertex_material_id: Any = None
    vertex_surface_contact: Any = None
    reference_pos: Vec3FType = (0.0, 0.0, 0.0)
    reference_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)
    link_reference_poses: dict[str, Any] | None = None
    sampler: str = "regular"
    urdf_inflate: NonNegativeFloat = 0.02
    voxel_pitch: PositiveFloat | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if (self.surface is None) != self.from_urdf_geoms:
            gs.raise_exception("Hybrid skin requires exactly one source: surface or from_urdf_geoms=True.")
        if self.volume_mode != "solid":
            gs.raise_exception("Hybrid volume sampling supports solid surfaces. Shell volume sampling is unavailable.")
        if self.from_urdf_geoms and any(
            value is not None
            for value in (self.skinning, self.regions, self.vertex_material_id, self.vertex_surface_contact)
        ):
            gs.raise_exception("Generated URDF union vertices require generated bindings and regions.")
        return self


class HybridBindingOptions(Options):
    """Linear-blend skin attachment with a conservative spring and objective damping.

    ``timeconst`` is the spring time constant in seconds, with stiffness ``m / timeconst**2`` and damping
    ``2 * damping_ratio * m / timeconst``. ``max_correction_force`` bounds each particle's correction in newtons by
    scaling its spring and damping parts together, so link reactions still balance while a bounded correction
    departs from the reported spring potential.
    ``phase`` places the correction before particle-to-grid transfer or after grid-to-particle transfer.
    ``nearest_link_blend_radius`` sets the Gaussian distance scale in meters when skinning weights are absent.
    ``max_influences`` fixes binding storage at build time. Links and particles share one substep interval.

    The supported local linear stability range accounts for current rigid mass, rotational inertia and both spring
    and damping lever arms. The range is checked at the simulator's error-check cadence, and a step outside it raises
    an error: increase ``timeconst`` or reduce the substep duration. Large nonlinear deformation can require a smaller
    step even within the local range.
    Spring reactions act at each bound link point; damping reactions act at the current particle point. This preserves
    spring work and total angular momentum when skin blends across independently rotating links.
    """

    method: Literal["linear_blend"] = "linear_blend"
    max_influences: PositiveInt = 4
    timeconst: PositiveFloat = 0.01
    damping_ratio: NonNegativeFloat = 1.0
    max_correction_force: PositiveFloat | None = None
    phase: Literal["before_coupling", "after_coupling"] = "after_coupling"
    nearest_link_blend_radius: PositiveFloat = 0.03


class HybridMassOptions(Options):
    """Mass allocation between skin particles and rigid links.

    ``additive`` retains both authored masses. ``rigid_total`` subtracts the particles' weighted mass, first moment,
    and inertia from their associated links at the reference pose. The remaining link inertias must be positive.
    ``split_total`` uniformly scales rigid density and particle density to ``total_mass`` kilograms, assigning
    ``soft_mass_fraction`` to the particles. Link centers of mass remain fixed under uniform density scaling.
    """

    mode: Literal["additive", "rigid_total", "split_total"] = "additive"
    total_mass: PositiveFloat | None = None
    soft_mass_fraction: PositiveFloat | None = None

    @model_validator(mode="after")
    def _validate_mass(self) -> Self:
        if self.mode == "split_total":
            if self.total_mass is None or self.soft_mass_fraction is None or self.soft_mass_fraction >= 1.0:
                gs.raise_exception("split_total requires total_mass > 0 and 0 < soft_mass_fraction < 1.")
        elif self.total_mass is not None or self.soft_mass_fraction is not None:
            gs.raise_exception("total_mass and soft_mass_fraction require mass mode='split_total'.")
        return self


class HybridCollisionOptions(Options):
    """Choose external contact representations for a covered articulation.

    ``soft`` uses skin contact, ``rigid`` uses skeleton contact, and ``both`` enables both surfaces. ``link_overrides``
    selects individual rigid-link roles. ``region_overrides`` selects which semantic skin regions couple externally
    with rigid geometry. Each region excludes its binding links and their immediate articulation neighbors.
    """

    external_surface: Literal["soft", "rigid", "both"] = "soft"
    link_overrides: dict[str, Literal["soft", "rigid", "both"]] | None = None
    region_overrides: dict[str, Literal["soft", "rigid", "both"]] | None = None
