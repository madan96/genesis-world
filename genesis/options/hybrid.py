from math import isclose
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
    morph transform. ``from_urdf_geoms`` instead samples the signed-distance union of every collision geometry and
    requires an identity reference transform. Position generated skins through the URDF morph or its collision geometry.

    ``urdf_inflate`` expands the generated union in meters (default 0.02). Larger values cover more of the skeleton and
    increase skin volume. ``voxel_pitch`` sets its sampling resolution in meters. Smaller values improve geometric
    detail at greater preprocessing time and memory cost. If None, it uses the MPM particle size. Both settings apply
    to generated skins. A supplied ``surface`` requires the default ``urdf_inflate`` and ``voxel_pitch=None``.

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
        if self.from_urdf_geoms and any(
            value is not None
            for value in (self.skinning, self.regions, self.vertex_material_id, self.vertex_surface_contact)
        ):
            gs.raise_exception("Generated URDF union vertices require generated bindings and regions.")
        if self.from_urdf_geoms:
            if any(not isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0) for value in self.reference_pos):
                gs.raise_exception("Hybrid skin reference_pos requires a supplied surface. Position the URDF morph.")
            if any(not isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0) for value in self.reference_quat[1:]):
                gs.raise_exception("Hybrid skin reference_quat requires a supplied surface. Rotate the URDF morph.")
        else:
            if not isclose(
                self.urdf_inflate, type(self).model_fields["urdf_inflate"].default, rel_tol=0.0, abs_tol=0.0
            ):
                gs.raise_exception("Hybrid skin urdf_inflate requires from_urdf_geoms=True. Edit the supplied surface.")
            if self.voxel_pitch is not None:
                gs.raise_exception("Hybrid skin voxel_pitch requires from_urdf_geoms=True.")
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
    an error: increase ``timeconst``, lower ``damping_ratio`` or reduce the substep duration. Damping uses most of the
    range whenever ``4 * damping_ratio`` exceeds ``substep_dt / timeconst``. Large nonlinear deformation can require a
    smaller step even within the local range.
    Spring reactions act at each bound link point; damping reactions act at the current particle point. This preserves
    spring work and total angular momentum when skin blends across independently rotating links.
    """

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
    ``split_total`` scales the authored link masses to ``total_mass`` kilograms and the particle density so that the
    particles carry ``soft_mass_fraction`` of it. Every link keeps its scaled authored mass: the skin mass bound to it
    by skinning weight replaces part of its rigid mass, whose inertia scales by the same ratio, so link centers of mass
    remain fixed. Links without skin keep their scaled authored mass, and a link bound to at least that much skin mass
    raises an error. ``rigid_total`` keeps the material densities and the exact link inertia at the cost of moving link
    centers of mass, while ``split_total`` sets the body mass and skin fraction directly.
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
