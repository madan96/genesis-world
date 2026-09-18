from collections.abc import Callable
from math import isclose
from typing import TYPE_CHECKING

from pydantic import StrictBool, model_validator
from typing_extensions import Self

import genesis as gs
from genesis.options.hybrid import HybridBindingOptions, HybridCollisionOptions, HybridMassOptions, HybridSkinOptions
from genesis.typing import NonNegativeFloat, ValidFloat

from .base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.hybrid_entity import HybridEntity


class Hybrid(Material["HybridEntity"]):
    """
    The class for hybrid body material (soft skin actuated by inner rigid skeleton).

    Setting ``skin``, ``binding``, ``mass``, ``collision``, or a nonempty ``material_soft_palette`` selects continuous
    skin. Its geometry comes from ``skin`` and its attachment parameters from ``binding``. The per-link settings
    ``damping``, ``thickness``, and ``soft_dv_coef`` must retain their defaults with continuous skin.

    Parameters
    ----------
    material_rigid : Material
        The material of the rigid body.
    material_soft : Material
        The material of the soft body.
    use_default_coupling : bool, optional
        Whether to use default solver coupling. Default is False.
    damping : float, optional
        Damping coefficient for per-link attachment. Continuous skin uses ``binding.damping_ratio``. Default is 0.0.
    thickness : float, optional
        Thickness for per-link skin construction. Continuous skin uses the geometry selected by ``skin``. Default is 0.05.
    soft_dv_coef : float, optional
        Velocity correction coefficient for per-link attachment. Continuous skin uses ``binding.timeconst``.
        Default is 0.01.
    func_instantiate_rigid_from_soft : callable, optional
        The function to instantiate rigid body from the geometry of soft body. Default is None.
    func_instantiate_soft_from_rigid : callable, optional
        The function to instantiate soft body from the geometry of rigid body. Default is None.
    func_instantiate_rigid_soft_association : callable, optional
        The function that determines the association of the rigid and the soft body. Default is None.
    """

    material_rigid: Material = ...
    material_soft: Material = ...
    material_soft_palette: tuple[Material, ...] = ()
    use_default_coupling: StrictBool = False
    damping: NonNegativeFloat = 0.0
    thickness: ValidFloat = 0.05
    soft_dv_coef: ValidFloat = 0.01
    skin: HybridSkinOptions | None = None
    binding: HybridBindingOptions | None = None
    mass: HybridMassOptions | None = None
    collision: HybridCollisionOptions | None = None
    func_instantiate_rigid_from_soft: Callable | None = None
    func_instantiate_soft_from_rigid: Callable | None = None
    func_instantiate_rigid_soft_association: Callable | None = None

    @model_validator(mode="after")
    def _validate_attachment_options(self) -> Self:
        if self.is_explicit:
            for name, value in (
                ("damping", self.damping),
                ("thickness", self.thickness),
                ("soft_dv_coef", self.soft_dv_coef),
            ):
                if not isclose(value, type(self).model_fields[name].default, rel_tol=0.0, abs_tol=0.0):
                    gs.raise_exception(
                        f"Hybrid {name} configures per-link attachment and must retain its default with continuous "
                        "skin. Configure continuous skin with the skin and binding options."
                    )
        return self

    @property
    def is_explicit(self) -> bool:
        """Whether typed skin, binding, mass or collision options select the continuous skin path."""
        return bool(self.material_soft_palette) or any(
            option is not None for option in (self.skin, self.binding, self.mass, self.collision)
        )
