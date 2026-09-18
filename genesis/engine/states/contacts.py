"""Owned device records for contact observations captured by an explicit scene snapshot."""

from typing import NamedTuple

import quadrants as qd
import torch

import genesis as gs
from genesis.utils.misc import assign_indexed_tensor, indices_to_mask, qd_to_torch


class RigidContactRows(NamedTuple):
    """Logical contact rows with leading environment and compact contact dimensions."""

    geom_a: torch.Tensor
    geom_b: torch.Tensor
    penetration: torch.Tensor
    normal: torch.Tensor
    pos: torch.Tensor
    friction: torch.Tensor
    friction_torsional: torch.Tensor
    friction_rolling: torch.Tensor
    sol_params: torch.Tensor
    force: torch.Tensor
    link_a: torch.Tensor
    link_b: torch.Tensor
    pair_idx: torch.Tensor


class RigidContactState(NamedTuple):
    """Contact rows and the world-frame link quantities their public observations read."""

    count: torch.Tensor
    rows: RigidContactRows
    link_contact_force: torch.Tensor
    link_cd_vel: torch.Tensor
    link_cd_ang: torch.Tensor
    link_root_com: torch.Tensor


class MPMContactRows(NamedTuple):
    """Retained external collision contributions in their original append order."""

    entity: torch.Tensor
    region: torch.Tensor
    geom: torch.Tensor
    link: torch.Tensor
    substep: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    impulse: torch.Tensor
    relative_velocity: torch.Tensor
    area_time: torch.Tensor


class MPMContactLinks(NamedTuple):
    """Unbounded per-link reductions, including contributions absent from the bounded rows."""

    impulse: torch.Tensor
    angular_impulse: torch.Tensor
    normal_impulse: torch.Tensor
    position_weighted: torch.Tensor
    area_time: torch.Tensor
    peak_normal_impulse: torch.Tensor


class MPMContactState(NamedTuple):
    """Owned MPM observations with attempted counts preserving overflow and intervals in seconds."""

    attempted_count: torch.Tensor
    interval: torch.Tensor
    rows: MPMContactRows
    links: MPMContactLinks


class HybridAttachmentState(NamedTuple):
    """Owned per-environment attachment diagnostics, link impulses and elapsed interval in seconds."""

    link_impulse: torch.Tensor
    link_angular_impulse: torch.Tensor
    position_error_max: torch.Tensor
    particle_impulse: torch.Tensor
    particle_angular_impulse: torch.Tensor
    damping_work: torch.Tensor
    spring_potential: torch.Tensor
    stability_index: torch.Tensor
    interval: torch.Tensor


class ContactObservationState(NamedTuple):
    """Contact and attachment observations captured alongside solver and sensor state."""

    rigid: RigidContactState | None
    mpm: MPMContactState | None
    attachments: tuple[HybridAttachmentState, ...]


def restore_contact_field(field: qd.Tensor | qd.Field, value: torch.Tensor, envs_idx=None) -> None:
    """Write owned batch-first observation values into selected environments, including compact row buffers.

    Fields with a row dimension receive only the captured prefix. Counts bound its validity. On Metal the caller
    synchronizes once after all writes, before any kernel consumes them.
    """
    target = qd_to_torch(field, transpose=True, copy=not gs.use_zerocopy)
    mask = indices_to_mask(envs_idx, slice(value.shape[1]) if len(field.shape) > 1 else None)
    assign_indexed_tensor(target, mask, value[mask])
    if not gs.use_zerocopy:
        if len(field.shape) > 1:
            target = target.movedim(0, len(field.shape) - 1)
        field.from_torch(target.contiguous())
