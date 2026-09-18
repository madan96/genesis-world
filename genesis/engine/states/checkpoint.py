"""Array inventories for typed simulator observations and native trajectory frames."""

import dataclasses
from collections.abc import Iterator, Mapping

import numpy as np
import torch

import genesis as gs


def state_tensors(state, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Walk a typed state record using stable field names and class or dtype names for mapping keys."""
    if isinstance(state, torch.Tensor):
        yield prefix, state
    elif dataclasses.is_dataclass(state):
        for name, value in vars(state).items():
            yield from state_tensors(value, f"{prefix}.{name}")
    elif isinstance(state, Mapping):
        for key, value in state.items():
            name = key.__name__ if isinstance(key, type) else str(key)
            yield from state_tensors(value, f"{prefix}.{name}")
    elif isinstance(state, tuple):
        names = state._fields if isinstance(state, tuple) and type(state) is not tuple else range(len(state))
        for name, value in zip(names, state):
            yield from state_tensors(value, f"{prefix}.{name}")
    elif state is not None:
        gs.raise_exception(f"Checkpoint state '{prefix}' has unsupported type {type(state).__name__}.")


def check_arrays(current: Mapping[str, torch.Tensor], saved: Mapping[str, np.ndarray | torch.Tensor]) -> None:
    """Validate every array before a checkpoint restore changes the simulator."""
    mismatch = sorted(set(current) ^ set(saved))
    if mismatch:
        gs.raise_exception(f"Checkpoint observation fields differ from this build: {mismatch}.")
    for name, target in current.items():
        value = torch.as_tensor(saved[name])
        if value.shape != target.shape or value.dtype != target.dtype:
            gs.raise_exception(
                f"Checkpoint '{name}' requires {tuple(target.shape)} {target.dtype}, "
                f"received {tuple(value.shape)} {value.dtype}."
            )


def random_state() -> dict[str, torch.Tensor]:
    """Capture PyTorch's CPU stream and the simulation device stream without advancing either."""
    states = {"cpu": torch.random.get_rng_state()}
    if gs.device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(gs.device)
    elif gs.device.type == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_random_state(arrays: Mapping[str, np.ndarray | torch.Tensor]) -> None:
    """Explicitly rewind process-wide PyTorch streams. Callers must coordinate unrelated random consumers."""
    torch.random.set_rng_state(torch.as_tensor(arrays["rng.cpu"], device="cpu"))
    if gs.device.type == "cuda":
        torch.cuda.set_rng_state(torch.as_tensor(arrays["rng.cuda"], device="cpu"), gs.device)
    elif gs.device.type == "mps":
        torch.mps.set_rng_state(torch.as_tensor(arrays["rng.mps"], device="cpu"))
