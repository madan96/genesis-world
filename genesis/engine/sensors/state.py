"""Owned sensor snapshots for in-memory scene state restoration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .base_sensor import Sensor


@dataclass(frozen=True)
class SensorManagerState:
    """Owned sensor buffers with an environment axis first.

    Timeline arrays use [B, age, channel], newest first, independently of the live ring cursor. Restoring selected
    environments preserves every other environment's history. Hardware randomization, drift and captured noisy
    samples are included. Restoring rewinds no random number generator (see 'Scene.reset').
    """

    ground_truth_intermediate: dict[torch.dtype, torch.Tensor]
    measured_intermediate: dict[torch.dtype, torch.Tensor]
    ground_truth_timeline: dict[torch.dtype, torch.Tensor]
    measured_timeline: dict[torch.dtype, torch.Tensor]
    ground_truth_return: dict[type["Sensor"], torch.Tensor]
    measured_return: dict[type["Sensor"], torch.Tensor]
    ground_truth_return_timeline: dict[type["Sensor"], torch.Tensor]
    measured_return_timeline: dict[type["Sensor"], torch.Tensor]
    sensor_tensors: dict[type["Sensor"], dict[str, torch.Tensor]]
