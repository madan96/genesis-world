"""Device-resident evaluation of calibrated tactile magnitude responses."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genesis.options.sensors.tactile_profile import TactileTransferCurve


class TactileTransferEvaluator:
    """Snapshot a curve and allocate its constants once on the selected device.

    Inputs to ``apply_magnitude`` use the selected device and dtype. Editing the
    source options later does not change this evaluator; build another one to
    apply a different calibration.
    """

    def __init__(self, curve: "TactileTransferCurve", *, device: torch.device, dtype: torch.dtype) -> None:
        self._kind = curve.kind
        self._scale = curve.scale
        self._coeffs = curve.coeffs
        self._input_max = curve.input_max
        self._dead_band = curve.dead_band
        self._saturation = curve.saturation
        self._inputs = self._outputs = None
        if curve.kind == "lookup":
            self._inputs = torch.tensor(curve.input_points, device=device, dtype=dtype)
            self._outputs = torch.tensor(curve.output_points, device=device, dtype=dtype)
            self._input_max = curve.input_points[-1]

    def apply_magnitude(self, magnitude: torch.Tensor, saturation_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Return an owned response for nonnegative finite magnitudes in newtons.

        Input validity is the caller's responsibility in the sensor step loop.
        A positive ``saturation_scale`` broadcasts over the input shape and
        scales the calibrated output saturation for domain randomization.
        """
        if self._kind == "linear":
            output = magnitude * self._scale
        elif self._kind == "polynomial":
            bounded = magnitude.clamp(max=self._input_max)
            output = torch.zeros_like(magnitude)
            for coefficient in reversed(self._coeffs):
                output = (output + coefficient) * bounded
        else:
            bounded = magnitude.clamp(max=self._input_max)
            indices = torch.searchsorted(self._inputs, bounded.contiguous()).clamp(1, self._inputs.numel() - 1)
            lower = self._inputs[indices - 1]
            fraction = (bounded - lower) / (self._inputs[indices] - lower)
            output = self._outputs[indices - 1] + fraction * (self._outputs[indices] - self._outputs[indices - 1])
        if self._dead_band > 0.0:
            output = torch.where(magnitude < self._dead_band, 0.0, output)
        if self._saturation is not None:
            if saturation_scale is None:
                output = output.clamp(max=self._saturation)
            else:
                output = torch.minimum(output, self._saturation * saturation_scale)
        return output
