"""Domain-informed initialisation and losses for 1-D Correncoder regression."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


_HANN_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


@dataclass
class MatchedFilter1DReport:
    method: str
    patches_used: int
    kernel_size: int
    filters: int
    shrinkage: float
    ridge: float
    target_offsets: list[int]
    minimum_covariance_eigenvalue: float
    maximum_covariance_eigenvalue: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AffineCalibrationReport:
    gain: float
    offset: float
    train_mse_before: float
    train_mse_after: float
    train_correlation_before: float
    train_correlation_after: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SafeCalibrationHeadReport:
    least_squares_gain: float
    applied_initial_gain: float
    offset: float
    minimum_backward_gain: float
    maximum_absolute_gain: float
    train_mse_before: float
    train_mse_after: float
    train_correlation_before: float
    train_correlation_after: float

    def to_dict(self) -> dict:
        return asdict(self)


class SafeAffineCalibration(nn.Module):
    """Affine output calibration with a straight-through gradient floor.

    The forward pass uses the fitted physical scale exactly. During backward,
    the derivative with respect to the base waveform is prevented from
    vanishing, while the gain parameter still receives its exact gradient.
    """

    def __init__(
        self,
        gain: float,
        offset: float,
        *,
        minimum_absolute_gain: float = 0.05,
        maximum_absolute_gain: float = 10.0,
    ):
        super().__init__()
        if not 0.0 < minimum_absolute_gain < maximum_absolute_gain:
            raise ValueError("Gain bounds must satisfy 0 < minimum < maximum")
        sign = 1.0 if gain >= 0.0 else -1.0
        clipped_gain = sign * min(abs(float(gain)), maximum_absolute_gain)
        self.gain_parameter = nn.Parameter(
            torch.tensor(clipped_gain, dtype=torch.float32)
        )
        self.offset = nn.Parameter(torch.tensor(float(offset), dtype=torch.float32))
        self.register_buffer("gain_sign", torch.tensor(sign, dtype=torch.float32))
        self.minimum_backward_gain = float(minimum_absolute_gain)
        self.maximum_absolute_gain = float(maximum_absolute_gain)

    def gain(self) -> torch.Tensor:
        return self.gain_parameter.clamp(
            min=-self.maximum_absolute_gain, max=self.maximum_absolute_gain
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        gain = self.gain().to(signal.dtype)
        detached_gain = gain.detach()
        sign = torch.where(
            detached_gain.abs() > torch.finfo(signal.dtype).eps,
            detached_gain.sign(),
            self.gain_sign.to(signal.dtype),
        )
        backward_gain = sign * detached_gain.abs().clamp_min(
            self.minimum_backward_gain
        )
        straight_through = (backward_gain - detached_gain) * (
            signal - signal.detach()
        )
        return gain * signal + straight_through + self.offset.to(signal.dtype)


def pearson_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Mean sample-wise ``1 - Pearson correlation`` over the time axis."""

    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (prediction * target).sum(dim=-1)
    # Clamp the energies before sqrt. Clamping only the final product leaves an
    # infinite sqrt derivative at a constant prediction and can create NaN
    # gradients even though the forward denominator is finite.
    prediction_norm = prediction.square().sum(dim=-1).clamp_min(eps).sqrt()
    target_norm = target.square().sum(dim=-1).clamp_min(eps).sqrt()
    correlation = numerator / (prediction_norm * target_norm)
    return (1.0 - correlation.clamp(-1.0, 1.0)).mean()


def lag_tolerant_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_lag_samples: int = 30,
    lag_step: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``1 - max Pearson`` over fixed-overlap temporal shifts.

    Every lag uses the same central prediction support, avoiding the unequal
    overlap and padding bias that arise from rolling full windows.
    """

    if max_lag_samples < 0 or lag_step < 1:
        raise ValueError("max_lag_samples must be non-negative and lag_step positive")
    if max_lag_samples == 0:
        return pearson_correlation_loss(prediction, target, eps=eps)
    length = prediction.shape[-1]
    if prediction.shape != target.shape or 2 * max_lag_samples >= length:
        raise ValueError("Signals must match and retain positive fixed overlap")
    core_length = length - 2 * max_lag_samples
    prediction_core = prediction[..., max_lag_samples : length - max_lag_samples]
    target_windows = target.unfold(-1, core_length, lag_step)
    expected_windows = 2 * max_lag_samples // lag_step + 1
    target_windows = target_windows[..., :expected_windows, :]
    prediction_core = prediction_core.unsqueeze(-2)
    prediction_core = prediction_core - prediction_core.mean(dim=-1, keepdim=True)
    target_windows = target_windows - target_windows.mean(dim=-1, keepdim=True)
    numerator = (prediction_core * target_windows).sum(dim=-1)
    prediction_norm = prediction_core.square().sum(dim=-1).clamp_min(eps).sqrt()
    target_norm = target_windows.square().sum(dim=-1).clamp_min(eps).sqrt()
    correlations = numerator / (prediction_norm * target_norm)
    best_correlation = correlations.clamp(-1.0, 1.0).amax(dim=-1)
    return (1.0 - best_correlation).mean()


def soft_lag_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_lag_samples: int = 30,
    lag_step: int = 3,
    temperature: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Smooth fixed-overlap lag selection using correlation-weighted softmax."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if max_lag_samples < 0 or lag_step < 1:
        raise ValueError("max_lag_samples must be non-negative and lag_step positive")
    if max_lag_samples == 0:
        return pearson_correlation_loss(prediction, target, eps=eps)
    length = prediction.shape[-1]
    if prediction.shape != target.shape or 2 * max_lag_samples >= length:
        raise ValueError("Signals must match and retain positive fixed overlap")
    core_length = length - 2 * max_lag_samples
    prediction_core = prediction[..., max_lag_samples : length - max_lag_samples]
    target_windows = target.unfold(-1, core_length, lag_step)
    expected_windows = 2 * max_lag_samples // lag_step + 1
    target_windows = target_windows[..., :expected_windows, :]
    prediction_core = prediction_core.unsqueeze(-2)
    prediction_core = prediction_core - prediction_core.mean(dim=-1, keepdim=True)
    target_windows = target_windows - target_windows.mean(dim=-1, keepdim=True)
    numerator = (prediction_core * target_windows).sum(dim=-1)
    prediction_norm = prediction_core.square().sum(dim=-1).clamp_min(eps).sqrt()
    target_norm = target_windows.square().sum(dim=-1).clamp_min(eps).sqrt()
    correlations = (numerator / (prediction_norm * target_norm)).clamp(-1.0, 1.0)
    weights = torch.softmax(correlations / temperature, dim=-1)
    soft_correlation = (weights * correlations).sum(dim=-1)
    return (1.0 - soft_correlation).mean()


def spectral_shape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sampling_hz: float = 30.0,
    low_hz: float = 0.08,
    high_hz: float = 0.8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L1 distance between normalised respiratory-band magnitude spectra."""

    length = prediction.shape[-1]
    low_bin = max(0, int(math.ceil(low_hz * length / sampling_hz)))
    high_bin = min(length // 2 + 1, int(math.floor(high_hz * length / sampling_hz)) + 1)
    if low_bin >= high_bin:
        raise ValueError("The selected respiratory band contains no FFT bins")
    window_key = (length, str(prediction.device), prediction.dtype)
    window = _HANN_WINDOW_CACHE.get(window_key)
    if window is None:
        window = torch.hann_window(
            length, periodic=False, device=prediction.device, dtype=prediction.dtype
        )
        _HANN_WINDOW_CACHE[window_key] = window
    prediction = (prediction - prediction.mean(dim=-1, keepdim=True)) * window
    target = (target - target.mean(dim=-1, keepdim=True)) * window
    prediction_spectrum = torch.fft.rfft(prediction, dim=-1).abs()[..., low_bin:high_bin]
    target_spectrum = torch.fft.rfft(target, dim=-1).abs()[..., low_bin:high_bin]
    prediction_spectrum = prediction_spectrum / prediction_spectrum.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    target_spectrum = target_spectrum / target_spectrum.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    return F.l1_loss(prediction_spectrum, target_spectrum)


def correncoder_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    correlation_lambda: float = 0.0,
    spectral_lambda: float = 0.0,
    sampling_hz: float = 30.0,
    correlation_mode: str = "zero_lag",
    max_lag_samples: int = 30,
    lag_step: int = 3,
    lag_temperature: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine waveform MSE with optional shape and frequency objectives."""

    mse = F.mse_loss(prediction, target)
    if correlation_lambda > 0.0:
        if correlation_mode == "zero_lag":
            correlation = pearson_correlation_loss(prediction, target)
        elif correlation_mode == "max_lag":
            correlation = lag_tolerant_pearson_loss(
                prediction,
                target,
                max_lag_samples=max_lag_samples,
                lag_step=lag_step,
            )
        elif correlation_mode == "soft_lag":
            correlation = soft_lag_pearson_loss(
                prediction,
                target,
                max_lag_samples=max_lag_samples,
                lag_step=lag_step,
                temperature=lag_temperature,
            )
        else:
            raise ValueError(f"Unknown correlation_mode: {correlation_mode}")
    else:
        correlation = prediction.new_zeros(())
    spectral = (
        spectral_shape_loss(prediction, target, sampling_hz=sampling_hz)
        if spectral_lambda > 0.0
        else prediction.new_zeros(())
    )
    total = mse + correlation_lambda * correlation + spectral_lambda * spectral
    return total, {
        "mse": mse,
        "correlation_loss": correlation,
        "spectral_loss": spectral,
    }


@torch.no_grad()
def calibrate_regression_output(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int = 512,
    maximum_absolute_gain: float = 10.0,
) -> AffineCalibrationReport:
    """Fit and absorb ``target ~= gain * output + offset`` on training data."""

    if inputs.shape != targets.shape or inputs.ndim != 3:
        raise ValueError("inputs and targets must have matching [N, 1, L] shape")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        outputs.append(model(inputs[start : start + batch_size].to(device)).cpu())
    prediction = torch.cat(outputs).to(torch.float64)
    target = targets.detach().cpu().to(torch.float64)
    prediction_mean = prediction.mean()
    target_mean = target.mean()
    centred_prediction = prediction - prediction_mean
    centred_target = target - target_mean
    variance = centred_prediction.square().mean().clamp_min(1e-12)
    gain = (centred_prediction * centred_target).mean() / variance
    gain = gain.clamp(-maximum_absolute_gain, maximum_absolute_gain)
    offset = target_mean - gain * prediction_mean
    calibrated = gain * prediction + offset

    def correlation(first, second):
        first = first.reshape(-1) - first.mean()
        second = second.reshape(-1) - second.mean()
        return float(
            (first * second).sum()
            / (
                first.square().sum().clamp_min(1e-12).sqrt()
                * second.square().sum().clamp_min(1e-12).sqrt()
            )
        )

    layer = model.decoder1
    layer.weight.mul_(gain.to(layer.weight.dtype).to(layer.weight.device))
    if layer.bias is None:
        raise ValueError("The output layer requires a bias for affine calibration")
    layer.bias.mul_(gain.to(layer.bias.dtype).to(layer.bias.device))
    layer.bias.add_(offset.to(layer.bias.dtype).to(layer.bias.device))
    model.train(was_training)
    return AffineCalibrationReport(
        gain=float(gain),
        offset=float(offset),
        train_mse_before=float(F.mse_loss(prediction, target)),
        train_mse_after=float(F.mse_loss(calibrated, target)),
        train_correlation_before=correlation(prediction, target),
        train_correlation_after=correlation(calibrated, target),
    )


@torch.no_grad()
def fit_safe_calibration_head(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int = 512,
    minimum_absolute_gain: float = 0.05,
    maximum_absolute_gain: float = 10.0,
) -> SafeCalibrationHeadReport:
    """Fit a bounded trainable output head using fold-training tensors only."""

    if inputs.shape != targets.shape or inputs.ndim != 3:
        raise ValueError("inputs and targets must have matching [N, 1, L] shape")
    if getattr(model, "output_calibration", None) is not None:
        raise ValueError("The model already has an output calibration module")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        outputs.append(model(inputs[start : start + batch_size].to(device)).cpu())
    prediction = torch.cat(outputs).to(torch.float64)
    target = targets.detach().cpu().to(torch.float64)
    prediction_mean = prediction.mean()
    target_mean = target.mean()
    centred_prediction = prediction - prediction_mean
    centred_target = target - target_mean
    variance = centred_prediction.square().mean().clamp_min(1e-12)
    least_squares_gain = (centred_prediction * centred_target).mean() / variance
    sign = 1.0 if float(least_squares_gain) >= 0.0 else -1.0
    applied_gain = sign * min(
        abs(float(least_squares_gain)),
        maximum_absolute_gain,
    )
    offset = float(target_mean) - applied_gain * float(prediction_mean)
    calibrated = applied_gain * prediction + offset

    def correlation(first, second):
        first = first.reshape(-1) - first.mean()
        second = second.reshape(-1) - second.mean()
        denominator = (
            first.square().sum().clamp_min(1e-12).sqrt()
            * second.square().sum().clamp_min(1e-12).sqrt()
        )
        return float((first * second).sum() / denominator)

    model.output_calibration = SafeAffineCalibration(
        applied_gain,
        offset,
        minimum_absolute_gain=minimum_absolute_gain,
        maximum_absolute_gain=maximum_absolute_gain,
    ).to(device)
    model.train(was_training)
    return SafeCalibrationHeadReport(
        least_squares_gain=float(least_squares_gain),
        applied_initial_gain=float(model.output_calibration.gain()),
        offset=offset,
        minimum_backward_gain=float(minimum_absolute_gain),
        maximum_absolute_gain=float(maximum_absolute_gain),
        train_mse_before=float(F.mse_loss(prediction, target)),
        train_mse_after=float(F.mse_loss(calibrated, target)),
        train_correlation_before=correlation(prediction, target),
        train_correlation_after=correlation(calibrated, target),
    )


@torch.no_grad()
def initialise_1d_matched_filter(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_patches: int = 50_000,
    shrinkage: float = 0.1,
    seed: int = 0,
) -> MatchedFilter1DReport:
    """Initialise the Correncoder stem from a regularised matched subspace.

    The method estimates Wiener/matched filters from PPG patches to respiration
    samples at several nearby phase offsets.  The left singular vectors of the
    resulting filter bank form eight stable, target-correlated templates.  Only
    the supplied fold-training tensors are used.
    """

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if inputs.ndim != 3 or targets.shape != inputs.shape:
        raise ValueError("inputs and targets must both have shape [N, 1, L]")
    convolution = model.conv1
    kernel_size = int(convolution.kernel_size[0])
    padding = int(convolution.padding[0])
    filters = int(convolution.out_channels)

    inputs = inputs.detach().to(device="cpu", dtype=torch.float64)
    targets = targets.detach().to(device="cpu", dtype=torch.float64)
    patches = F.pad(inputs, (padding, padding)).unfold(-1, kernel_size, 1)
    patches = patches[:, 0]
    output_length = patches.shape[1]

    centres = (
        torch.arange(output_length, dtype=torch.float64)
        - padding
        + (kernel_size - 1) / 2.0
    ).round().to(torch.long)
    radius = max(1, int(round(0.9 * 30.0)))
    offsets = torch.linspace(-radius, radius, filters).round().to(torch.long)
    target_indices = (centres[:, None] + offsets[None, :]).clamp(
        0, targets.shape[-1] - 1
    )
    responses = targets[:, 0, target_indices]

    design = patches.reshape(-1, kernel_size)
    responses = responses.reshape(-1, filters)
    available = design.shape[0]
    if available > max_patches:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(available, generator=generator)[:max_patches]
        design = design[selected]
        responses = responses[selected]

    design = design - design.mean(dim=0, keepdim=True)
    responses = responses - responses.mean(dim=0, keepdim=True)
    denominator = max(design.shape[0] - 1, 1)
    covariance = design.T @ design / denominator
    diagonal = torch.diag(torch.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    diagonal_mean = float(torch.diag(covariance).mean())
    ridge = max(1e-8, 1e-4 * diagonal_mean)
    covariance = covariance + ridge * torch.eye(kernel_size, dtype=covariance.dtype)
    cross_covariance = design.T @ responses / denominator
    wiener_bank = torch.linalg.solve(covariance, cross_covariance)
    basis, _, _ = torch.linalg.svd(wiener_bank, full_matrices=False)
    if basis.shape[1] < filters:
        raise RuntimeError("Matched-filter bank has insufficient rank")
    weights = basis[:, :filters].T
    weights = weights - weights.mean(dim=1, keepdim=True)
    # The published 1D reconstruction uses a fixed synthesis
    # gain after per-filter unit-norm scaling.  This is a scale convention for
    # the encoder/decoder signal path, not the image bank's Kaiming-variance
    # convention; it is applied identically in every matched 1D variant.
    weights = F.normalize(weights, dim=1) * math.sqrt(4.0 / 3.0)
    weights = weights.to(dtype=convolution.weight.dtype).view(filters, 1, kernel_size)
    convolution.weight.copy_(weights.to(convolution.weight.device))
    if convolution.bias is not None:
        convolution.bias.zero_()

    # Initialise the final synthesis atoms from the same physically auditable
    # templates while leaving the intermediate layers under the common random
    # initialisation.
    if model.decoder1.weight.shape == convolution.weight.shape:
        model.decoder1.weight.copy_(convolution.weight)
        if model.decoder1.bias is not None:
            model.decoder1.bias.zero_()

    eigenvalues = torch.linalg.eigvalsh(covariance)
    return MatchedFilter1DReport(
        method="regularised_multiphase_wiener_matched_subspace",
        patches_used=int(design.shape[0]),
        kernel_size=kernel_size,
        filters=filters,
        shrinkage=float(shrinkage),
        ridge=float(ridge),
        target_offsets=[int(value) for value in offsets],
        minimum_covariance_eigenvalue=float(eigenvalues.min()),
        maximum_covariance_eigenvalue=float(eigenvalues.max()),
    )


@torch.no_grad()
def _initialise_deep_matched_layer(
    convolution,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_centre_start: float,
    max_patches: int,
    shrinkage: float,
    covariance_model: str,
    covariance_rank: int,
    seed: int,
) -> dict:
    """Fit one activation-space matched layer with diagonal/low-rank whitening."""

    features = features.detach().cpu().to(torch.float64)
    targets = targets.detach().cpu().to(torch.float64)
    kernel_size = int(convolution.kernel_size[0])
    padding = int(convolution.padding[0])
    filters = int(convolution.out_channels)
    channels = int(convolution.in_channels)
    patches = F.pad(features, (padding, padding)).unfold(-1, kernel_size, 1)
    output_length = patches.shape[-2]
    available = features.shape[0] * output_length
    used = min(available, max_patches)
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(available, generator=generator)[:used]
    examples = torch.div(selected, output_length, rounding_mode="floor")
    positions = selected.remainder(output_length)
    design = patches[examples, :, positions, :].reshape(used, channels * kernel_size)

    radius = max(1, int(round(0.9 * 30.0)))
    offsets = torch.linspace(-radius, radius, filters).round().to(torch.long)
    centres = (
        torch.arange(output_length, dtype=torch.float64) + target_centre_start
    ).round().to(torch.long)
    target_indices = (centres[positions, None] + offsets[None, :]).clamp(
        0, targets.shape[-1] - 1
    )
    responses = targets[examples[:, None], 0, target_indices]

    design = design - design.mean(dim=0, keepdim=True)
    responses = responses - responses.mean(dim=0, keepdim=True)
    denominator = max(used - 1, 1)
    variance = design.square().sum(dim=0) / denominator
    cross_covariance = design.T @ responses / denominator
    target_variance = variance.mean().clamp_min(1e-8)
    ridge = max(1e-8, 1e-4 * float(target_variance))
    effective_rank = None
    explained_covariance_fraction = None
    if covariance_model == "diagonal":
        regularised_variance = (
            (1.0 - shrinkage) * variance + shrinkage * target_variance + ridge
        )
        wiener_bank = cross_covariance / regularised_variance[:, None]
    elif covariance_model == "lowrank":
        dimension = design.shape[1]
        effective_rank = min(max(1, int(covariance_rank)), dimension - 1, used - 1)
        # Randomised PCA avoids forming a dense 600x600 covariance for every
        # LOSO fold while retaining the dominant correlated activation modes.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            _, singular_values, vectors = torch.pca_lowrank(
                design,
                q=effective_rank,
                center=False,
                niter=2,
            )
        eigenvalues = singular_values.square() / denominator
        lowrank_diagonal = (vectors.square() * eigenvalues[None, :]).sum(dim=1)
        residual_variance = (variance - lowrank_diagonal).clamp_min(0.0)
        regularised_variance = (
            (1.0 - shrinkage) * residual_variance
            + shrinkage * target_variance
            + ridge
        )
        lowrank_values = ((1.0 - shrinkage) * eigenvalues).clamp_min(1e-12)
        inverse_diagonal = regularised_variance.reciprocal()
        scaled_vectors = inverse_diagonal[:, None] * vectors
        middle = torch.diag(lowrank_values.reciprocal()) + vectors.T @ scaled_vectors
        diagonal_solution = inverse_diagonal[:, None] * cross_covariance
        wiener_bank = diagonal_solution - scaled_vectors @ torch.linalg.solve(
            middle, vectors.T @ diagonal_solution
        )
        explained_covariance_fraction = float(
            eigenvalues.sum() / variance.sum().clamp_min(1e-12)
        )
    else:
        raise ValueError(f"Unknown covariance_model: {covariance_model}")
    basis, _, _ = torch.linalg.svd(wiener_bank, full_matrices=False)
    weights = basis[:, :filters].T
    weights = weights - weights.mean(dim=1, keepdim=True)
    # Keep the same fixed 1D encoder/decoder gain as the matched stem above.
    weights = F.normalize(weights, dim=1) * math.sqrt(4.0 / 3.0)
    weights = weights.to(convolution.weight.dtype).reshape(
        filters, channels, kernel_size
    )
    convolution.weight.copy_(weights.to(convolution.weight.device))
    if convolution.bias is not None:
        convolution.bias.zero_()
    return {
        "method": f"{covariance_model}_whitened_multiphase_matched_subspace",
        "patches_used": int(used),
        "available_patches": int(available),
        "kernel_size": kernel_size,
        "input_channels": channels,
        "filters": filters,
        "shrinkage_to_isotropic_variance": float(shrinkage),
        "ridge": float(ridge),
        "target_centre_start": float(target_centre_start),
        "target_offsets": [int(value) for value in offsets],
        "minimum_regularised_variance": float(regularised_variance.min()),
        "maximum_regularised_variance": float(regularised_variance.max()),
        "covariance_rank": effective_rank,
        "explained_covariance_fraction": explained_covariance_fraction,
    }


@torch.no_grad()
def initialise_1d_layerwise_matched_filter(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    stem_max_patches: int = 50_000,
    deep_max_patches: int = 10_000,
    shrinkage: float = 0.1,
    depth: int = 3,
    deep_covariance: str = "diagonal",
    covariance_rank: int = 16,
    seed: int = 0,
) -> dict:
    """Initialise one to three encoder layers and mirrored decoder atoms."""

    if depth not in {1, 2, 3}:
        raise ValueError("depth must be 1, 2, or 3")
    if deep_covariance not in {"diagonal", "lowrank"}:
        raise ValueError("deep_covariance must be diagonal or lowrank")

    stem_report = initialise_1d_matched_filter(
        model,
        inputs,
        targets,
        max_patches=stem_max_patches,
        shrinkage=shrinkage,
        seed=seed,
    ).to_dict()
    reports = {"conv1": stem_report}
    if depth == 1:
        return {
            "method": "layerwise_1d_matched_filter",
            "initialised_depth": depth,
            "covariance_model": "full_stem_only",
            "layers": reports,
        }
    cpu_inputs = inputs.detach().cpu().to(torch.float32)
    conv1_weight = model.conv1.weight.detach().cpu()
    hidden1 = F.relu(
        F.conv1d(cpu_inputs, conv1_weight, bias=None, padding=model.conv1.padding[0])
    )
    start1 = (model.conv1.kernel_size[0] - 1) / 2.0 - model.conv1.padding[0]
    start2 = start1 + (model.conv2.kernel_size[0] - 1) / 2.0 - model.conv2.padding[0]
    layer2_report = _initialise_deep_matched_layer(
        model.conv2,
        hidden1,
        targets,
        target_centre_start=start2,
        max_patches=deep_max_patches,
        shrinkage=shrinkage,
        covariance_model=deep_covariance,
        covariance_rank=covariance_rank,
        seed=seed + 1,
    )
    reports["conv2"] = layer2_report
    model.decoder2.weight.copy_(model.conv2.weight)
    if model.decoder2.bias is not None:
        model.decoder2.bias.zero_()
    if depth == 2:
        return {
            "method": "layerwise_1d_matched_filter",
            "initialised_depth": depth,
            "covariance_model": f"full_stem_then_{deep_covariance}_deep_layers",
            "covariance_rank": covariance_rank if deep_covariance == "lowrank" else None,
            "layers": reports,
        }
    conv2_weight = model.conv2.weight.detach().cpu()
    hidden2 = F.relu(
        F.conv1d(hidden1, conv2_weight, bias=None, padding=model.conv2.padding[0])
    )
    start3 = start2 + (model.conv3.kernel_size[0] - 1) / 2.0 - model.conv3.padding[0]
    layer3_report = _initialise_deep_matched_layer(
        model.conv3,
        hidden2,
        targets,
        target_centre_start=start3,
        max_patches=deep_max_patches,
        shrinkage=shrinkage,
        covariance_model=deep_covariance,
        covariance_rank=covariance_rank,
        seed=seed + 2,
    )
    reports["conv3"] = layer3_report
    model.decoder3.weight.copy_(model.conv3.weight)
    if model.decoder3.bias is not None:
        model.decoder3.bias.zero_()
    return {
        "method": "layerwise_1d_matched_filter",
        "initialised_depth": depth,
        "covariance_model": f"full_stem_then_{deep_covariance}_deep_layers",
        "covariance_rank": covariance_rank if deep_covariance == "lowrank" else None,
        "layers": reports,
    }
