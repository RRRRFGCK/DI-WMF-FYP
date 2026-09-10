import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    MFOneLayer,
    MFTwoLayer,
    SharedConceptCNN,
    SmallResNet,
    StandardCNN,
    StandardResNet18,
)


@dataclass
class InitialisationReport:
    method: str
    layers_initialised: int
    covariance: str
    shrinkage: float
    minimum_variance: float | None = None
    maximum_variance: float | None = None
    output_scale: float | None = None
    initialised_modules: tuple[str, ...] = ()
    covariance_rank: int | None = None


def _stabilise_diagonal_variance(
    variance: torch.Tensor, shrinkage: float, eps: float
) -> torch.Tensor:
    positive = variance[variance > 0]
    target = positive.mean() if positive.numel() else variance.new_tensor(1.0)
    variance = (1.0 - shrinkage) * variance + shrinkage * target
    return variance.clamp_min(max(eps, float(target) * 1e-6))


def _rescale_bank(weight: torch.Tensor, bias: torch.Tensor):
    fan_in = weight[0].numel()
    target_std = math.sqrt(2.0 / fan_in)
    current_std = float(weight.std())
    if current_std > 0:
        scale = target_std / current_std
        weight = weight * scale
        bias = bias * scale
    return weight.float(), bias.float()


def _class_feature_stats(loader, extractor, num_classes: int, device: torch.device):
    sums = None
    sums_sq = None
    counts = torch.zeros(num_classes, dtype=torch.float64)
    was_training = getattr(extractor, "training", False)
    with torch.no_grad():
        for inputs, labels in loader:
            features = extractor(
                inputs.to(device, non_blocking=device.type == "cuda")
            ).detach().cpu().double()
            features = features.flatten(1)
            if sums is None:
                sums = torch.zeros(num_classes, features.shape[1], dtype=torch.float64)
                sums_sq = torch.zeros_like(sums)
            for class_id in labels.unique().tolist():
                mask = labels == class_id
                selected = features[mask]
                sums[class_id] += selected.sum(0)
                sums_sq[class_id] += selected.square().sum(0)
                counts[class_id] += selected.shape[0]
    if was_training and isinstance(extractor, nn.Module):
        extractor.train()
    if sums is None or (counts == 0).any():
        raise RuntimeError("Every class needs at least one initialisation sample")
    means = sums / counts[:, None]
    residual_ss = sums_sq - sums.square() / counts[:, None]
    pooled_variance = residual_ss.sum(0) / (counts.sum() - num_classes)
    global_mean = sums.sum(0) / counts.sum()
    priors = counts / counts.sum()
    return means, global_mean, pooled_variance, priors


def _class_feature_full_stats(
    loader, extractor, num_classes: int, device: torch.device
):
    """Class means and pooled within-class full covariance for small features."""
    sums = None
    sum_outer = None
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    with torch.no_grad():
        for inputs, labels in loader:
            features = extractor(
                inputs.to(device, non_blocking=device.type == "cuda")
            ).detach().flatten(1).double()
            labels = labels.to(device)
            if sums is None:
                dimension = features.shape[1]
                sums = torch.zeros(
                    num_classes, dimension, dtype=torch.float64, device=device
                )
                sum_outer = torch.zeros(
                    dimension, dimension, dtype=torch.float64, device=device
                )
            sum_outer += features.T @ features
            for class_id in labels.unique().tolist():
                selected = features[labels == class_id]
                sums[class_id] += selected.sum(0)
                counts[class_id] += selected.shape[0]
    if sums is None or (counts == 0).any():
        raise RuntimeError("Every class needs at least one initialisation sample")
    means = sums / counts[:, None]
    between = sum(
        torch.outer(sums[class_id], sums[class_id]) / counts[class_id]
        for class_id in range(num_classes)
    )
    covariance = (sum_outer - between) / max(1, int(counts.sum()) - num_classes)
    covariance = 0.5 * (covariance + covariance.T)
    background = sums.sum(0) / counts.sum()
    priors = counts / counts.sum()
    return means, background, covariance, priors


def _local_patch_stats(model: MFTwoLayer, loader, device: torch.device):
    spec = model.spec
    patch_dim = spec.input_channels * spec.patch_size * spec.patch_size
    positions = spec.grid_size ** 2
    sums = torch.zeros(
        spec.num_classes, positions, patch_dim, dtype=torch.float64
    )
    sums_sq = torch.zeros_like(sums)
    counts = torch.zeros(spec.num_classes, dtype=torch.float64)
    with torch.no_grad():
        for inputs, labels in loader:
            patches = F.unfold(
                inputs.to(device, non_blocking=device.type == "cuda"),
                kernel_size=spec.patch_size,
                stride=spec.stride,
            ).detach().cpu().double()
            patches = patches.transpose(1, 2)
            for class_id in labels.unique().tolist():
                selected = patches[labels == class_id]
                sums[class_id] += selected.sum(0)
                sums_sq[class_id] += selected.square().sum(0)
                counts[class_id] += selected.shape[0]
    if (counts == 0).any():
        raise RuntimeError("Every class needs at least one initialisation sample")
    means = sums / counts[:, None, None]
    residual_ss = sums_sq - sums.square() / counts[:, None, None]
    degrees = positions * (counts.sum() - spec.num_classes)
    pooled_variance = residual_ss.sum((0, 1)) / degrees
    background = sums.sum(0) / counts.sum()
    priors = counts / counts.sum()
    return means, background, pooled_variance, priors


def _collect_salient_patches(
    loader,
    extractor,
    num_classes: int,
    kernel_size: int,
    padding: int,
    sampling_stride: int,
    device: torch.device,
    patches_per_image: int = 2,
    max_patches_per_class: int = 1024,
):
    """Collect high-variance local patches without using validation/test data."""
    collected = [[] for _ in range(num_classes)]
    counts = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in loader:
            features = extractor(
                inputs.to(device, non_blocking=device.type == "cuda")
            )
            patches = F.unfold(
                features,
                kernel_size=kernel_size,
                padding=padding,
                stride=sampling_stride,
            ).transpose(1, 2)
            keep = min(patches_per_image, patches.shape[1])
            scores = patches.var(dim=2, unbiased=False)
            indices = scores.topk(keep, dim=1).indices
            selected = patches.gather(
                1, indices.unsqueeze(2).expand(-1, -1, patches.shape[2])
            )
            labels_device = labels.to(device)
            for class_id in labels.unique().tolist():
                remaining = max_patches_per_class - counts[class_id]
                if remaining <= 0:
                    continue
                class_patches = selected[labels_device == class_id].reshape(
                    -1, selected.shape[2]
                )
                class_patches = class_patches[:remaining].detach().cpu()
                if class_patches.numel():
                    collected[class_id].append(class_patches)
                    counts[class_id] += class_patches.shape[0]
            if min(counts) >= max_patches_per_class:
                break
    result = []
    for class_id, chunks in enumerate(collected):
        if not chunks:
            raise RuntimeError(f"No patches collected for class {class_id}")
        result.append(torch.cat(chunks, dim=0))
    return result


def _squared_distances(samples: torch.Tensor, centers: torch.Tensor):
    return (
        samples.square().sum(1, keepdim=True)
        + centers.square().sum(1).unsqueeze(0)
        - 2.0 * samples @ centers.T
    ).clamp_min_(0.0)


def _deterministic_kmeans(samples: torch.Tensor, clusters: int, iterations: int = 10):
    """Small deterministic k-means used to form an auditable template bank."""
    if samples.shape[0] < clusters:
        raise RuntimeError(
            f"Need at least {clusters} patches, received {samples.shape[0]}"
        )
    first = samples.square().sum(1).argmax()
    chosen = [int(first)]
    minimum_distance = _squared_distances(samples, samples[first].unsqueeze(0))[:, 0]
    for _ in range(1, clusters):
        next_index = int(minimum_distance.argmax())
        chosen.append(next_index)
        distance = _squared_distances(
            samples, samples[next_index].unsqueeze(0)
        )[:, 0]
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = samples[chosen].clone()
    for _ in range(iterations):
        assignment = _squared_distances(samples, centers).argmin(1)
        updated = centers.clone()
        for cluster_id in range(clusters):
            mask = assignment == cluster_id
            if mask.any():
                updated[cluster_id] = samples[mask].mean(0)
        if torch.allclose(updated, centers, rtol=1e-5, atol=1e-6):
            centers = updated
            break
        centers = updated
    assignment = _squared_distances(samples, centers).argmin(1)
    return centers, assignment


def _prototype_filter_bank(
    loader,
    extractor,
    num_classes: int,
    input_channels: int,
    kernel_size: int,
    padding: int,
    sampling_stride: int,
    prototypes_per_class: int,
    method: str,
    shrinkage: float,
    eps: float,
    device: torch.device,
    max_patches_per_class: int,
    covariance_rank: int = 16,
):
    patches_by_class = _collect_salient_patches(
        loader,
        extractor,
        num_classes,
        kernel_size,
        padding,
        sampling_stride,
        device,
        max_patches_per_class=max_patches_per_class,
    )
    centers = []
    assignments = []
    device_patches = []
    for patches in patches_by_class:
        patches = patches.to(device=device, dtype=torch.float64)
        class_centers, assignment = _deterministic_kmeans(
            patches, prototypes_per_class
        )
        device_patches.append(patches)
        centers.append(class_centers)
        assignments.append(assignment)
    centers = torch.stack(centers)
    total_count = sum(patches.shape[0] for patches in device_patches)
    background = sum(patches.sum(0) for patches in device_patches) / total_count
    counts = torch.tensor(
        [patches.shape[0] for patches in device_patches],
        dtype=torch.float64,
        device=device,
    )
    priors = counts / counts.sum()
    # These banks are intermediate ReLU detectors, not mutually exclusive
    # classifiers. A class-prior log bias would shift every detector negative
    # and can suppress an entire bank before the next layer.
    detector_priors = torch.ones_like(priors)
    variance = None
    if method in {"di_wmf", "lowrank_wmf"}:
        residual_ss = torch.zeros_like(background)
        residuals = []
        for class_id, patches in enumerate(device_patches):
            residual = patches - centers[class_id][assignments[class_id]]
            residual_ss += residual.square().sum(0)
            if method == "lowrank_wmf":
                residuals.append(residual)
        degrees = max(1, total_count - num_classes * prototypes_per_class)
        variance = residual_ss / degrees
        if method == "lowrank_wmf":
            residual_matrix = torch.cat(residuals)
            covariance = residual_matrix.T @ residual_matrix / degrees
            weight, bias, variance, _ = _lowrank_discriminants(
                centers,
                background,
                covariance,
                detector_priors,
                shrinkage,
                eps,
                covariance_rank,
            )
        else:
            weight, bias, variance = _diagonal_discriminants(
                centers, background, variance, detector_priors, shrinkage, eps
            )
    else:
        weight, bias = _unwhitened_discriminants(
            centers, background, detector_priors
        )
    weight = weight.reshape(
        num_classes * prototypes_per_class,
        input_channels,
        kernel_size,
        kernel_size,
    )
    bias = bias.reshape(-1)
    weight, bias = _rescale_bank(weight, bias)
    return weight, bias, variance


def _diagonal_discriminants(
    means: torch.Tensor,
    background: torch.Tensor,
    variance: torch.Tensor,
    priors: torch.Tensor,
    shrinkage: float,
    eps: float,
):
    variance = _stabilise_diagonal_variance(variance, shrinkage, eps)
    inverse_variance = variance.reciprocal()
    weight = (means - background) * inverse_variance
    bias = -0.5 * (
        (means.square() - background.square()) * inverse_variance
    ).sum(-1)
    while priors.ndim < bias.ndim:
        priors = priors.unsqueeze(-1)
    bias = bias + priors.clamp_min(1e-12).log()
    return weight, bias, variance


def _lowrank_discriminants(
    means: torch.Tensor,
    background: torch.Tensor,
    covariance: torch.Tensor,
    priors: torch.Tensor,
    shrinkage: float,
    eps: float,
    rank: int,
):
    """Matched discriminants under a diagonal-plus-low-rank covariance.

    A shrunk covariance is approximated as ``D + U diag(lambda) U.T`` and
    inverted with the Woodbury identity. This retains correlated noise modes
    without requiring an unstable full inverse.
    """
    dimension = covariance.shape[0]
    effective_rank = min(max(1, int(rank)), dimension - 1)
    target = covariance.diag().mean().clamp_min(eps)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * target * torch.eye(
        dimension, dtype=covariance.dtype, device=covariance.device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(shrunk)
    values = eigenvalues[-effective_rank:].clamp_min(eps)
    vectors = eigenvectors[:, -effective_rank:]
    lowrank_diagonal = (vectors.square() * values[None, :]).sum(1)
    diagonal = (shrunk.diag() - lowrank_diagonal).clamp_min(
        max(eps, float(target) * 1e-6)
    )
    inverse_diagonal = diagonal.reciprocal()
    scaled_vectors = inverse_diagonal[:, None] * vectors
    middle = torch.diag(values.reciprocal()) + vectors.T @ scaled_vectors
    precision = torch.diag(inverse_diagonal) - scaled_vectors @ torch.linalg.solve(
        middle, scaled_vectors.T
    )

    original_shape = means.shape
    flat_means = means.reshape(-1, original_shape[-1])
    flat_background = background.expand_as(means).reshape_as(flat_means)
    weight = ((flat_means - flat_background) @ precision).reshape(original_shape)
    mean_energy = (flat_means * (flat_means @ precision)).sum(1)
    background_energy = (
        flat_background * (flat_background @ precision)
    ).sum(1)
    bias = (-0.5 * (mean_energy - background_energy)).reshape(original_shape[:-1])
    while priors.ndim < bias.ndim:
        priors = priors.unsqueeze(-1)
    bias = bias + priors.clamp_min(1e-12).log()
    return weight, bias, diagonal, effective_rank


def _unwhitened_discriminants(
    means: torch.Tensor, background: torch.Tensor, priors: torch.Tensor
):
    """Matched templates under an identity within-class covariance model."""
    weight = means - background
    bias = -0.5 * (means.square() - background.square()).sum(-1)
    while priors.ndim < bias.ndim:
        priors = priors.unsqueeze(-1)
    bias = bias + priors.clamp_min(1e-12).log()
    return weight, bias


def _pca_stem_bank(
    loader,
    model: StandardCNN,
    device: torch.device,
):
    """Paired-sign principal component filters for a controlled stem baseline."""
    patches_by_class = _collect_salient_patches(
        loader,
        lambda x: x,
        model.spec.num_classes,
        model.conv1.kernel_size[0],
        model.conv1.padding[0],
        sampling_stride=2,
        device=device,
        max_patches_per_class=1024,
    )
    samples = torch.cat(patches_by_class).to(device=device, dtype=torch.float64)
    mean = samples.mean(0)
    centered = samples - mean
    covariance = centered.T @ centered / max(1, samples.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    pairs = (model.conv1.out_channels + 1) // 2
    pairs = min(pairs, eigenvectors.shape[1])
    components = eigenvectors[:, -pairs:].T.flip(0)
    filters = torch.stack((components, -components), dim=1).flatten(0, 1)
    if filters.shape[0] < model.conv1.out_channels:
        repeats = math.ceil(model.conv1.out_channels / filters.shape[0])
        filters = filters.repeat(repeats, 1)
    filters = filters[: model.conv1.out_channels]
    bias = -(filters * mean).sum(1)
    filters = filters.reshape_as(model.conv1.weight)
    filters, bias = _rescale_bank(filters, bias)
    return filters, bias, min(pairs, int((eigenvalues > 1e-12).sum()))


def _gabor_stem_bank(model: StandardCNN):
    """Deterministic multi-scale, multi-orientation analytic Gabor bank."""
    out_channels = model.conv1.out_channels
    input_channels = model.conv1.in_channels
    kernel_size = model.conv1.kernel_size[0]
    coordinates = torch.arange(kernel_size, dtype=torch.float64)
    coordinates = coordinates - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    orientations = torch.linspace(0, math.pi, 8 + 1, dtype=torch.float64)[:-1]
    frequencies = (0.18, 0.32)
    phases = (0.0, math.pi / 2)
    if input_channels == 1:
        colour_vectors = [torch.ones(1, dtype=torch.float64)]
    else:
        colour_vectors = [
            torch.tensor((1.0, 1.0, 1.0), dtype=torch.float64),
            torch.tensor((1.0, -1.0, 0.0), dtype=torch.float64),
            torch.tensor((0.0, 1.0, -1.0), dtype=torch.float64),
        ]
    filters = []
    index = 0
    while len(filters) < out_channels:
        theta = orientations[index % len(orientations)]
        frequency = frequencies[(index // len(orientations)) % len(frequencies)]
        phase = phases[
            (index // (len(orientations) * len(frequencies))) % len(phases)
        ]
        colour = colour_vectors[index % len(colour_vectors)]
        colour = colour / colour.norm().clamp_min(1e-12)
        rotated_x = xx * torch.cos(theta) + yy * torch.sin(theta)
        rotated_y = -xx * torch.sin(theta) + yy * torch.cos(theta)
        sigma = 0.55 * kernel_size
        envelope = torch.exp(
            -(rotated_x.square() + 0.5**2 * rotated_y.square()) / (2 * sigma**2)
        )
        spatial = envelope * torch.cos(2 * math.pi * frequency * rotated_x + phase)
        spatial = spatial - spatial.mean()
        kernel = colour[:, None, None] * spatial[None, :, :]
        kernel = kernel / kernel.norm().clamp_min(1e-12)
        filters.append(kernel)
        index += 1
    weight = torch.stack(filters).reshape_as(model.conv1.weight)
    bias = torch.zeros(out_channels, dtype=weight.dtype)
    return _rescale_bank(weight, bias)


def _shared_prototype_filter_bank(
    loader,
    extractor,
    model,
    module: nn.Conv2d,
    sampling_stride: int,
    device: torch.device,
    shrinkage: float,
    eps: float,
    max_patches_per_class: int,
    method: str = "di_wmf",
):
    patches_by_class = _collect_salient_patches(
        loader,
        extractor,
        model.spec.num_classes,
        module.kernel_size[0],
        module.padding[0],
        sampling_stride,
        device,
        max_patches_per_class=max_patches_per_class,
    )
    samples = torch.cat(patches_by_class).to(device=device, dtype=torch.float64)
    centers, assignment = _deterministic_kmeans(samples, module.out_channels)
    background = samples.mean(0)
    residual = samples - centers[assignment]
    priors = torch.ones(module.out_channels, dtype=torch.float64, device=device)
    variance = None
    if method == "di_wmf":
        variance = residual.square().mean(0)
        weight, bias, variance = _diagonal_discriminants(
            centers, background, variance, priors, shrinkage, eps
        )
    else:
        weight, bias = _unwhitened_discriminants(
            centers, background, priors
        )
    weight = weight.reshape_as(module.weight)
    weight, bias = _rescale_bank(weight, bias)
    return weight, bias, variance


def _set_kaiming(module: nn.Module):
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _set_kaiming_model(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            _set_kaiming(module)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _calibrate_batch_norm(model: nn.Module, fit_loader, device: torch.device):
    """Estimate BatchNorm running statistics from training inputs only."""
    batch_norms = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    if not batch_norms:
        return
    was_training = model.training
    momenta = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    with torch.no_grad():
        for inputs, _ in fit_loader:
            model(inputs.to(device, non_blocking=device.type == "cuda"))
    for module, momentum in zip(batch_norms, momenta):
        module.momentum = momentum
    model.train(was_training)


def _copy_parameters(module: nn.Module, weight: torch.Tensor, bias: torch.Tensor):
    if tuple(weight.shape) != tuple(module.weight.shape):
        raise ValueError(f"Weight shape {tuple(weight.shape)} != {tuple(module.weight.shape)}")
    with torch.no_grad():
        module.weight.copy_(weight.to(module.weight.device, module.weight.dtype))
        if module.bias is not None:
            module.bias.copy_(bias.to(module.bias.device, module.bias.dtype))


def fit_classifier_head(
    model: nn.Module,
    fit_loader,
    method: str,
    device: torch.device,
    shrinkage: float = 0.1,
    eps: float = 1e-6,
    covariance_rank: int = 16,
) -> dict:
    """Fit only the final classifier from training-fold features.

    This deliberately leaves every feature-extractor parameter unchanged.  It
    is used by the Epoch-0 factorial control to separate the contribution of a
    fitted discriminant head from the contribution of DI-WMF convolutional
    templates.  The implementation is identical to the classifier stage in
    the layer-wise initialiser.
    """
    method = method.lower()
    if method not in {"classmean", "di_wmf", "lowrank_wmf"}:
        raise ValueError(f"Unsupported fitted-head method: {method}")
    if not hasattr(model, "classifier") or not hasattr(model, "penultimate_features"):
        raise TypeError("The model must expose classifier and penultimate_features")

    was_training = model.training
    model.eval()
    if method == "lowrank_wmf":
        means, background, covariance, priors = _class_feature_full_stats(
            fit_loader,
            model.penultimate_features,
            model.spec.num_classes,
            device,
        )
        weight, bias, variance, retained_rank = _lowrank_discriminants(
            means,
            background,
            covariance,
            priors,
            shrinkage,
            eps,
            covariance_rank,
        )
    else:
        means, background, variance, priors = _class_feature_stats(
            fit_loader,
            model.penultimate_features,
            model.spec.num_classes,
            device,
        )
        retained_rank = None
        if method == "di_wmf":
            weight, bias, variance = _diagonal_discriminants(
                means, background, variance, priors, shrinkage, eps
            )
        else:
            weight, bias = _unwhitened_discriminants(means, background, priors)

    weight, bias = _rescale_bank(weight, bias)
    _copy_parameters(model.classifier, weight, bias)
    model.train(was_training)
    return {
        "method": method,
        "covariance_rank": retained_rank,
        "minimum_variance": float(variance.min()),
        "maximum_variance": float(variance.max()),
    }


def _initialise_standard_cnn(
    model: StandardCNN,
    fit_loader,
    method: str,
    device: torch.device,
    layers: int,
    shrinkage: float,
    eps: float,
    covariance_rank: int,
):
    if method in {"gabor", "pca"}:
        if layers != 1:
            raise ValueError(f"{method} is a controlled stem-only initialiser")
        if method == "gabor":
            stem_weight, stem_bias = _gabor_stem_bank(model)
            pca_rank = None
        else:
            stem_weight, stem_bias, pca_rank = _pca_stem_bank(
                fit_loader, model, device
            )
        _copy_parameters(model.conv1, stem_weight, stem_bias)
        _set_kaiming(model.conv2)
        _set_kaiming(model.classifier)
        return InitialisationReport(
            method=method,
            layers_initialised=1,
            covariance="pca" if method == "pca" else "analytic_gabor",
            shrinkage=shrinkage,
            initialised_modules=("conv1",),
            covariance_rank=pca_rank,
        )

    variances = []
    initialised_modules = []
    stem_weight, stem_bias, stem_variance = _prototype_filter_bank(
        fit_loader,
        lambda x: x,
        model.spec.num_classes,
        model.spec.input_channels,
        model.conv1.kernel_size[0],
        model.conv1.padding[0],
        sampling_stride=2,
        prototypes_per_class=model.stem_templates_per_class,
        method=method,
        shrinkage=shrinkage,
        eps=eps,
        device=device,
        max_patches_per_class=1024,
        covariance_rank=covariance_rank,
    )
    _copy_parameters(model.conv1, stem_weight, stem_bias)
    initialised_modules.append("conv1")
    if stem_variance is not None:
        variances.append(stem_variance)

    if layers >= 2:
        model.eval()
        body_weight, body_bias, body_variance = _prototype_filter_bank(
            fit_loader,
            model.first_layer_features,
            model.spec.num_classes,
            model.conv2.in_channels,
            model.conv2.kernel_size[0],
            model.conv2.padding[0],
            sampling_stride=2,
            prototypes_per_class=model.body_templates_per_class,
            method=method,
            shrinkage=shrinkage,
            eps=eps,
            device=device,
            max_patches_per_class=512,
            covariance_rank=covariance_rank,
        )
        _copy_parameters(model.conv2, body_weight, body_bias)
        initialised_modules.append("conv2")
        if body_variance is not None:
            variances.append(body_variance)
    else:
        _set_kaiming(model.conv2)

    if layers >= 3:
        model.eval()
        if method == "lowrank_wmf":
            means, background, covariance, priors = _class_feature_full_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
            head_weight, head_bias, variance, _ = _lowrank_discriminants(
                means,
                background,
                covariance,
                priors,
                shrinkage,
                eps,
                covariance_rank,
            )
            variances.append(variance)
        else:
            means, background, variance, priors = _class_feature_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
        if method == "di_wmf":
            head_weight, head_bias, variance = _diagonal_discriminants(
                means, background, variance, priors, shrinkage, eps
            )
            variances.append(variance)
        elif method != "lowrank_wmf":
            head_weight, head_bias = _unwhitened_discriminants(
                means, background, priors
            )
        head_weight, head_bias = _rescale_bank(head_weight, head_bias)
        _copy_parameters(model.classifier, head_weight, head_bias)
        initialised_modules.append("classifier")
    else:
        _set_kaiming(model.classifier)
    model.train()
    minimum = min(float(variance.min()) for variance in variances) if variances else None
    maximum = max(float(variance.max()) for variance in variances) if variances else None
    return InitialisationReport(
        method=method,
        layers_initialised=len(initialised_modules),
        covariance=(
            "diagonal_plus_lowrank"
            if method == "lowrank_wmf"
            else "diagonal" if method == "di_wmf" else "none"
        ),
        shrinkage=shrinkage,
        minimum_variance=minimum,
        maximum_variance=maximum,
        initialised_modules=tuple(initialised_modules),
        covariance_rank=covariance_rank if method == "lowrank_wmf" else None,
    )


def _initialise_shared_concept_cnn(
    model: SharedConceptCNN,
    fit_loader,
    method: str,
    device: torch.device,
    layers: int,
    shrinkage: float,
    eps: float,
):
    if method not in {"kmeans", "classmean", "di_wmf"}:
        raise ValueError(
            "SharedConceptCNN data initialisation supports kmeans or di_wmf"
        )
    variances = []
    stem_weight, stem_bias, variance = _shared_prototype_filter_bank(
        fit_loader,
        lambda x: x,
        model,
        model.conv1,
        sampling_stride=2,
        device=device,
        shrinkage=shrinkage,
        eps=eps,
        max_patches_per_class=1024,
        method=method,
    )
    _copy_parameters(model.conv1, stem_weight, stem_bias)
    if variance is not None:
        variances.append(variance)
    modules = ["conv1"]
    if layers >= 2:
        model.eval()
        body_weight, body_bias, variance = _shared_prototype_filter_bank(
            fit_loader,
            model.first_layer_features,
            model,
            model.conv2,
            sampling_stride=2,
            device=device,
            shrinkage=shrinkage,
            eps=eps,
            max_patches_per_class=512,
            method=method,
        )
        _copy_parameters(model.conv2, body_weight, body_bias)
        if variance is not None:
            variances.append(variance)
        modules.append("conv2")
    else:
        _set_kaiming(model.conv2)
    if layers >= 3:
        means, background, variance, priors = _class_feature_stats(
            fit_loader,
            model.penultimate_features,
            model.spec.num_classes,
            device,
        )
        if method == "di_wmf":
            weight, bias, variance = _diagonal_discriminants(
                means, background, variance, priors, shrinkage, eps
            )
        else:
            weight, bias = _unwhitened_discriminants(
                means, background, priors
            )
        weight, bias = _rescale_bank(weight, bias)
        _copy_parameters(model.classifier, weight, bias)
        if method == "di_wmf":
            variances.append(variance)
        modules.append("classifier")
    else:
        _set_kaiming(model.classifier)
    model.train()
    return InitialisationReport(
        method=method,
        layers_initialised=len(modules),
        covariance="diagonal_shared" if method == "di_wmf" else "none",
        shrinkage=shrinkage,
        minimum_variance=(
            min(float(item.min()) for item in variances) if variances else None
        ),
        maximum_variance=(
            max(float(item.max()) for item in variances) if variances else None
        ),
        initialised_modules=tuple(modules),
    )


def _initialise_small_resnet(
    model: SmallResNet,
    fit_loader,
    method: str,
    device: torch.device,
    layers: int,
    shrinkage: float,
    eps: float,
    covariance_rank: int,
):
    # Residual transformations remain conventionally initialised. The domain
    # prior is tested at the input matched-filter bank and, for layers >= 3,
    # at the pooled classifier head.
    _set_kaiming_model(model)
    stem_weight, stem_bias, stem_variance = _prototype_filter_bank(
        fit_loader,
        lambda x: x,
        model.spec.num_classes,
        model.spec.input_channels,
        model.conv1.kernel_size[0],
        model.conv1.padding[0],
        sampling_stride=2,
        prototypes_per_class=model.stem_templates_per_class,
        method=method,
        shrinkage=shrinkage,
        eps=eps,
        device=device,
        max_patches_per_class=1024,
        covariance_rank=covariance_rank,
    )
    _copy_parameters(model.conv1, stem_weight, stem_bias)
    modules = ["conv1"]
    variances = [stem_variance] if stem_variance is not None else []
    if layers >= 3:
        model.eval()
        if method == "lowrank_wmf":
            means, background, covariance, priors = _class_feature_full_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
            weight, bias, variance, _ = _lowrank_discriminants(
                means,
                background,
                covariance,
                priors,
                shrinkage,
                eps,
                covariance_rank,
            )
        else:
            means, background, variance, priors = _class_feature_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
            if method == "di_wmf":
                weight, bias, variance = _diagonal_discriminants(
                    means, background, variance, priors, shrinkage, eps
                )
            else:
                weight, bias = _unwhitened_discriminants(
                    means, background, priors
                )
        weight, bias = _rescale_bank(weight, bias)
        _copy_parameters(model.classifier, weight, bias)
        if method in {"di_wmf", "lowrank_wmf"}:
            variances.append(variance)
        modules.append("classifier")
    model.train()
    return InitialisationReport(
        method=method,
        layers_initialised=len(modules),
        covariance=(
            "diagonal_plus_lowrank"
            if method == "lowrank_wmf"
            else "diagonal" if method == "di_wmf" else "none"
        ),
        shrinkage=shrinkage,
        minimum_variance=(
            min(float(item.min()) for item in variances) if variances else None
        ),
        maximum_variance=(
            max(float(item.max()) for item in variances) if variances else None
        ),
        initialised_modules=tuple(modules),
        covariance_rank=covariance_rank if method == "lowrank_wmf" else None,
    )


def _initialise_resnet18(
    model: StandardResNet18,
    fit_loader,
    method: str,
    device: torch.device,
    layers: int,
    shrinkage: float,
    eps: float,
    covariance_rank: int,
):
    """Initialise a canonical ResNet-18 stem and pooled linear classifier.

    ResNet-18 has 64 stem channels, which is not divisible by ten classes. We
    form the next-largest balanced class-conditional dictionary and retain
    six templates per class plus one extra for the first four classes. The
    residual transformations remain conventionally Kaiming-initialised.
    """
    _set_kaiming_model(model)
    prototypes_per_class = math.ceil(
        model.conv1.out_channels / model.spec.num_classes
    )
    stem_weight, stem_bias, stem_variance = _prototype_filter_bank(
        fit_loader,
        lambda x: x,
        model.spec.num_classes,
        model.spec.input_channels,
        model.conv1.kernel_size[0],
        model.conv1.padding[0],
        sampling_stride=2,
        prototypes_per_class=prototypes_per_class,
        method=method,
        shrinkage=shrinkage,
        eps=eps,
        device=device,
        max_patches_per_class=1024,
        covariance_rank=covariance_rank,
    )
    class_ids = model.anchor_classes("conv1")
    per_class_seen = torch.zeros(model.spec.num_classes, dtype=torch.long)
    selected = []
    for class_id in class_ids.tolist():
        selected.append(
            class_id * prototypes_per_class + int(per_class_seen[class_id])
        )
        per_class_seen[class_id] += 1
    selected = torch.tensor(selected, dtype=torch.long, device=stem_weight.device)
    _copy_parameters(
        model.conv1,
        stem_weight.index_select(0, selected),
        stem_bias.index_select(0, selected),
    )
    _calibrate_batch_norm(model, fit_loader, device)
    modules = ["conv1"]
    variances = [stem_variance] if stem_variance is not None else []
    if layers >= 3:
        model.eval()
        if method == "lowrank_wmf":
            means, background, covariance, priors = _class_feature_full_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
            weight, bias, variance, _ = _lowrank_discriminants(
                means,
                background,
                covariance,
                priors,
                shrinkage,
                eps,
                covariance_rank,
            )
        else:
            means, background, variance, priors = _class_feature_stats(
                fit_loader,
                model.penultimate_features,
                model.spec.num_classes,
                device,
            )
            if method == "di_wmf":
                weight, bias, variance = _diagonal_discriminants(
                    means, background, variance, priors, shrinkage, eps
                )
            else:
                weight, bias = _unwhitened_discriminants(
                    means, background, priors
                )
        weight, bias = _rescale_bank(weight, bias)
        _copy_parameters(model.classifier, weight, bias)
        if method in {"di_wmf", "lowrank_wmf"}:
            variances.append(variance)
        modules.append("classifier")
    model.train()
    return InitialisationReport(
        method=method,
        layers_initialised=len(modules),
        covariance=(
            "diagonal_plus_lowrank"
            if method == "lowrank_wmf"
            else "diagonal" if method == "di_wmf" else "none"
        ),
        shrinkage=shrinkage,
        minimum_variance=(
            min(float(item.min()) for item in variances) if variances else None
        ),
        maximum_variance=(
            max(float(item.max()) for item in variances) if variances else None
        ),
        initialised_modules=tuple(modules),
        covariance_rank=covariance_rank if method == "lowrank_wmf" else None,
    )


def calibrate_logit_scale(
    model: nn.Module,
    fit_loader,
    device: torch.device,
    target_std: float = 1.0,
    max_batches: int = 10,
) -> float:
    """Match output scale across initialisers using unlabelled training inputs.

    A single positive scale is applied to the final layer, so class rankings and
    epoch-0 accuracy are unchanged. This prevents high-dimensional feature
    spaces from creating overconfident logits before optimisation.
    """
    if target_std <= 0:
        raise ValueError("target_std must be positive")
    model.eval()
    count = 0
    total = 0.0
    total_sq = 0.0
    with torch.no_grad():
        for batch_index, (inputs, _) in enumerate(fit_loader):
            if batch_index >= max_batches:
                break
            logits = model(
                inputs.to(device, non_blocking=device.type == "cuda")
            ).double()
            count += logits.numel()
            total += float(logits.sum())
            total_sq += float(logits.square().sum())
    if count < 2:
        raise RuntimeError("Not enough logits to calibrate output scale")
    variance = max(0.0, (total_sq - total * total / count) / (count - 1))
    current_std = math.sqrt(variance)
    if not math.isfinite(current_std) or current_std <= 1e-12:
        raise RuntimeError(f"Invalid initial logit standard deviation: {current_std}")
    scale = target_std / current_std
    output_layer = model.output_layer
    with torch.no_grad():
        output_layer.weight.mul_(scale)
        output_layer.bias.mul_(scale)
    model.train()
    return scale


def initialise_model(
    model: nn.Module,
    fit_loader,
    method: str,
    device: torch.device,
    layers: int = 2,
    shrinkage: float = 0.1,
    eps: float = 1e-6,
    covariance_rank: int = 16,
) -> InitialisationReport:
    """Initialise a model from training data only.

    DI-WMF uses pooled within-class diagonal covariance. The common background
    discriminant is removed because it does not affect a softmax decision.
    """
    method = method.lower()
    supported = {
        "random",
        "random_stem",
        "kaiming",
        "classmean",
        "kmeans",
        "gabor",
        "pca",
        "di_wmf",
        "lowrank_wmf",
    }
    if method not in supported:
        raise ValueError(f"Unsupported initialiser: {method}")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")

    if method == "random":
        return InitialisationReport(method, 0, "none", shrinkage)
    if method == "random_stem":
        if not isinstance(model, StandardCNN) or layers != 1:
            raise ValueError("random_stem is a StandardCNN stem-only control")
        _set_kaiming(model.conv2)
        _set_kaiming(model.classifier)
        return InitialisationReport(
            method, 1, "none", shrinkage, initialised_modules=("conv1",)
        )
    if method == "kaiming":
        _set_kaiming_model(model)
        if isinstance(model, (StandardCNN, SharedConceptCNN)):
            _set_kaiming(model.classifier)
            return InitialisationReport(
                method, 3, "none", shrinkage,
                initialised_modules=("conv1", "conv2", "classifier")
            )
        if isinstance(model, StandardResNet18):
            _calibrate_batch_norm(model, fit_loader, device)
            return InitialisationReport(
                method, 3, "none", shrinkage,
                initialised_modules=("conv1", "residual_blocks", "classifier")
            )
        if isinstance(model, SmallResNet):
            return InitialisationReport(
                method, 3, "none", shrinkage,
                initialised_modules=("conv1", "residual_blocks", "classifier")
            )
        return InitialisationReport(
            method,
            2 if isinstance(model, MFTwoLayer) else 1,
            "none",
            shrinkage,
            initialised_modules=("conv1", "conv2")
            if isinstance(model, MFTwoLayer)
            else ("conv1",),
        )

    if method == "classmean" and isinstance(
        model, (StandardCNN, SharedConceptCNN, SmallResNet, StandardResNet18)
    ):
        raise ValueError(
            "classmean is defined only for the one-/two-layer mean-template "
            "models. StandardCNN-family models require multiple templates per "
            "class; use kmeans for the unwhitened multi-prototype control."
        )

    if isinstance(model, SharedConceptCNN):
        return _initialise_shared_concept_cnn(
            model, fit_loader, method, device, layers, shrinkage, eps
        )

    if isinstance(model, SmallResNet):
        if method in {"gabor", "pca"}:
            raise ValueError("Gabor/PCA controlled baselines currently use StandardCNN")
        return _initialise_small_resnet(
            model,
            fit_loader,
            method,
            device,
            layers,
            shrinkage,
            eps,
            covariance_rank,
        )

    if isinstance(model, StandardResNet18):
        if method in {"gabor", "pca"}:
            raise ValueError("Gabor/PCA controlled baselines currently use StandardCNN")
        return _initialise_resnet18(
            model,
            fit_loader,
            method,
            device,
            layers,
            shrinkage,
            eps,
            covariance_rank,
        )

    if isinstance(model, StandardCNN):
        return _initialise_standard_cnn(
            model,
            fit_loader,
            method,
            device,
            layers,
            shrinkage,
            eps,
            covariance_rank,
        )

    if method in {"gabor", "pca", "lowrank_wmf", "kmeans"}:
        raise ValueError(f"{method} currently supports StandardCNN-family models")

    minimum_variance = None
    maximum_variance = None
    if isinstance(model, MFOneLayer):
        means, background, variance, priors = _class_feature_stats(
            fit_loader, lambda x: x, model.spec.num_classes, device
        )
        if method == "classmean":
            weight, bias = _unwhitened_discriminants(means, background, priors)
        else:
            weight, bias, variance = _diagonal_discriminants(
                means, background, variance, priors, shrinkage, eps
            )
            minimum_variance = float(variance.min())
            maximum_variance = float(variance.max())
        weight = weight.reshape_as(model.conv1.weight.cpu())
        weight, bias = _rescale_bank(weight, bias)
        _copy_parameters(model.conv1, weight, bias)
        return InitialisationReport(
            method, 1, "diagonal" if method == "di_wmf" else "none", shrinkage,
            minimum_variance, maximum_variance
        )

    if not isinstance(model, MFTwoLayer):
        raise TypeError(type(model))

    means, background, variance, priors = _local_patch_stats(model, fit_loader, device)
    if method == "classmean":
        first_weight, first_bias = _unwhitened_discriminants(
            means, background[None, ...], priors
        )
    else:
        first_weight, first_bias, variance = _diagonal_discriminants(
            means, background[None, ...], variance, priors, shrinkage, eps
        )
        minimum_variance = float(variance.min())
        maximum_variance = float(variance.max())
    first_weight = first_weight.reshape_as(model.conv1.weight.cpu())
    first_bias = first_bias.reshape(-1)
    first_weight, first_bias = _rescale_bank(first_weight, first_bias)
    _copy_parameters(model.conv1, first_weight, first_bias)

    if layers >= 2:
        model.eval()
        feature_means, feature_background, feature_variance, feature_priors = (
            _class_feature_stats(
                fit_loader,
                lambda x: model.first_layer_features(x),
                model.spec.num_classes,
                device,
            )
        )
        if method == "classmean":
            second_weight, second_bias = _unwhitened_discriminants(
                feature_means, feature_background, feature_priors
            )
        else:
            second_weight, second_bias, _ = _diagonal_discriminants(
                feature_means,
                feature_background,
                feature_variance,
                feature_priors,
                shrinkage,
                eps,
            )
        second_weight = second_weight.reshape_as(model.conv2.weight.cpu())
        second_weight, second_bias = _rescale_bank(second_weight, second_bias)
        _copy_parameters(model.conv2, second_weight, second_bias)
    else:
        _set_kaiming(model.conv2)

    model.train()
    return InitialisationReport(
        method,
        min(layers, 2),
        "diagonal" if method == "di_wmf" else "none",
        shrinkage,
        minimum_variance,
        maximum_variance,
    )
