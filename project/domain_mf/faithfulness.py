"""Exact logit attribution and bottleneck interventions for linear heads.

The supported models end in a linear classifier over pooled convolutional
features. This makes every channel contribution exact rather than a post-hoc
approximation and permits controlled causal interventions at the bottleneck.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LogitDecomposition:
    features: torch.Tensor
    contributions: torch.Tensor
    bias: torch.Tensor
    logits: torch.Tensor
    reconstructed_logits: torch.Tensor


def decompose_logits(model: nn.Module, inputs: torch.Tensor) -> LogitDecomposition:
    """Return an exact per-channel decomposition of every class logit."""
    if not hasattr(model, "penultimate_features") or not hasattr(model, "classifier"):
        raise TypeError("Model needs penultimate_features and a linear classifier")
    features = model.penultimate_features(inputs)
    contributions = features[:, None, :] * model.classifier.weight[None, :, :]
    if model.classifier.bias is None:
        bias = torch.zeros(
            model.spec.num_classes, device=features.device, dtype=features.dtype
        )
    else:
        bias = model.classifier.bias
    logits = F.linear(features, model.classifier.weight, model.classifier.bias)
    reconstructed = contributions.sum(dim=-1) + bias[None, :]
    return LogitDecomposition(
        features=features,
        contributions=contributions,
        bias=bias,
        logits=logits,
        reconstructed_logits=reconstructed,
    )


def target_contributions(
    contributions: torch.Tensor, target_classes: torch.Tensor
) -> torch.Tensor:
    """Select one target-class contribution vector per sample."""
    batch_indices = torch.arange(contributions.shape[0], device=contributions.device)
    return contributions[batch_indices, target_classes]


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask selecting the k greatest signed scores in each row."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, channels]")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be in [1, {scores.shape[1]}], got {k}")
    indices = scores.topk(k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(1, indices, True)


def random_k_mask(
    batch_size: int,
    channels: int,
    k: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample k distinct channels/sample using a reproducible generator."""
    scores = torch.rand(
        (batch_size, channels), generator=generator, device=device
    )
    return topk_mask(scores, k)


def intervened_logits(
    model: nn.Module, features: torch.Tensor, retained_mask: torch.Tensor
) -> torch.Tensor:
    """Classify after retaining only selected bottleneck channels."""
    if retained_mask.shape != features.shape:
        raise ValueError("retained_mask and features must have the same shape")
    return F.linear(
        features * retained_mask.to(features.dtype),
        model.classifier.weight,
        model.classifier.bias,
    )


def mask_jaccard(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Per-sample Jaccard similarity between two boolean explanation masks."""
    intersection = (first & second).sum(dim=1).float()
    union = (first | second).sum(dim=1).float()
    return intersection / union.clamp_min(1.0)


def contribution_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity between signed contribution vectors."""
    return F.cosine_similarity(first, second, dim=1, eps=1e-12)


def ordinal_rank_correlation(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Per-sample Spearman-style correlation using ordinal channel ranks.

    Exactly tied scores are ordered by channel index. Top-k Jaccard and cosine
    should remain the primary measures when many ReLU channels are zero.
    """
    first_ranks = first.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
    second_ranks = second.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
    first_centered = first_ranks.float() - first_ranks.float().mean(dim=1, keepdim=True)
    second_centered = second_ranks.float() - second_ranks.float().mean(
        dim=1, keepdim=True
    )
    numerator = (first_centered * second_centered).sum(dim=1)
    denominator = torch.sqrt(
        first_centered.square().sum(dim=1)
        * second_centered.square().sum(dim=1)
    )
    return numerator / denominator.clamp_min(1e-12)
