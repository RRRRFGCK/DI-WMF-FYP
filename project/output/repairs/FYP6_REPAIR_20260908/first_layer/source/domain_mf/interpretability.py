from collections import defaultdict

import torch
import torch.nn.functional as F


def capture_anchors(model, layers: int = 2):
    anchors = {}
    layer_names = getattr(model, "anchor_layer_names", ("conv1", "conv2"))
    for name in layer_names[:layers]:
        layer = getattr(model, name, None)
        if layer is not None:
            anchors[name] = layer.weight.detach().clone()
    return anchors


def anchor_regularisation(model, anchors):
    losses = []
    for name, anchor in anchors.items():
        current = getattr(model, name).weight.flatten(1)
        reference = anchor.to(current.device).flatten(1)
        losses.append((1.0 - F.cosine_similarity(current, reference, dim=1)).mean())
    return torch.stack(losses).mean() if losses else next(model.parameters()).new_tensor(0.0)


def kernel_drift(model, anchors):
    result = {}
    with torch.no_grad():
        for name, anchor in anchors.items():
            current = getattr(model, name).weight.detach().cpu().flatten(1)
            reference = anchor.detach().cpu().flatten(1)
            drift = 1.0 - F.cosine_similarity(current, reference, dim=1)
            result[name] = {
                "mean": float(drift.mean()),
                "std": float(drift.std(unbiased=False)),
                "max": float(drift.max()),
            }
    return result


def template_assignment_stability(model, anchors):
    result = {}
    with torch.no_grad():
        for name, anchor in anchors.items():
            current = F.normalize(
                getattr(model, name).weight.detach().cpu().flatten(1), dim=1
            )
            reference = F.normalize(anchor.detach().cpu().flatten(1), dim=1)
            nearest = (current @ reference.T).argmax(1)
            expected_classes = model.anchor_classes(name).cpu()
            assigned_classes = expected_classes[nearest]
            result[name] = float((assigned_classes == expected_classes).float().mean())
    return result


def class_group_evidence_gap(model, loader, device):
    """Contrast each pre-assigned class group's evidence on target/other inputs.

    This is deliberately distinct from the per-channel best-class selectivity
    used by ``evaluate_filter_selectivity.py``.
    """
    target_sum = defaultdict(float)
    target_count = defaultdict(int)
    other_sum = defaultdict(float)
    other_count = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            evidence = model.class_evidence(
                inputs.to(device, non_blocking=device.type == "cuda")
            ).detach().cpu()
            for class_id in range(evidence.shape[1]):
                target_mask = labels == class_id
                other_mask = ~target_mask
                if target_mask.any():
                    target_sum[class_id] += float(evidence[target_mask, class_id].sum())
                    target_count[class_id] += int(target_mask.sum())
                if other_mask.any():
                    other_sum[class_id] += float(evidence[other_mask, class_id].sum())
                    other_count[class_id] += int(other_mask.sum())
    per_class = {}
    for class_id in sorted(target_count):
        target_mean = target_sum[class_id] / target_count[class_id]
        other_mean = other_sum[class_id] / other_count[class_id]
        denominator = abs(target_mean) + abs(other_mean) + 1e-12
        per_class[str(class_id)] = (target_mean - other_mean) / denominator
    return {
        "mean": sum(per_class.values()) / len(per_class),
        "per_class": per_class,
    }


# Backward-compatible import for old result readers.  New metrics are stored
# under the explicit ``class_group_evidence_gap`` name by the trainer.
class_selectivity = class_group_evidence_gap
