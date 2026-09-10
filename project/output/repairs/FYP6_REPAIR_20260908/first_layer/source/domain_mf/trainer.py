import copy
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .interpretability import (
    anchor_regularisation,
    class_group_evidence_gap,
    kernel_drift,
    template_assignment_stability,
)


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    count = 0
    auxiliary_total = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels))
            if hasattr(model, "auxiliary_loss"):
                auxiliary_total += float(model.auxiliary_loss(inputs)) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()
    return {
        "loss": total_loss / count,
        "accuracy": 100.0 * correct / count,
        "auxiliary_loss": auxiliary_total / count if hasattr(model, "auxiliary_loss") else 0.0,
    }


def _lambda_for_epoch(base_lambda, schedule, epoch, epochs):
    if schedule == "none":
        return 0.0
    if schedule == "fixed":
        return base_lambda
    if schedule == "linear":
        end = max(1.0, 0.6 * epochs)
        return base_lambda * max(0.0, 1.0 - epoch / end)
    raise ValueError(schedule)


def _write_history(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    anchors,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    reg_lambda: float,
    reg_schedule: str,
    sparsity_lambda: float = 0.0,
    sparsity_mode: str = "l1",
    auxiliary_lambda: float = 0.0,
    lr_scheduler: str = "none",
    min_learning_rate: float = 1e-5,
):
    if sparsity_mode not in {"l1", "proximal"}:
        raise ValueError(f"Unknown sparsity mode: {sparsity_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_learning_rate
        )
    elif lr_scheduler != "none":
        raise ValueError(f"Unknown learning-rate scheduler: {lr_scheduler}")
    history = []
    start = time.perf_counter()

    initial_val = evaluate(model, val_loader, device)
    history.append(
        {
            "epoch": 0,
            "train_loss": "",
            "train_accuracy": "",
            "val_loss": initial_val["loss"],
            "val_accuracy": initial_val["accuracy"],
            "val_auxiliary_loss": initial_val["auxiliary_loss"],
            "regularisation": 0.0,
            "lambda": 0.0,
            "sparsity": 0.0,
            "sparsity_lambda": sparsity_lambda,
            "auxiliary_loss": 0.0,
            "auxiliary_lambda": auxiliary_lambda,
            "learning_rate": learning_rate,
            "elapsed_seconds": 0.0,
        }
    )
    best_accuracy = initial_val["accuracy"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        reg_sum = 0.0
        sparsity_sum = 0.0
        auxiliary_sum = 0.0
        current_learning_rate = optimizer.param_groups[0]["lr"]
        current_lambda = _lambda_for_epoch(
            reg_lambda, reg_schedule, epoch - 1, epochs
        )
        for inputs, labels in train_loader:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            classification_loss = criterion(logits, labels)
            regularisation = anchor_regularisation(model, anchors)
            sparsity = (
                model.sparsity_penalty()
                if hasattr(model, "sparsity_penalty")
                else classification_loss.new_tensor(0.0)
            )
            auxiliary = (
                model.auxiliary_loss(inputs)
                if auxiliary_lambda > 0 and hasattr(model, "auxiliary_loss")
                else classification_loss.new_tensor(0.0)
            )
            loss_sparsity = (
                sparsity
                if sparsity_mode == "l1"
                else classification_loss.new_tensor(0.0)
            )
            loss = (
                classification_loss
                + current_lambda * regularisation
                + sparsity_lambda * loss_sparsity
                + auxiliary_lambda * auxiliary
            )
            loss.backward()
            optimizer.step()
            if (
                sparsity_mode == "proximal"
                and sparsity_lambda > 0
                and hasattr(model, "proximal_sparsity_step")
            ):
                model.proximal_sparsity_step(
                    optimizer.param_groups[0]["lr"] * sparsity_lambda
                )
            batch_size = labels.numel()
            loss_sum += float(classification_loss.detach()) * batch_size
            reg_sum += float(regularisation.detach()) * batch_size
            sparsity_sum += float(sparsity.detach()) * batch_size
            auxiliary_sum += float(auxiliary.detach()) * batch_size
            correct += int((logits.argmax(1) == labels).sum())
            count += batch_size

        validation = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "train_accuracy": 100.0 * correct / count,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_auxiliary_loss": validation["auxiliary_loss"],
                "regularisation": reg_sum / count,
                "lambda": current_lambda,
                "sparsity": sparsity_sum / count,
                "sparsity_lambda": sparsity_lambda,
                "auxiliary_loss": auxiliary_sum / count,
                "auxiliary_lambda": auxiliary_lambda,
                "learning_rate": current_learning_rate,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"train={100.0 * correct / count:.2f}% "
            f"val={validation['accuracy']:.2f}%"
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    accuracies = [float(row["val_accuracy"]) for row in history]
    aulc_0t = sum(accuracies) / len(accuracies)
    post_update_accuracies = accuracies[1:]
    aulc_1t = sum(post_update_accuracies) / len(post_update_accuracies)
    threshold = 0.95 * max(accuracies)
    epoch_to_95 = next(
        int(row["epoch"])
        for row in history
        if float(row["val_accuracy"]) >= threshold
    )
    metrics = {
        "initial_val_accuracy": initial_val["accuracy"],
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "initial_val_auxiliary_loss": initial_val["auxiliary_loss"],
        "test_auxiliary_loss": test["auxiliary_loss"],
        # ``validation_aulc`` remains as a compatibility alias, but now follows
        # the dissertation's primary post-update definition.
        "validation_aulc": aulc_1t,
        "validation_aulc_0T": aulc_0t,
        "validation_aulc_1T": aulc_1t,
        "epoch_to_95pct_best": epoch_to_95,
        "training_seconds": time.perf_counter() - start,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "state_size_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
        ),
        "lr_scheduler": lr_scheduler,
        "minimum_learning_rate": min_learning_rate,
        "auxiliary_lambda": auxiliary_lambda,
        "sparsity_mode": sparsity_mode,
        "kernel_drift": kernel_drift(model, anchors),
        "assignment_stability": template_assignment_stability(model, anchors),
        "class_group_evidence_gap": class_group_evidence_gap(
            model, test_loader, device
        ),
    }
    if hasattr(model, "sparsity_metrics"):
        metrics["sparsity_metrics"] = model.sparsity_metrics()
    _write_history(output_dir / "history.csv", history)
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(best_state, output_dir / "checkpoint_best.pt")
    torch.save({key: value.cpu() for key, value in anchors.items()}, output_dir / "filters_initial.pt")
    torch.save(
        {
            name: getattr(model, name).weight.detach().cpu()
            for name in anchors
        },
        output_dir / "filters_final.pt",
    )
    return metrics
