"""Training-data-only selection of a domain-informed initialisation.

The selector evaluates candidate initial states on the validation split before
gradient training.  The test loader is deliberately not accepted by this API.
This keeps the adaptive choice auditable and prevents test-set selection.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .initializers import (
    InitialisationReport,
    calibrate_logit_scale,
    initialise_model,
)
from .trainer import evaluate


@dataclass(frozen=True)
class AdaptiveCandidate:
    method: str
    covariance_rank: int = 16
    shrinkage: float = 0.1

    @property
    def name(self) -> str:
        if self.method == "lowrank_wmf":
            return f"{self.method}_rank{self.covariance_rank}"
        return self.method


@dataclass
class AdaptiveSelection:
    selected_candidate: str
    selected_method: str
    selected_covariance_rank: int
    selected_shrinkage: float
    criterion: str
    candidates: list[dict]
    total_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def select_adaptive_initialisation(
    model_factory: Callable[[], nn.Module],
    fit_loader,
    selection_loader,
    device: torch.device,
    candidates: Sequence[AdaptiveCandidate],
    *,
    layers: int,
    seed: int,
    calibrate_logits: bool = True,
    target_logit_std: float = 1.0,
) -> tuple[nn.Module, InitialisationReport, AdaptiveSelection]:
    """Return the best validation-only initial state and its audit trail.

    Candidates share the same random seed so conventionally initialised
    residual parameters do not confound the comparison.  Accuracy is the
    primary criterion and validation loss is the deterministic tie-breaker.
    """

    if not candidates:
        raise ValueError("Adaptive initialisation needs at least one candidate")

    started = time.perf_counter()
    rows = []
    best_key = None
    best_model = None
    best_report = None
    best_candidate = None

    for index, candidate in enumerate(candidates):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = model_factory().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_started = time.perf_counter()
        report = initialise_model(
            model,
            fit_loader,
            candidate.method,
            device,
            layers=layers,
            shrinkage=candidate.shrinkage,
            covariance_rank=candidate.covariance_rank,
        )
        if hasattr(model, "initialise_decoder_from_encoder"):
            model.initialise_decoder_from_encoder()
        if calibrate_logits:
            report.output_scale = calibrate_logit_scale(
                model, fit_loader, device, target_std=target_logit_std
            )
        metrics = evaluate(model, selection_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_seconds = time.perf_counter() - candidate_started
        row = {
            "candidate": candidate.name,
            "method": candidate.method,
            "covariance_rank": candidate.covariance_rank,
            "shrinkage": candidate.shrinkage,
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_loss": float(metrics["loss"]),
            "initialisation_seconds": candidate_seconds,
            "initialisation_report": asdict(report),
        }
        rows.append(row)
        # Earlier candidate order is the final deterministic tie-breaker.
        key = (row["validation_accuracy"], -row["validation_loss"], -index)
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_report = report
            best_candidate = candidate

    assert best_model is not None and best_report is not None
    assert best_candidate is not None
    selection = AdaptiveSelection(
        selected_candidate=best_candidate.name,
        selected_method=best_candidate.method,
        selected_covariance_rank=best_candidate.covariance_rank,
        selected_shrinkage=best_candidate.shrinkage,
        criterion="maximum_epoch0_validation_accuracy_then_minimum_loss",
        candidates=rows,
        total_seconds=time.perf_counter() - started,
    )
    return best_model, best_report, selection
