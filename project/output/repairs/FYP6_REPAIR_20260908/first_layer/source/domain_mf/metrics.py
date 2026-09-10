"""Shared learning-curve metric definitions.

The dissertation treats post-update AULC as the primary learning-speed metric.
Keeping the calculation here prevents analysis scripts from silently mixing the
inclusive Epoch-0 average with the post-update average.
"""

from __future__ import annotations

import csv
from pathlib import Path


def validation_aulcs(history_path: Path) -> tuple[float, float]:
    """Return ``(AULC_0T, AULC_1T)`` from a run history.

    Both quantities are discrete epoch averages.  ``AULC_0T`` includes the
    pre-optimisation evaluation, whereas ``AULC_1T`` contains only epochs after
    at least one gradient-training epoch.
    """

    with Path(history_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training history: {history_path}")
    inclusive = [float(row["val_accuracy"]) for row in rows]
    post_update = [
        float(row["val_accuracy"]) for row in rows if int(row["epoch"]) >= 1
    ]
    # An epochs=0 diagnostic has a valid pre-optimisation value but no learning
    # trajectory. Represent AULC_1T as undefined instead of silently treating
    # Epoch 0 as a training epoch.
    post_aulc = sum(post_update) / len(post_update) if post_update else float("nan")
    return sum(inclusive) / len(inclusive), post_aulc
