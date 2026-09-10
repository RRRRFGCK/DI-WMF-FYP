"""Create an illustrative MNIST-to-matched-convolution diagram.

The figure is explanatory only: templates are estimated from the MNIST
training split without gradient optimisation, and no MNIST result is reported
in the dissertation.  It visualises the fit--slide--pool logic used to explain
why a convolution can be read as a bank of matched detectors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
THESIS_OUT = ROOT / "thesis_overleaf_draft" / "figures" / "main"
POSTER_OUT = ROOT / "poster_imperial" / "Figures"

BLUE = "#003E74"
RED = "#B2182B"
TEAL = "#008C95"
GREY = "#5A646E"
LIGHT = "#EEF3F7"


def _arrow(fig, left, right, y: float) -> None:
    fig.add_artist(
        ConnectionPatch(
            xyA=(left.get_position().x1 + 0.004, y),
            xyB=(right.get_position().x0 - 0.004, y),
            coordsA=fig.transFigure,
            coordsB=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color=GREY,
        )
    )


def build_example() -> dict[str, np.ndarray | int]:
    train = MNIST(DATA_ROOT, train=True, download=False)
    test = MNIST(DATA_ROOT, train=False, download=False)
    train_images = train.data.float().div(255.0)
    train_labels = train.targets

    # Compress complete digits to a local 9x9 detector.  Balanced class means
    # and pooled residual variance provide a small diagonal matched-filter bank.
    selected = []
    selected_labels = []
    for class_id in range(10):
        indices = torch.nonzero(train_labels == class_id, as_tuple=False).flatten()[:250]
        selected.append(train_images.index_select(0, indices))
        selected_labels.extend([class_id] * len(indices))
    balanced = torch.cat(selected, dim=0)
    labels = torch.tensor(selected_labels)
    small = F.interpolate(
        balanced[:, None], size=(9, 9), mode="bilinear", align_corners=False
    )[:, 0]
    class_means = torch.stack([small[labels == class_id].mean(0) for class_id in range(10)])
    global_mean = class_means.mean(0)
    residuals = torch.cat(
        [small[labels == class_id] - class_means[class_id] for class_id in range(10)],
        dim=0,
    )
    variance = residuals.square().mean(0)
    variance = 0.9 * variance + 0.1 * variance.mean()
    weights = (class_means - global_mean) / variance.clamp_min(1e-3)
    weights = weights - weights.mean(dim=(1, 2), keepdim=True)
    weights = F.normalize(weights.flatten(1), dim=1).view_as(weights)

    candidates = test.data.float().div(255.0)
    candidate_labels = test.targets
    chosen = None
    for index in torch.nonzero(candidate_labels == 3, as_tuple=False).flatten()[:500]:
        image = candidates[index]
        reduced = F.interpolate(
            image[None, None], size=(14, 14), mode="bilinear", align_corners=False
        )
        maps = F.conv2d(reduced, weights[:, None])
        scores = maps.flatten(2).amax(2)[0]
        if int(scores.argmax()) == 3:
            chosen = (image, reduced[0, 0], maps[0], scores)
            break
    if chosen is None:
        index = torch.nonzero(candidate_labels == 3, as_tuple=False).flatten()[0]
        image = candidates[index]
        reduced = F.interpolate(
            image[None, None], size=(14, 14), mode="bilinear", align_corners=False
        )
        maps = F.conv2d(reduced, weights[:, None])
        chosen = (image, reduced[0, 0], maps[0], maps.flatten(2).amax(2)[0])

    image, reduced, maps, scores = chosen
    target_map = maps[3]
    row, col = np.unravel_index(int(target_map.argmax()), target_map.shape)
    patch = reduced[row : row + 9, col : col + 9]
    return {
        "image": image.numpy(),
        "reduced": reduced.numpy(),
        "patch": patch.numpy(),
        "kernel": weights[3].numpy(),
        "response": target_map.numpy(),
        "scores": scores.numpy(),
        "row": int(row),
        "col": int(col),
        "prediction": int(scores.argmax()),
    }


def wide_figure(example: dict[str, np.ndarray | int]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(12.8, 2.9), gridspec_kw={"wspace": 0.45})
    for axis in axes:
        axis.set_facecolor(LIGHT)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    axes[0].imshow(example["image"], cmap="gray", vmin=0, vmax=1)
    scale = 28 / 14
    axes[0].add_patch(
        Rectangle(
            (example["col"] * scale, example["row"] * scale),
            9 * scale,
            9 * scale,
            fill=False,
            edgecolor=RED,
            linewidth=2.0,
        )
    )
    axes[0].set_title("1  MNIST input", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("digit 3", fontsize=8)

    axes[1].imshow(example["patch"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2  Local patch", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("one receptive field", fontsize=8)

    vmax = float(np.abs(example["kernel"]).max())
    axes[2].imshow(example["kernel"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("3  Fitted detector", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("training-fold digit-3 direction", fontsize=8)

    axes[3].imshow(example["response"], cmap="magma")
    axes[3].scatter(
        [example["col"]], [example["row"]], s=34, facecolors="none", edgecolors="white", lw=1.4
    )
    axes[3].set_title("4  Correlation map", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("slide the same detector", fontsize=8)

    scores = np.asarray(example["scores"])
    scores = scores - scores.min()
    colours = [RED if digit == 3 else "#AEB8C2" for digit in range(10)]
    axes[4].bar(np.arange(10), scores, color=colours, width=0.72)
    axes[4].set_xticks(np.arange(10), [str(i) for i in range(10)], fontsize=7)
    axes[4].tick_params(axis="y", labelsize=7)
    axes[4].spines["left"].set_visible(True)
    axes[4].spines["bottom"].set_visible(True)
    axes[4].set_title("5  Pool class evidence", fontsize=10, fontweight="bold")
    axes[4].set_xlabel(f"largest score: {example['prediction']}", fontsize=8)

    for left, right in zip(axes[:-1], axes[1:]):
        _arrow(fig, left, right, 0.52)

    fig.suptitle(
        "From an image to a CNN decision: fit a detector, slide it, then pool local evidence",
        x=0.5,
        y=1.03,
        fontsize=12.5,
        fontweight="bold",
        color=BLUE,
    )
    fig.text(
        0.5,
        -0.02,
        "Illustrative no-gradient walkthrough; MNIST is used only to explain the operation, not as a reported experiment.",
        ha="center",
        fontsize=8.5,
        color=GREY,
    )
    THESIS_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_OUT / "F25_mnist_cnn_convolution.png", dpi=240, bbox_inches="tight")
    fig.savefig(THESIS_OUT / "F25_mnist_cnn_convolution.pdf", bbox_inches="tight")
    plt.close(fig)


def poster_figure(example: dict[str, np.ndarray | int]) -> None:
    """Poster-scale left-to-right explanation with all four stages in one row."""
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.2), gridspec_kw={"wspace": 0.34})
    panels = [
        (example["image"], "1  Input: digit 3", "gray"),
        (example["kernel"], "2  Matched detector", "RdBu_r"),
        (example["response"], "3  Response map", "magma"),
    ]
    for axis, (array, title, cmap) in zip(axes[:3], panels):
        if cmap == "RdBu_r":
            vmax = float(np.abs(array).max())
            axis.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            axis.imshow(array, cmap=cmap)
        axis.set_title(title, fontsize=22, fontweight="bold", pad=12)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    scores = np.asarray(example["scores"])
    scores = scores - scores.min()
    colours = [RED if digit == 3 else "#AEB8C2" for digit in range(10)]
    axes[3].bar(np.arange(10), scores, color=colours, width=0.75)
    axes[3].set_xticks(np.arange(10), [str(i) for i in range(10)], fontsize=16)
    axes[3].tick_params(axis="y", labelsize=16)
    axes[3].set_title("4  Class evidence", fontsize=22, fontweight="bold", pad=12)
    axes[3].spines[["top", "right"]].set_visible(False)
    for x in (0.255, 0.505, 0.755):
        fig.text(x, 0.43, r"$\rightarrow$", color=BLUE, fontsize=30,
                 ha="center", va="center", fontweight="bold")
    fig.suptitle(r"Fit a detector  $\rightarrow$  slide it  $\rightarrow$  pool class evidence",
                 color=BLUE, fontweight="bold", fontsize=26, y=0.965)
    fig.text(0.5, 0.055, "Illustration only - no MNIST result is claimed",
             ha="center", color=GREY, fontsize=18)
    fig.subplots_adjust(left=0.025, right=0.99, top=0.72, bottom=0.22)
    POSTER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER_OUT / "poster_mnist_cnn_flow.png", dpi=260)
    fig.savefig(POSTER_OUT / "poster_mnist_cnn_flow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    sample = build_example()
    wide_figure(sample)
    poster_figure(sample)
    print(f"Saved MNIST explanatory figures; illustrative prediction={sample['prediction']}")
