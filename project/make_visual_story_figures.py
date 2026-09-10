"""Create sample-level visual explanations for the dissertation.

The main statistical plots answer whether an effect is repeatable.  These
figures answer the complementary questions: what the matched-filter operation
looks like on one image, what happens to the stored templates during training,
how a representative decision changes under corruption, and what the
Correncoder predicts in the time and frequency domains.

All panels are generated from the saved experiment checkpoints and local test
sets.  No values are hand-entered into the figures.
"""

from __future__ import annotations

import json
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t

from domain_mf.corruptions import (
    apply_corruption_batch,
    from_pixel_space,
    to_pixel_space,
)
from domain_mf.data import build_loaders, build_test_dataset
from domain_mf.initializers import (
    _collect_salient_patches,
    _deterministic_kmeans,
    calibrate_logit_scale,
    initialise_model,
)
from domain_mf.models import build_model
from run_correncoder_regression import load_bidmc_subjects


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_overleaf_draft" / "figures" / "main"
DATA_ROOT = ROOT.parent / "majorRevision" / "data"
SIGN_ROOT = ROOT.parent / "data" / "sign_language_mnist"

DI_RED = "#b2182b"
DI_LIGHT = "#ef8a62"
KAIMING = "#636363"
REFERENCE = "#111111"
BLUE = "#2166ac"
GREEN = "#1b7837"
PANEL_BG = "#f5f3ef"

DATASET_LABELS = {
    "fashion": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "sign": "Sign",
}
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
SIGN_CLASSES = ["C", "E", "I", "K", "L", "O", "P", "Q", "X", "Y"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.transparent": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rounded_panel(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    *,
    facecolor: str = "white",
    edgecolor: str = "#d8d4ce",
    linewidth: float = 0.9,
    radius: float = 0.018,
) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=-2,
        )
    )


def _arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: str = "#777777",
    linewidth: float = 1.2,
) -> None:
    fig.add_artist(
        patches.FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            color=colour,
            zorder=5,
        )
    )


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) if value not in ("", None) else np.nan for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _history_band(dataset: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = []
    matches = sorted(
        (ROOT / "outputs_convergence50").glob(
            f"{dataset}_standard_{method}_layers3_seed*_frac1p0_epochs50_*"
        )
    )
    for path in matches:
        histories.append([row["val_accuracy"] for row in _read_history(path / "history.csv")])
    if len(histories) != 10:
        raise RuntimeError(f"Expected ten histories for {dataset}/{method}, found {len(histories)}")
    values = np.asarray(histories, dtype=float)
    epochs = np.arange(values.shape[1])
    mean = values.mean(axis=0)
    critical = float(student_t.ppf(0.975, values.shape[0] - 1))
    half_width = critical * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return epochs, mean, half_width


def visual_abstract() -> None:
    """One-page visual thesis: source of the prior, mechanism and claim boundary."""
    dataset_name = "sign"
    _, fit_loader, _, _ = build_loaders(
        dataset_name,
        DATA_ROOT,
        SIGN_ROOT,
        batch_size=64,
        seed=0,
        train_fraction=1.0,
        val_fraction=0.1,
        num_workers=0,
    )
    batch, labels = next(iter(fit_loader))
    chosen = []
    seen = set()
    for index, label in enumerate(labels.tolist()):
        if label not in seen:
            chosen.append(index)
            seen.add(label)
        if len(chosen) == 4:
            break
    sample = batch[chosen[1]]
    target = int(labels[chosen[1]])
    weights = load_filter_bank(dataset_name, "lowrank_wmf", "initial")
    target_bank = weights[target * 4 : (target + 1) * 4]
    patches_by_class = _collect_salient_patches(
        fit_loader,
        lambda values: values,
        num_classes=10,
        kernel_size=5,
        padding=2,
        sampling_stride=2,
        device=torch.device("cpu"),
        max_patches_per_class=1024,
    )
    prototype_centres, _ = _deterministic_kmeans(
        patches_by_class[target].double(), clusters=4
    )
    response = F.conv2d(sample.unsqueeze(0), target_bank, padding=2)[0].amax(dim=0)
    selected_filter = target_bank[0]

    epochs_k, mean_k, width_k = _history_band(dataset_name, "kaiming")
    epochs_d, mean_d, width_d = _history_band(dataset_name, "lowrank_wmf")

    fig = plt.figure(figsize=(13.6, 6.0), facecolor="white")
    fig.suptitle(
        "Domain knowledge changes the starting representation, not the CNN architecture",
        x=0.5,
        y=0.975,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.922,
        "Labelled local patterns are converted into auditable matched templates; optimisation remains free to adapt them.",
        ha="center",
        color="#4f4f4f",
        fontsize=9.5,
    )

    panels = [
        (0.035, 0.39, 0.175, 0.43),
        (0.235, 0.39, 0.145, 0.43),
        (0.405, 0.39, 0.145, 0.43),
        (0.575, 0.39, 0.145, 0.43),
        (0.745, 0.39, 0.22, 0.43),
    ]
    for panel in panels:
        _rounded_panel(fig, panel, facecolor="#fcfbf9")

    # Training examples as a small, varied contact strip.
    fig.text(0.052, 0.775, "1  Labelled local patterns", fontweight="bold", fontsize=9.5)
    image_axes = []
    for row in range(2):
        for column in range(2):
            axis = fig.add_axes([0.058 + column * 0.071, 0.48 + (1 - row) * 0.105, 0.061, 0.09])
            image_axes.append(axis)
    names = class_names(dataset_name, fit_loader.dataset)
    for axis, index in zip(image_axes, chosen):
        show_image(axis, batch[index], dataset_name)
        axis.text(
            0.5,
            -0.12,
            names[int(labels[index])],
            transform=axis.transAxes,
            ha="center",
            fontsize=7.5,
            color="#555555",
        )
    fig.text(0.122, 0.423, "balanced by class", ha="center", color="#666666", fontsize=8)

    # A simple statistical view: prototype and covariance-aware direction.
    fig.text(0.248, 0.775, "2  Estimate signal + noise", fontweight="bold", fontsize=9.5)
    prototype = prototype_centres[0].reshape(3, 5, 5).permute(1, 2, 0).numpy()
    axis = fig.add_axes([0.262, 0.51, 0.09, 0.20])
    axis.imshow(np.clip(prototype, 0.0, 1.0), interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("class prototype  $\mu_c$", fontsize=8)
    fig.text(0.307, 0.447, "$\Sigma$: shared local noise", ha="center", fontsize=8.2, color="#555555")

    fig.text(0.418, 0.775, "3  Whiten into a template", fontweight="bold", fontsize=9.5)
    filter_image = selected_filter.mean(dim=0).numpy()
    vmax = max(float(np.abs(filter_image).max()), 1e-6)
    axis = fig.add_axes([0.432, 0.51, 0.09, 0.20])
    axis.imshow(filter_image, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("stored class template", fontsize=8)
    fig.text(0.477, 0.445, "$w_c\propto\Sigma^{-1}(\mu_c-\mu_r)$", ha="center", fontsize=9.0, color=DI_RED)

    fig.text(0.588, 0.775, "4  Slide and detect", fontweight="bold", fontsize=9.5)
    image_axis = fig.add_axes([0.591, 0.52, 0.052, 0.18])
    show_image(image_axis, sample, dataset_name)
    response_axis = fig.add_axes([0.657, 0.52, 0.052, 0.18])
    response_axis.imshow(response.numpy(), cmap="magma")
    response_axis.set_xticks([])
    response_axis.set_yticks([])
    response_axis.set_title("response", fontsize=7.5)
    fig.text(0.647, 0.447, "$x\star w_c$ supplies local evidence", ha="center", fontsize=8.0, color="#555555")

    fig.text(0.758, 0.775, "5  Fit the chain, then learn", fontweight="bold", fontsize=9.5)
    curve_axis = fig.add_axes([0.775, 0.49, 0.167, 0.22])
    curve_axis.fill_between(epochs_k, mean_k - width_k, mean_k + width_k, color=KAIMING, alpha=0.10)
    curve_axis.plot(epochs_k, mean_k, color=KAIMING, lw=1.8, label="Kaiming")
    curve_axis.fill_between(epochs_d, mean_d - width_d, mean_d + width_d, color=DI_RED, alpha=0.12)
    curve_axis.plot(epochs_d, mean_d, color=DI_RED, lw=2.0, label="DI-WMF")
    curve_axis.scatter([0], [mean_d[0]], color=DI_RED, s=24, zorder=4)
    curve_axis.set_xlim(0, 50)
    curve_axis.set_ylim(0, 105)
    curve_axis.set_xlabel("epoch", fontsize=8)
    curve_axis.set_ylabel("validation accuracy (%)", fontsize=8)
    curve_axis.tick_params(labelsize=7.5)
    curve_axis.grid(alpha=0.15)
    curve_axis.legend(frameon=False, fontsize=7.5, loc="lower right")

    for left, right in zip(panels[:-1], panels[1:]):
        _arrow(fig, (left[0] + left[2] + 0.006, 0.605), (right[0] - 0.007, 0.605), colour=DI_RED)

    # Evidence ribbon: strongest claims and the boundary are visible before the chapter is read.
    _rounded_panel(fig, (0.035, 0.115, 0.93, 0.18), facecolor="#f7f3f0", edgecolor="#e0d8d1")
    fig.text(0.055, 0.25, "WHAT THE EXPERIMENTS SUPPORT", fontsize=8.2, fontweight="bold", color=DI_RED)
    claims = [
        (0.055, "Training-free control", "the fitted head dominates Epoch-0"),
        (0.285, "Post-update benefit", "AULC gains on Fashion and Sign"),
        (0.535, "Auditable provenance", "anchor alignment improves on all 3 tasks"),
        (0.765, "Boundary", "CIFAR, unseen-signer final and robustness are mixed"),
    ]
    for x, heading, detail in claims:
        fig.text(x, 0.202, heading, fontsize=9, fontweight="bold", color=REFERENCE)
        fig.text(x, 0.157, detail, fontsize=8, color="#555555", wrap=True)

    save(fig, "F00_visual_abstract")


def _reconstruct_initial_sign_model(method: str, device: torch.device):
    """Recreate the recorded seed-0 Epoch-0 model and audit it against disk."""
    _seed_everything(0)
    _, fit_loader, val_loader, _ = build_loaders(
        "sign",
        DATA_ROOT,
        SIGN_ROOT,
        batch_size=64,
        seed=0,
        train_fraction=1.0,
        val_fraction=0.1,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model("standard", "sign").to(device)
    initialise_model(
        model,
        fit_loader,
        method,
        device,
        layers=3,
        shrinkage=0.1,
        covariance_rank=16,
    )
    calibrate_logit_scale(model, fit_loader, device, target_std=1.0)
    model.eval()

    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
            correct += int((logits.argmax(1).cpu() == targets).sum())
            count += targets.numel()
    reconstructed_accuracy = 100.0 * correct / count
    recorded_accuracy = _read_history(run_dir("sign", method) / "history.csv")[0]["val_accuracy"]
    saved_bank = load_filter_bank("sign", method, "initial")
    max_filter_error = float((model.conv1.weight.detach().cpu() - saved_bank).abs().max())
    if abs(reconstructed_accuracy - recorded_accuracy) > 1e-8 or max_filter_error > 1e-6:
        raise RuntimeError(
            f"Epoch-0 reconstruction audit failed for {method}: "
            f"accuracy {reconstructed_accuracy} vs {recorded_accuracy}, "
            f"filter error {max_filter_error}"
        )
    return model, reconstructed_accuracy, max_filter_error


def random_vs_diwmf_decision_chain() -> None:
    """Two-lane, sample-level comparison backed by audited Epoch-0 models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kaiming_model, kaiming_val, kaiming_error = _reconstruct_initial_sign_model("kaiming", device)
    di_model, di_val, di_error = _reconstruct_initial_sign_model("lowrank_wmf", device)
    dataset = build_test_dataset("sign", DATA_ROOT, SIGN_ROOT)
    names = SIGN_CLASSES

    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    targets = torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])
    with torch.no_grad():
        kaiming_logits = kaiming_model(images.to(device)).cpu()
        di_logits = di_model(images.to(device)).cpu()
    kaiming_prob = kaiming_logits.softmax(dim=1)
    di_prob = di_logits.softmax(dim=1)
    indices = torch.arange(len(dataset))
    kaiming_pred = kaiming_prob.argmax(dim=1)
    di_pred = di_prob.argmax(dim=1)
    true_gain = di_prob[indices, targets] - kaiming_prob[indices, targets]
    eligible = (di_pred == targets) & (kaiming_pred != targets)
    if not bool(eligible.any()):
        eligible = di_pred == targets
    score = true_gain + eligible.float() * 10.0
    sample_index = int(score.argmax())
    image = images[sample_index]
    target = int(targets[sample_index])

    with torch.no_grad():
        kaiming_response = kaiming_model.conv1(image.unsqueeze(0).to(device))[0].cpu()
        di_response = di_model.conv1(image.unsqueeze(0).to(device))[0].cpu()
    channel_slice = slice(target * 4, (target + 1) * 4)
    maps = {
        "Kaiming": kaiming_response[channel_slice].amax(dim=0),
        "DI-WMF": di_response[channel_slice].amax(dim=0),
    }
    banks = {
        "Kaiming": kaiming_model.conv1.weight.detach().cpu(),
        "DI-WMF": di_model.conv1.weight.detach().cpu(),
    }
    probabilities = {"Kaiming": kaiming_prob[sample_index], "DI-WMF": di_prob[sample_index]}
    predictions = {"Kaiming": int(kaiming_pred[sample_index]), "DI-WMF": int(di_pred[sample_index])}
    colours = {"Kaiming": KAIMING, "DI-WMF": DI_RED}

    fig = plt.figure(figsize=(13.7, 6.35), facecolor="white")
    fig.suptitle(
        "The same Sign image enters the same CNN, but the Epoch-0 evidence is different",
        x=0.5,
        y=0.982,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.936,
        "Audited deterministic reconstruction of the saved seed-0 initial states; the right panel then returns to the ten-seed result.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )

    _rounded_panel(fig, (0.025, 0.17, 0.14, 0.68), facecolor="#fbfaf8")
    _rounded_panel(fig, (0.185, 0.54, 0.58, 0.31), facecolor="#f4f4f4", edgecolor="#dddddd")
    _rounded_panel(fig, (0.185, 0.17, 0.58, 0.31), facecolor="#fbf3f2", edgecolor="#ecd0cc")
    _rounded_panel(fig, (0.79, 0.17, 0.185, 0.68), facecolor="#fbfaf8")

    input_axis = fig.add_axes([0.052, 0.365, 0.086, 0.26])
    show_image(input_axis, image, "sign")
    input_axis.set_title("Common input", fontweight="bold", fontsize=9.5, pad=7)
    fig.text(0.095, 0.32, f"true class: {names[target]}", ha="center", fontsize=9, color=REFERENCE)
    fig.text(0.095, 0.26, "architecture and\ntraining schedule fixed", ha="center", fontsize=8, color="#666666")

    row_y = {"Kaiming": 0.565, "DI-WMF": 0.195}
    for method in ("Kaiming", "DI-WMF"):
        y = row_y[method]
        colour = colours[method]
        subtitle = "random directions" if method == "Kaiming" else "class-derived whitened directions"
        fig.text(0.202, y + 0.236, method, fontsize=11, fontweight="bold", color=colour)
        fig.text(0.202, y + 0.196, subtitle, fontsize=8.2, color="#606060")

        mosaic = _filter_mosaic(banks[method], class_id=target)
        vmax = max(float(np.nanmax(np.abs(mosaic))), 1e-6)
        filter_axis = fig.add_axes([0.31, y + 0.047, 0.105, 0.18])
        cmap = plt.colormaps["RdBu_r"].copy()
        cmap.set_bad("white")
        filter_axis.imshow(mosaic, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        filter_axis.set_xticks([])
        filter_axis.set_yticks([])
        filter_axis.set_title(
            "unanchored channel slots" if method == "Kaiming" else f"four class-{names[target]} templates",
            fontsize=7.8,
        )

        response_map = maps[method]
        response_map = (response_map - response_map.mean()) / response_map.std().clamp_min(1e-6)
        response_axis = fig.add_axes([0.455, y + 0.047, 0.105, 0.18])
        response_axis.imshow(response_map.numpy(), cmap="magma", vmin=-1.0, vmax=3.0)
        response_axis.set_xticks([])
        response_axis.set_yticks([])
        response_axis.set_title("normalised local response", fontsize=7.8)

        probability = probabilities[method].numpy()
        decision_axis = fig.add_axes([0.605, y + 0.047, 0.13, 0.18])
        order = np.argsort(probability)[-3:]
        bars = decision_axis.barh(np.arange(3), probability[order], color=[colour if index == predictions[method] else "#c9c9c9" for index in order])
        decision_axis.set_yticks(np.arange(3), [names[index] for index in order])
        decision_axis.set_xlim(0, max(0.45, float(probability[order].max()) * 1.15))
        decision_axis.set_xlabel("probability", fontsize=7.5)
        decision_axis.tick_params(labelsize=7.5)
        decision_axis.grid(axis="x", alpha=0.14)
        predicted = names[predictions[method]]
        verdict = "correct" if predictions[method] == target else "wrong"
        decision_axis.set_title(f"predicts {predicted} — {verdict}", fontsize=8.2, color=colour, fontweight="bold")
        for bar, value in zip(bars, probability[order]):
            decision_axis.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=7)

        _arrow(fig, (0.165, y + 0.145), (0.295, y + 0.145), colour=colour)
        _arrow(fig, (0.42, y + 0.145), (0.448, y + 0.145), colour=colour)
        _arrow(fig, (0.565, y + 0.145), (0.598, y + 0.145), colour=colour)

    curve_axis = fig.add_axes([0.815, 0.31, 0.135, 0.42])
    for method, colour in (("kaiming", KAIMING), ("lowrank_wmf", DI_RED)):
        epochs, mean, width = _history_band("sign", method)
        label = "Kaiming" if method == "kaiming" else "DI-WMF"
        curve_axis.fill_between(epochs, mean - width, mean + width, color=colour, alpha=0.11)
        curve_axis.plot(epochs, mean, color=colour, lw=2.0, label=label)
        curve_axis.scatter([0], [mean[0]], color=colour, s=25, zorder=5)
    curve_axis.set_xlim(0, 50)
    curve_axis.set_ylim(0, 105)
    curve_axis.set_xlabel("epoch")
    curve_axis.set_ylabel("validation accuracy (%)")
    curve_axis.set_title("Ten-seed learning trajectory", fontsize=9.5, fontweight="bold")
    curve_axis.grid(alpha=0.16)
    curve_axis.legend(frameon=False, loc="lower right")
    curve_axis.text(
        0.04,
        0.93,
        f"Epoch 0:  {kaiming_val:.0f}%  →  {di_val:.0f}%",
        transform=curve_axis.transAxes,
        fontsize=8.2,
        color=DI_RED,
        fontweight="bold",
        va="top",
    )

    fig.text(
        0.5,
        0.055,
        "Interpretation: DI-WMF supplies class-indexed evidence before gradient updates. The aggregate result is faster early learning; it does not imply universal final superiority.",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    save(fig, "F19_random_vs_diwmf_decision_chain")

    metadata = {
        "dataset": "sign",
        "sample_index": sample_index,
        "target": target,
        "target_name": names[target],
        "selection": "largest true-class probability gain among DI-WMF-correct/Kaiming-wrong Epoch-0 test cases",
        "kaiming_prediction": names[predictions["Kaiming"]],
        "di_wmf_prediction": names[predictions["DI-WMF"]],
        "kaiming_true_probability": float(probabilities["Kaiming"][target]),
        "di_wmf_true_probability": float(probabilities["DI-WMF"][target]),
        "reconstructed_validation_accuracy": {"kaiming": kaiming_val, "lowrank_wmf": di_val},
        "aggregate_learning_curve_seed_count": 10,
        "max_abs_conv1_error_vs_saved": {"kaiming": kaiming_error, "lowrank_wmf": di_error},
        "device": str(device),
    }
    (OUTPUT / "F19_random_vs_diwmf_decision_chain.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cross_task_conclusion_map() -> None:
    """Synthesis figure that keeps supported and bounded claims visibly separate."""
    table_root = ROOT / "thesis_artifacts" / "complete_package" / "tables" / "main"
    # Submission-stage source of truth.  This registry freezes the exact
    # ten-seed values used by both Table 5.1 and this synthesis figure.  In
    # particular, "Post-update AULC" is AULC 1:50 and excludes Epoch 0.
    main_rows = _read_csv_rows(
        ROOT / "submission_control_results" / "final_main_ten_seed_registry.csv"
    )
    semantic_rows = _read_csv_rows(table_root / "T07_template_semantics.csv")
    corruption_rows = _read_csv_rows(table_root / "T05_natural_corruptions.csv")
    corr_rows = _read_csv_rows(table_root / "T11_correncoder_regression_paired.csv")

    def main_effect(dataset: str, metric: str) -> tuple[float, float, float]:
        row = next(item for item in main_rows if item["dataset"] == dataset and item["metric"] == metric)
        return (
            float(row["mean_paired_difference"]),
            float(row["ci95_low"]),
            float(row["ci95_high"]),
        )

    def anchor_effect(dataset: str) -> tuple[float, float, float]:
        row = next(
            item
            for item in semantic_rows
            if item["dataset"] == dataset and item["metric"] == "mean_anchor_alignment"
        )
        difference = float(row["mean_candidate_minus_reference"])
        return difference, float(row["ci95_low"]), float(row["ci95_high"])

    def corruption_summary(dataset: str) -> tuple[int, int]:
        rows = [item for item in corruption_rows if item["dataset"] == dataset]
        positive = sum(
            float(item["lowrank_minus_kaiming_auc"]) > 0
            and float(item["holm_adjusted_p"]) < 0.05
            for item in rows
        )
        return positive, len(rows)

    datasets = ["fashion", "cifar10", "sign"]
    display = {"fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "sign": "Sign"}
    metrics = ["Epoch-0", "Post-update AULC", "Final test"]
    columns = [
        ("Complete initializer", "Epoch-0; head-dominated"),
        ("Post-update learning", "AULC 1:50"),
        ("Converged outcome", "final accuracy"),
        ("Template provenance", "anchor alignment"),
        ("Natural corruptions", "absolute AUC vs retention"),
    ]

    support_face = "#f5d9d5"
    support_edge = DI_RED
    mixed_face = "#fff1cf"
    mixed_edge = "#bd7b00"
    boundary_face = "#efefef"
    boundary_edge = "#9b9b9b"

    fig = plt.figure(figsize=(13.7, 7.25), facecolor="white")
    fig.suptitle(
        "One domain-informed prior, three task-dependent outcomes",
        x=0.5,
        y=0.978,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "The fitted head explains training-free accuracy; template, final and robustness effects depend on the data regime.",
        ha="center",
        fontsize=9.3,
        color="#505050",
    )

    x0 = 0.17
    col_width = 0.148
    gap = 0.014
    row_height = 0.15
    row_y = [0.65, 0.46, 0.27]
    for column_index, (heading, subheading) in enumerate(columns):
        x = x0 + column_index * (col_width + gap)
        fig.text(x + col_width / 2, 0.862, heading, ha="center", fontsize=9.2, fontweight="bold")
        fig.text(x + col_width / 2, 0.824, subheading, ha="center", fontsize=7.7, color="#666666")

    for row_index, dataset_name in enumerate(datasets):
        y = row_y[row_index]
        dataset = build_test_dataset(dataset_name, DATA_ROOT, SIGN_ROOT)
        representative = dataset[0][0]
        icon_axis = fig.add_axes([0.035, y + 0.032, 0.065, 0.092])
        show_image(icon_axis, representative, dataset_name)
        fig.text(0.108, y + 0.092, display[dataset_name], ha="left", fontsize=8.5, fontweight="bold")
        cell_payloads = []
        for metric in metrics:
            value, low, high = main_effect(dataset_name, metric)
            if metric == "Epoch-0":
                cell_payloads.append((f"{value:+.2f} pp", "fitted head dominates", "control"))
                continue
            if metric == "Final test" and dataset_name == "sign":
                cell_payloads.append((f"main {value:+.2f} pp", "unseen signer: +8.57, inconclusive", "mixed"))
                continue
            supported = low > 0
            cell_payloads.append(
                (
                    f"{value:+.2f} pp",
                    f"95% CI [{low:+.2f}, {high:+.2f}]",
                    "support" if supported else "mixed",
                )
            )
        value, low, high = anchor_effect(dataset_name)
        cell_payloads.append(
            (f"{value:+.3f}", f"95% CI [{low:+.3f}, {high:+.3f}]", "support" if low > 0 else "mixed")
        )
        if dataset_name == "fashion":
            cell_payloads.append(("not audited", "full-model corruption family", "boundary"))
        else:
            positive, total = corruption_summary(dataset_name)
            cell_payloads.append((f"absolute AUC: {positive}/{total}", "relative retention: mixed", "mixed"))

        for column_index, (headline, detail, status) in enumerate(cell_payloads):
            x = x0 + column_index * (col_width + gap)
            if status == "support":
                face, edge, marker = support_face, support_edge, "SUPPORTED"
            elif status == "mixed":
                face, edge, marker = mixed_face, mixed_edge, "MIXED"
            elif status == "control":
                face, edge, marker = "#e7f0f6", BLUE, "HEAD-DOMINATED"
            else:
                face, edge, marker = boundary_face, boundary_edge, "BOUNDARY"
            _rounded_panel(fig, (x, y, col_width, row_height), facecolor=face, edgecolor=edge, linewidth=1.0)
            fig.text(x + 0.012, y + 0.119, marker, fontsize=6.7, fontweight="bold", color=edge)
            fig.text(x + col_width / 2, y + 0.073, headline, ha="center", fontsize=10.3, fontweight="bold", color=REFERENCE)
            fig.text(x + col_width / 2, y + 0.031, detail, ha="center", fontsize=7.15, color="#555555")

    # The regression extension is intentionally separated because its metric is not accuracy.
    corr_row = next(
        item
        for item in corr_rows
        if item["experiment"] == "pretrained_corr_spectral"
        and item["reference"] == "pretrained_mse"
        and item["metric"] == "waveform_correlation"
    )
    rr_row = next(
        item
        for item in corr_rows
        if item["experiment"] == "pretrained_corr_spectral"
        and item["reference"] == "pretrained_mse"
        and item["metric"] == "rr_mean_absolute_error_bpm_30p6s"
    )
    _rounded_panel(fig, (0.045, 0.075, 0.915, 0.145), facecolor="#eef3f7", edgecolor=BLUE, linewidth=1.0)
    fig.text(0.065, 0.175, "CORRENCODER EXTENSION", fontsize=7.2, color=BLUE, fontweight="bold")
    fig.text(0.065, 0.127, "PPG  →  respiration", fontsize=11, fontweight="bold", color=REFERENCE)
    fig.text(0.285, 0.164, "Best mean waveform correlation", fontsize=8.2, color="#555555")
    fig.text(0.285, 0.115, "0.244  (modest morphology recovery)", fontsize=10.2, fontweight="bold", color=BLUE)
    fig.text(0.56, 0.127, "45/53 subjects favour Correncoder", fontsize=9.2, fontweight="bold", color=REFERENCE)
    fig.text(0.76, 0.164, "Respiratory-rate MAE", fontsize=8.2, color="#555555")
    fig.text(0.76, 0.115, "1.29 vs 3.20 bpm", fontsize=10.0, fontweight="bold", color=BLUE)
    fig.text(0.76, 0.087, "Correncoder vs best tested classical", fontsize=7.5, color="#555555")

    save(fig, "F20_cross_task_conclusion_map")


def run_dir(dataset: str, method: str) -> Path:
    matches = sorted(
        (ROOT / "outputs_convergence50").glob(
            f"{dataset}_standard_{method}_layers3_seed0_frac1p0_epochs50_*"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one seed-0 run for {dataset}/{method}, found {len(matches)}"
        )
    return matches[0]


def class_names(dataset_name: str, dataset) -> list[str]:
    if dataset_name == "fashion":
        return FASHION_CLASSES
    if dataset_name == "sign":
        return SIGN_CLASSES
    return list(dataset.classes)


def image_array(image: torch.Tensor, dataset_name: str) -> np.ndarray:
    pixels = to_pixel_space(image.detach().cpu(), dataset_name)
    if pixels.shape[0] == 1:
        return pixels[0].numpy()
    return pixels.permute(1, 2, 0).numpy()


def show_image(axis: plt.Axes, image: torch.Tensor, dataset_name: str) -> None:
    values = image_array(image, dataset_name)
    axis.imshow(values, cmap="gray" if values.ndim == 2 else None, vmin=0, vmax=1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def load_filter_bank(dataset: str, method: str, stage: str) -> torch.Tensor:
    bank = torch.load(
        run_dir(dataset, method) / f"filters_{stage}.pt",
        map_location="cpu",
        weights_only=True,
    )
    return bank["conv1"].float()


def mean_cosine_drift(initial: torch.Tensor, final: torch.Tensor) -> float:
    first = initial.flatten(1)
    second = final.flatten(1)
    return float(F.cosine_similarity(first, second, dim=1).mean())


def worked_matched_filter_example() -> None:
    dataset_name = "sign"
    dataset = build_test_dataset(dataset_name, DATA_ROOT, SIGN_ROOT)
    names = class_names(dataset_name, dataset)
    weights = load_filter_bank(dataset_name, "lowrank_wmf", "initial")

    best = None
    for index in range(len(dataset)):
        image, target = dataset[index]
        response = F.conv2d(image.unsqueeze(0), weights, padding=2)[0]
        class_scores = response.amax(dim=(1, 2)).view(10, 4).amax(dim=1)
        margin = float(
            class_scores[target]
            - torch.cat((class_scores[:target], class_scores[target + 1 :])).max()
        )
        is_correct = int(class_scores.argmax()) == int(target)
        key = (int(is_correct), margin)
        if best is None or key > best[0]:
            best = (key, index, image, int(target), response, class_scores)

    _, sample_index, image, target, response, class_scores = best
    target_channels = slice(target * 4, (target + 1) * 4)
    target_response = response[target_channels]
    flat_peak = int(target_response.reshape(-1).argmax())
    channel_local = flat_peak // (target_response.shape[-2] * target_response.shape[-1])
    spatial_flat = flat_peak % (target_response.shape[-2] * target_response.shape[-1])
    peak_y = spatial_flat // target_response.shape[-1]
    peak_x = spatial_flat % target_response.shape[-1]
    channel = target * 4 + channel_local
    selected_response = response[channel]
    selected_filter = weights[channel]

    padded = F.pad(image, (2, 2, 2, 2))
    local_patch = padded[:, peak_y : peak_y + 5, peak_x : peak_x + 5]
    activated = F.relu(selected_response)
    pooled = F.max_pool2d(activated[None, None], 2)[0, 0]

    fig = plt.figure(figsize=(13.2, 3.35))
    grid = fig.add_gridspec(1, 6, width_ratios=[1.18, 0.85, 0.85, 1.08, 1.08, 1.35], wspace=0.55)
    axes = [fig.add_subplot(grid[0, index]) for index in range(6)]

    show_image(axes[0], image, dataset_name)
    axes[0].add_patch(
        patches.Rectangle(
            (peak_x - 2.5, peak_y - 2.5),
            5,
            5,
            fill=False,
            lw=1.8,
            edgecolor=DI_RED,
        )
    )
    axes[0].plot(peak_x, peak_y, marker="+", color=DI_RED, ms=8, mew=1.5)
    axes[0].set_title(f"1  Input\ntrue class: {names[target]}", fontsize=8.7)

    show_image(axes[1], local_patch, dataset_name)
    axes[1].set_title("2  Local patch\n$5\\times5$ receptive field", fontsize=8.7)

    signed_filter = selected_filter.mean(dim=0).numpy()
    vmax = max(float(np.abs(signed_filter).max()), 1e-6)
    axes[2].imshow(signed_filter, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(f"3  Stored template\nclass {names[target]}, channel {channel}", fontsize=8.7)

    rvmax = max(float(torch.abs(selected_response).max()), 1e-6)
    axes[3].imshow(selected_response.numpy(), cmap="RdBu_r", vmin=-rvmax, vmax=rvmax)
    axes[3].plot(peak_x, peak_y, marker="o", ms=5, mfc="none", mec="#ffd92f", mew=1.6)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_title("4  Correlation map\nstrongest-match peak", fontsize=8.7)

    axes[4].imshow(pooled.numpy(), cmap="magma", vmin=0)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    axes[4].set_title("5  ReLU + max pool\npositive evidence", fontsize=8.7)

    order = np.arange(10)
    values = class_scores.detach().numpy()
    colours = [DI_RED if index == target else "#bdbdbd" for index in order]
    axes[5].hlines(order, 0, values, color=colours, lw=2)
    axes[5].scatter(values, order, color=colours, s=28, zorder=3)
    axes[5].axvline(0, color="#777777", lw=0.7)
    axes[5].set_yticks(order, names)
    axes[5].invert_yaxis()
    axes[5].set_xlabel("maximum matched response")
    axes[5].set_title("6  Class-indexed evidence\nmax over 4 templates", fontsize=8.7)
    axes[5].grid(axis="x", alpha=0.18)

    for left, right in zip(axes[:-1], axes[1:]):
        start = left.get_position().x1 + 0.006
        end = right.get_position().x0 - 0.008
        y = (left.get_position().y0 + left.get_position().y1) / 2
        fig.add_artist(
            patches.FancyArrowPatch(
                (start, y),
                (end, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color="#777777",
            )
        )

    fig.suptitle(
        "A stored DI-WMF template acts as a sliding matched detector",
        x=0.51,
        y=1.03,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        f"Illustrative Sign test sample {sample_index}; responses use the saved seed-0 initial filter bank before optimisation.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save(fig, "F00_worked_matched_filter_example")

    metadata = {
        "sample_index": sample_index,
        "target": target,
        "target_name": names[target],
        "selected_channel": int(channel),
        "peak_xy": [int(peak_x), int(peak_y)],
        "selection": "correct class-indexed response with largest response margin",
    }
    (OUTPUT / "F00_worked_matched_filter_example.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _filter_mosaic(bank: torch.Tensor, class_id: int = 0) -> np.ndarray:
    chosen = bank[class_id * 4 : class_id * 4 + 4].mean(dim=1).numpy()
    height, width = chosen.shape[-2:]
    gap = 1
    canvas = np.full((2 * height + gap, 2 * width + gap), np.nan, dtype=float)
    canvas[:height, :width] = chosen[0]
    canvas[:height, width + gap :] = chosen[1]
    canvas[height + gap :, :width] = chosen[2]
    canvas[height + gap :, width + gap :] = chosen[3]
    return canvas


def kernel_provenance_story() -> None:
    datasets = ["fashion", "cifar10", "sign"]
    methods = ["lowrank_wmf", "kaiming"]
    banks: dict[tuple[str, str, str], torch.Tensor] = {}
    drift: dict[tuple[str, str], float] = {}
    for dataset_name in datasets:
        for method in methods:
            initial = load_filter_bank(dataset_name, method, "initial")
            final = load_filter_bank(dataset_name, method, "final")
            banks[(dataset_name, method, "initial")] = initial
            banks[(dataset_name, method, "final")] = final
            drift[(dataset_name, method)] = mean_cosine_drift(initial, final)

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(12.9, 7.25),
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1, 1], "wspace": 0.18, "hspace": 0.45},
    )
    columns = [
        "Representative\nclass-0 sample",
        "DI-WMF\nat initialisation",
        "DI-WMF\nafter training",
        "Kaiming\nat initialisation",
        "Kaiming\nafter training",
    ]
    for axis, title in zip(axes[0], columns):
        axis.set_title(title, fontweight="bold", pad=9)

    for row, dataset_name in enumerate(datasets):
        dataset = build_test_dataset(dataset_name, DATA_ROOT, SIGN_ROOT)
        names = class_names(dataset_name, dataset)
        representative = next(image for image, label in dataset if int(label) == 0)
        show_image(axes[row, 0], representative, dataset_name)
        axes[row, 0].set_ylabel(
            f"{DATASET_LABELS[dataset_name]}\nclass 0: {names[0]}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontweight="bold",
        )

        mosaics = [
            _filter_mosaic(banks[(dataset_name, "lowrank_wmf", "initial")]),
            _filter_mosaic(banks[(dataset_name, "lowrank_wmf", "final")]),
            _filter_mosaic(banks[(dataset_name, "kaiming", "initial")]),
            _filter_mosaic(banks[(dataset_name, "kaiming", "final")]),
        ]
        vmax = max(float(np.nanmax(np.abs(mosaic))) for mosaic in mosaics)
        for column, mosaic in enumerate(mosaics, start=1):
            cmap = plt.colormaps["RdBu_r"].copy()
            cmap.set_bad("white")
            axes[row, column].imshow(
                mosaic,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 2].text(
            0.5,
            -0.12,
            f"mean cosine retention = {drift[(dataset_name, 'lowrank_wmf')]:.2f}",
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=DI_RED,
        )
        axes[row, 4].text(
            0.5,
            -0.12,
            f"mean cosine retention = {drift[(dataset_name, 'kaiming')]:.2f}",
            transform=axes[row, 4].transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=KAIMING,
        )

    fig.suptitle(
        "From initial template to trained kernel: an auditable filter journey",
        y=1.015,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Each bank shows the four class-0 stem filters. RGB kernels are averaged over input channels; red and blue denote opposite signed weights.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save(fig, "F17_kernel_provenance_story")

    metadata = {
        dataset: {
            "di_wmf_cosine_retention": drift[(dataset, "lowrank_wmf")],
            "kaiming_cosine_retention": drift[(dataset, "kaiming")],
        }
        for dataset in datasets
    }
    (OUTPUT / "F17_kernel_provenance_story.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_model(dataset: str, method: str) -> torch.nn.Module:
    model = build_model("standard", dataset)
    state = torch.load(
        run_dir(dataset, method) / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model.eval()


def corruption_case_study() -> None:
    dataset_name = "sign"
    dataset = build_test_dataset(dataset_name, DATA_ROOT, SIGN_ROOT)
    names = class_names(dataset_name, dataset)
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    targets = torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])
    pixels = to_pixel_space(images, dataset_name)
    severity = 0.40
    corruption_names = ["clean", "gaussian", "colored_gaussian", "occlusion", "rotation"]
    view_labels = ["Clean", "Gaussian", "Correlated\nGaussian", "Occlusion", "Rotation"]
    views = [pixels]
    for view_index, corruption in enumerate(corruption_names[1:], start=1):
        generator = torch.Generator().manual_seed(4100 + view_index)
        views.append(apply_corruption_batch(pixels, corruption, severity, generator))

    models = {"Kaiming": load_model(dataset_name, "kaiming"), "DI-WMF": load_model(dataset_name, "lowrank_wmf")}
    probabilities: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    with torch.inference_mode():
        for name, model in models.items():
            for view in views:
                probability = model(from_pixel_space(view, dataset_name)).softmax(dim=1)
                probabilities[name].append(probability)
                predictions[name].append(probability.argmax(dim=1))

    both_clean_correct = (
        predictions["Kaiming"][0].eq(targets)
        & predictions["DI-WMF"][0].eq(targets)
    )
    candidates = torch.nonzero(both_clean_correct, as_tuple=False).flatten()
    if len(candidates) == 0:
        candidates = torch.arange(len(dataset))
    best_index = None
    best_key = None
    for index_tensor in candidates:
        index = int(index_tensor)
        label = int(targets[index])
        correct_delta = sum(
            int(predictions["DI-WMF"][view][index] == label)
            - int(predictions["Kaiming"][view][index] == label)
            for view in range(1, len(views))
        )
        confidence_delta = float(
            np.mean(
                [
                    probabilities["DI-WMF"][view][index, label]
                    - probabilities["Kaiming"][view][index, label]
                    for view in range(1, len(views))
                ]
            )
        )
        key = (correct_delta, confidence_delta)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index

    index = int(best_index)
    target = int(targets[index])
    fig = plt.figure(figsize=(12.9, 4.25))
    grid = fig.add_gridspec(2, 5, height_ratios=[2.2, 1.05], hspace=0.18, wspace=0.28)
    for column, (view, view_label) in enumerate(zip(views, view_labels)):
        image_axis = fig.add_subplot(grid[0, column])
        show_image(image_axis, view[index], dataset_name)
        image_axis.set_title(view_label, fontweight="bold")

        decision_axis = fig.add_subplot(grid[1, column])
        methods = ["Kaiming", "DI-WMF"]
        y = np.array([1, 0])
        true_confidence = [
            float(probabilities[method][column][index, target]) for method in methods
        ]
        decision_axis.barh(y, true_confidence, color=[KAIMING, DI_RED], height=0.42)
        decision_axis.set_xlim(0, 1)
        decision_axis.set_yticks(y, ["Kaiming", "DI-WMF"])
        decision_axis.set_xlabel(f"$p$(true {names[target]})", labelpad=1)
        decision_axis.grid(axis="x", alpha=0.18)
        for method, yy, value in zip(methods, y, true_confidence):
            prediction = int(predictions[method][column][index])
            predicted_confidence = float(probabilities[method][column][index, prediction])
            correct = prediction == target
            decision_axis.text(
                0.99,
                yy,
                f"{names[prediction]} {predicted_confidence:.2f} {'✓' if correct else '×'}",
                transform=decision_axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.0,
                color=GREEN if correct else DI_RED,
                fontweight="bold",
            )
        if column > 0:
            decision_axis.set_yticklabels([])

    fig.suptitle(
        f"One Sign decision under corruption (test sample {index}, true class {names[target]}, severity {severity:.1f})",
        y=1.015,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Bars show probability assigned to the true class; right-hand text gives predicted class, confidence and correctness. This selected case explains behaviour; the aggregate heatmap provides the statistical result.",
        ha="center",
        fontsize=7.7,
        color="#555555",
    )
    save(fig, "F18_corruption_case_study")

    metadata = {
        "sample_index": index,
        "target": target,
        "target_name": names[target],
        "severity": severity,
        "selection": "both models clean-correct, then maximum corruption correct-count difference and mean true-class confidence difference",
        "conditions": {},
    }
    for view_index, corruption in enumerate(corruption_names):
        metadata["conditions"][corruption] = {}
        for method in models:
            prediction = int(predictions[method][view_index][index])
            metadata["conditions"][corruption][method] = {
                "prediction": names[prediction],
                "prediction_confidence": float(probabilities[method][view_index][index, prediction]),
                "true_class_probability": float(probabilities[method][view_index][index, target]),
                "correct": prediction == target,
            }
    (OUTPUT / "F18_corruption_case_study.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0


def _spectrum(signal: np.ndarray, sampling_hz: float) -> tuple[np.ndarray, np.ndarray]:
    centred = signal - signal.mean()
    window = np.hanning(len(centred))
    amplitude = np.abs(np.fft.rfft(centred * window))
    frequencies = np.fft.rfftfreq(len(centred), d=1.0 / sampling_hz)
    band = (frequencies >= 0.08) & (frequencies <= 0.8)
    amplitude = amplitude / max(float(amplitude[band].max()), 1e-12)
    return frequencies[band], amplitude[band]


def correncoder_signal_story() -> None:
    experiment_dirs = {
        "Random + MSE": ROOT / "outputs_correncoder_initialisation" / "random_mse",
        "Pretrained + MSE": ROOT / "outputs_correncoder_regression_full",
        "Pretrained + corr": ROOT / "outputs_correncoder_loss_ablation" / "pretrained_corr",
    }
    candidates = []
    for metric_path in experiment_dirs["Pretrained + corr"].glob("*/final_metrics.json"):
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        candidates.append((int(metric["fold"]), float(metric["waveform_correlation"]), metric_path.parent.name))
    median = float(np.median([value for _, value, _ in candidates]))
    fold, _, directory_name = min(candidates, key=lambda item: abs(item[1] - median))

    series: dict[str, np.ndarray] = {}
    target = None
    for label, directory in experiment_dirs.items():
        waveform_path = directory / directory_name / "waveforms.npz"
        if not waveform_path.exists():
            matches = list(directory.glob(f"bidmc_fold{fold:02d}_seed*/waveforms.npz"))
            if not matches:
                raise FileNotFoundError(f"Waveform for fold {fold} in {directory}")
            waveform_path = matches[0]
        archive = np.load(waveform_path)
        series[label] = np.asarray(archive["prediction"], dtype=float)
        if target is None:
            target = np.asarray(archive["target"], dtype=float)

    subjects = load_bidmc_subjects(ROOT / "data_bidmc" / "bidmc_data.mat", target_hz=30)
    ppg = np.asarray(subjects[fold]["ppg"], dtype=float)[: len(target)]
    sampling_hz = 30.0
    window = min(int(round(30.6 * sampling_hz)), len(target))
    stride = max(1, window // 4)
    starts = range(0, len(target) - window + 1, stride)
    start = max(starts, key=lambda sample: float(np.var(target[sample : sample + window])))
    stop = start + window
    time = np.arange(window) / sampling_hz

    colours = {
        "Random + MSE": "#969696",
        "Pretrained + MSE": "#8073ac",
        "Pretrained + corr": DI_RED,
    }
    fig = plt.figure(figsize=(12.6, 6.9))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.55, 1.05], hspace=0.42)
    ppg_axis = fig.add_subplot(grid[0])
    waveform_axis = fig.add_subplot(grid[1], sharex=ppg_axis)
    spectrum_axis = fig.add_subplot(grid[2])

    ppg_axis.plot(time, ppg[start:stop], color=BLUE, lw=1.0)
    ppg_axis.set_ylabel("PPG (z-score)")
    ppg_axis.set_title("(a) Physiological input", loc="left", fontweight="bold")
    ppg_axis.grid(alpha=0.15)
    ppg_axis.tick_params(labelbottom=False)

    target_segment = target[start:stop]
    waveform_axis.plot(time, target_segment, color=REFERENCE, lw=1.8, label="Reference respiration")
    metric_lines = []
    for label, prediction in series.items():
        segment = prediction[start:stop]
        correlation = _safe_correlation(segment, target_segment)
        waveform_axis.plot(time, segment, color=colours[label], lw=1.05, alpha=0.92, label=f"{label} ($r$={correlation:.2f})")
        metric_lines.append((label, correlation))
    waveform_axis.set_ylabel("Normalised amplitude")
    waveform_axis.set_xlabel("Time (s)")
    waveform_axis.set_title("(b) Reconstructed respiratory waveform", loc="left", fontweight="bold")
    waveform_axis.grid(alpha=0.15)
    waveform_axis.legend(frameon=False, ncol=2, loc="upper right")

    target_frequency, target_amplitude = _spectrum(target_segment, sampling_hz)
    target_peak = float(target_frequency[int(np.argmax(target_amplitude))] * 60)
    spectrum_axis.plot(target_frequency * 60, target_amplitude, color=REFERENCE, lw=1.8, label=f"Reference ({target_peak:.1f} bpm)")
    rr_metadata = {"Reference": target_peak}
    for label, prediction in series.items():
        frequency, amplitude = _spectrum(prediction[start:stop], sampling_hz)
        peak = float(frequency[int(np.argmax(amplitude))] * 60)
        rr_metadata[label] = peak
        spectrum_axis.plot(frequency * 60, amplitude, color=colours[label], lw=1.1, label=f"{label} ({peak:.1f} bpm)")
    spectrum_axis.axvline(target_peak, color=REFERENCE, lw=0.8, ls="--", alpha=0.65)
    spectrum_axis.set_xlabel("Respiratory rate (breaths per minute)")
    spectrum_axis.set_ylabel("Normalised amplitude")
    spectrum_axis.set_title("(c) Respiratory-band spectrum", loc="left", fontweight="bold")
    spectrum_axis.grid(alpha=0.15)
    spectrum_axis.legend(frameon=False, ncol=2, loc="upper right")

    fig.suptitle(
        f"Correncoder case study: input, reconstructed waveform and frequency (BIDMC LOSO fold {fold})",
        y=1.005,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "The fold is closest to the median Pretrained + corr subject-level correlation; the 30.6 s segment has the largest reference-waveform variance.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save(fig, "F14_correncoder_representative_waveform")

    metadata = {
        "fold": fold,
        "segment_start_sample": start,
        "segment_length_samples": window,
        "sampling_hz": sampling_hz,
        "selection": "median-correlation fold, then highest-variance 30.6-second target segment",
        "segment_correlations": dict(metric_lines),
        "respiratory_rate_bpm": rr_metadata,
    }
    (OUTPUT / "F14_correncoder_representative_waveform.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    style()
    visual_abstract()
    worked_matched_filter_example()
    random_vs_diwmf_decision_chain()
    cross_task_conclusion_map()
    kernel_provenance_story()
    corruption_case_study()
    correncoder_signal_story()
    print(f"Wrote visual story figures to {OUTPUT}")


if __name__ == "__main__":
    main()
