import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
CIFAR10_STD = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)


def to_pixel_space(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Convert a transformed input to a common [0, 1] pixel domain."""
    if dataset_name == "cifar10":
        mean = CIFAR10_MEAN.to(image)
        std = CIFAR10_STD.to(image)
        image = image * std + mean
    return image.clamp(0.0, 1.0)


def from_pixel_space(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Restore the model's expected dataset-specific normalisation."""
    if dataset_name == "cifar10":
        mean = CIFAR10_MEAN.to(image)
        std = CIFAR10_STD.to(image)
        return (image - mean) / std
    return image


def _sample_generator(noise_seed: int, sample_index: int) -> torch.Generator:
    # The mapping depends only on corruption seed and sample index, ensuring
    # every initialisation method sees exactly the same corrupted image.
    seed = (int(noise_seed) * 1_000_003 + int(sample_index) * 97_409) % (2**63 - 1)
    return torch.Generator().manual_seed(seed)


def gaussian_noise(
    image: torch.Tensor,
    sigma: float,
    noise_seed: int,
    sample_index: int,
) -> torch.Tensor:
    if sigma < 0:
        raise ValueError("Gaussian sigma must be non-negative")
    if sigma == 0:
        return image.clone()
    generator = _sample_generator(noise_seed, sample_index)
    noise = torch.randn(
        image.shape, generator=generator, dtype=image.dtype, device="cpu"
    ).to(image.device)
    return (image + sigma * noise).clamp(0.0, 1.0)


CORRUPTIONS = (
    "gaussian",
    "colored_gaussian",
    "salt_pepper",
    "occlusion",
    "rotation",
    "translation",
    "brightness",
    "contrast",
)


def apply_corruption_batch(
    pixels: torch.Tensor,
    corruption: str,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply a deterministic batched corruption in the [0, 1] pixel domain.

    Severity is normalised to [0, 1]. For rotation it maps to 45 degrees and
    for translation to a fraction of image width/height; for the remaining
    corruptions it directly controls noise, probability, area or intensity.
    """
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")
    if severity == 0:
        return pixels.clone()
    squeeze = pixels.ndim == 3
    if squeeze:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4:
        raise ValueError("pixels must have shape [C,H,W] or [B,C,H,W]")
    batch, channels, height, width = pixels.shape
    device = pixels.device
    dtype = pixels.dtype

    if corruption in {"gaussian", "colored_gaussian"}:
        noise = torch.randn(
            pixels.shape, generator=generator, device=device, dtype=dtype
        )
        if corruption == "colored_gaussian":
            coordinates = torch.arange(5, device=device, dtype=dtype) - 2
            kernel_1d = torch.exp(-0.5 * (coordinates / 1.0).square())
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel = torch.outer(kernel_1d, kernel_1d)
            kernel = kernel.expand(channels, 1, 5, 5)
            noise = F.conv2d(
                F.pad(noise, (2, 2, 2, 2), mode="reflect"),
                kernel,
                groups=channels,
            )
            noise = noise / noise.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        corrupted = pixels + severity * noise
    elif corruption == "salt_pepper":
        uniform = torch.rand(
            pixels.shape, generator=generator, device=device, dtype=dtype
        )
        corrupted = torch.where(
            uniform < severity / 2,
            torch.zeros_like(pixels),
            torch.where(uniform > 1 - severity / 2, torch.ones_like(pixels), pixels),
        )
    elif corruption == "occlusion":
        corrupted = pixels.clone()
        side_fraction = math.sqrt(severity)
        box_h = max(1, min(height, int(round(height * side_fraction))))
        box_w = max(1, min(width, int(round(width * side_fraction))))
        tops = torch.randint(
            0, height - box_h + 1, (batch,), generator=generator, device=device
        )
        lefts = torch.randint(
            0, width - box_w + 1, (batch,), generator=generator, device=device
        )
        fill = pixels.mean(dim=(2, 3), keepdim=True)
        for index in range(batch):
            top = int(tops[index])
            left = int(lefts[index])
            corrupted[index, :, top : top + box_h, left : left + box_w] = fill[index]
    elif corruption in {"rotation", "translation"}:
        theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        if corruption == "rotation":
            angles = (
                2
                * torch.rand((batch,), generator=generator, device=device, dtype=dtype)
                - 1
            ) * (45.0 * severity * math.pi / 180.0)
            cosine = torch.cos(angles)
            sine = torch.sin(angles)
            theta[:, 0, 0] = cosine
            theta[:, 0, 1] = -sine
            theta[:, 1, 0] = sine
            theta[:, 1, 1] = cosine
        else:
            offsets = (
                2
                * torch.rand(
                    (batch, 2), generator=generator, device=device, dtype=dtype
                )
                - 1
            ) * severity
            theta[:, :, 2] = offsets
        grid = F.affine_grid(theta, pixels.shape, align_corners=False)
        corrupted = F.grid_sample(
            pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
    elif corruption == "brightness":
        shift = (
            2
            * torch.rand(
                (batch, 1, 1, 1), generator=generator, device=device, dtype=dtype
            )
            - 1
        ) * severity
        corrupted = pixels + shift
    else:  # contrast
        factor = 1 + (
            2
            * torch.rand(
                (batch, 1, 1, 1), generator=generator, device=device, dtype=dtype
            )
            - 1
        ) * severity
        mean = pixels.mean(dim=(2, 3), keepdim=True)
        corrupted = mean + factor * (pixels - mean)
    corrupted = corrupted.clamp(0.0, 1.0)
    return corrupted.squeeze(0) if squeeze else corrupted


class CorruptedDataset(Dataset):
    """Deterministic corruption wrapper around a clean test dataset."""

    def __init__(
        self,
        base_dataset,
        dataset_name: str,
        corruption: str,
        severity: float,
        noise_seed: int,
    ):
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unsupported corruption: {corruption}")
        self.base_dataset = base_dataset
        self.dataset_name = dataset_name
        self.corruption = corruption
        self.severity = float(severity)
        self.noise_seed = int(noise_seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        pixels = to_pixel_space(image, self.dataset_name)
        generator = _sample_generator(self.noise_seed, index)
        pixels = apply_corruption_batch(
            pixels, self.corruption, self.severity, generator
        )
        return from_pixel_space(pixels, self.dataset_name), label
