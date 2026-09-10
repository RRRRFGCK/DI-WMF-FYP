from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


@dataclass(frozen=True)
class ModelSpec:
    dataset: str
    image_size: int
    input_channels: int
    num_classes: int
    patch_size: int
    stride: int

    @property
    def grid_size(self) -> int:
        return (self.image_size - self.patch_size) // self.stride + 1


SPECS = {
    "mnist": ModelSpec("mnist", 20, 1, 10, 10, 5),
    "fashion": ModelSpec("fashion", 20, 1, 10, 10, 5),
    "cifar10": ModelSpec("cifar10", 32, 3, 10, 8, 6),
    "sign": ModelSpec("sign", 64, 3, 10, 32, 16),
}


class MFOneLayer(nn.Module):
    """A transparent full-image matched-filter classifier."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(
            spec.input_channels,
            spec.num_classes,
            kernel_size=spec.image_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x).flatten(1)

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        if layer_name != "conv1":
            raise KeyError(layer_name)
        return torch.arange(self.spec.num_classes)

    @property
    def anchor_layer_names(self):
        return ("conv1",)

    @property
    def output_layer(self):
        return self.conv1


class MFTwoLayer(nn.Module):
    """Local template bank followed by a full-feature matched filter."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        grid = spec.grid_size
        local_filters = spec.num_classes * grid * grid
        self.conv1 = nn.Conv2d(
            spec.input_channels,
            local_filters,
            kernel_size=spec.patch_size,
            stride=spec.stride,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            local_filters,
            spec.num_classes,
            kernel_size=grid,
            bias=True,
        )

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.first_layer_features(x)).flatten(1)

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        features = self.first_layer_features(x)
        batch, _, height, width = features.shape
        grid_positions = self.spec.grid_size ** 2
        grouped = features.view(
            batch,
            self.spec.num_classes,
            grid_positions,
            height,
            width,
        )
        return grouped.mean(dim=(2, 3, 4))

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        if layer_name == "conv1":
            return torch.arange(self.spec.num_classes).repeat_interleave(
                self.spec.grid_size ** 2
            )
        if layer_name == "conv2":
            return torch.arange(self.spec.num_classes)
        raise KeyError(layer_name)

    @property
    def anchor_layer_names(self):
        return ("conv1", "conv2")

    @property
    def output_layer(self):
        return self.conv2


class StandardCNN(nn.Module):
    """Conventional pooled CNN with class-indexed filter banks.

    The topology is deliberately independent of the MF-CNN classifier: local
    convolutions are followed by max pooling, global average pooling and a
    linear head. Channel counts are multiples of the class count so a portable
    matched-filter dictionary can retain an auditable class assignment.
    """

    stem_templates_per_class = 4
    body_templates_per_class = 8

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        stem_channels = spec.num_classes * self.stem_templates_per_class
        body_channels = spec.num_classes * self.body_templates_per_class
        self.conv1 = nn.Conv2d(
            spec.input_channels, stem_channels, kernel_size=5, padding=2, bias=True
        )
        self.conv2 = nn.Conv2d(
            stem_channels, body_channels, kernel_size=3, padding=1, bias=True
        )
        self.classifier = nn.Linear(body_channels, spec.num_classes, bias=True)

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)

    def second_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            F.relu(self.conv2(self.first_layer_features(x))), kernel_size=2
        )

    def penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.second_layer_features(x), 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.penultimate_features(x))

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        features = self.second_layer_features(x)
        batch, _, height, width = features.shape
        grouped = features.view(
            batch,
            self.spec.num_classes,
            self.body_templates_per_class,
            height,
            width,
        )
        return grouped.mean(dim=(2, 3, 4))

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        if layer_name == "conv1":
            return torch.arange(self.spec.num_classes).repeat_interleave(
                self.stem_templates_per_class
            )
        if layer_name == "conv2":
            return torch.arange(self.spec.num_classes).repeat_interleave(
                self.body_templates_per_class
            )
        raise KeyError(layer_name)

    @property
    def anchor_layer_names(self):
        return ("conv1", "conv2")

    @property
    def output_layer(self):
        return self.classifier


class C4RotationInvariantCNN(StandardCNN):
    """StandardCNN with an explicit four-orientation matched-filter orbit.

    Every learned base template is evaluated at 0, 90, 180 and 270 degrees.
    Max pooling over that orientation orbit is applied after *both*
    convolutional layers.  The resulting feature maps are C4-equivariant;
    global average pooling therefore makes the logits C4-invariant (up to
    floating-point round-off).  Only rotated views of the same parameters are
    used, so the trainable parameter count and class/template assignments stay
    identical to :class:`StandardCNN`.
    """

    rotations = (0, 1, 2, 3)

    @classmethod
    def _orbit_max_convolution(
        cls, x: torch.Tensor, convolution: nn.Conv2d
    ) -> torch.Tensor:
        rotated_weights = [
            torch.rot90(convolution.weight, turns, dims=(-2, -1))
            for turns in cls.rotations
        ]
        weight = torch.cat(rotated_weights, dim=0)
        bias = (
            convolution.bias.repeat(len(cls.rotations))
            if convolution.bias is not None
            else None
        )
        response = F.conv2d(
            x,
            weight,
            bias,
            stride=convolution.stride,
            padding=convolution.padding,
            dilation=convolution.dilation,
            groups=convolution.groups,
        )
        batch, _, height, width = response.shape
        response = response.view(
            batch,
            len(cls.rotations),
            convolution.out_channels,
            height,
            width,
        )
        return response.amax(dim=1)

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        response = self._orbit_max_convolution(x, self.conv1)
        return F.max_pool2d(F.relu(response), kernel_size=2)

    def second_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        response = self._orbit_max_convolution(
            self.first_layer_features(x), self.conv2
        )
        return F.max_pool2d(F.relu(response), kernel_size=2)


class HalfWidthCNN(StandardCNN):
    """Half-width StandardCNN for controlled parameter-efficiency tests."""

    stem_templates_per_class = 2
    body_templates_per_class = 4


class QuarterWidthCNN(StandardCNN):
    """Quarter-width StandardCNN for controlled parameter-efficiency tests."""

    stem_templates_per_class = 1
    body_templates_per_class = 2


class CorrEncoderCNN(StandardCNN):
    """Correncoder-style classifier with a tied convolutional decoder.

    The discriminative path is identical to ``StandardCNN``. During training,
    the pooled encoder maps are decoded with the transposed encoder filters so
    the latent representation must remain predictive of the input as well as
    the class. Tying the decoder makes every reconstruction atom the same
    auditable template used by the encoder and adds no decoder parameters.
    """

    def encoded_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_layer_features(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoded_maps(x)
        decoded = F.interpolate(encoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = F.relu(
            F.conv_transpose2d(decoded, self.conv2.weight, padding=1)
        )
        decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.conv_transpose2d(decoded, self.conv1.weight, padding=2)

    def auxiliary_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstruction = self.reconstruct(inputs).flatten(1)
        target = inputs.flatten(1)
        reconstruction = reconstruction - reconstruction.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        correlation = F.cosine_similarity(reconstruction, target, dim=1, eps=1e-8)
        return (1.0 - correlation).mean()


class FullCorrEncoderCNN(StandardCNN):
    """Correncoder-style classifier with a separately trainable decoder.

    This is the fuller encoder--decoder ablation used in this project: unlike
    :class:`CorrEncoderCNN`, decoder filters are not tied after their initial
    copy from the encoder. It tests whether reconstruction capacity, rather
    than strict template sharing, resolves the classification/reconstruction
    trade-off. It remains a classification adaptation and is not presented as
    an exact reproduction of the paper's regression task.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.decoder2 = nn.ConvTranspose2d(
            self.conv2.out_channels,
            self.conv2.in_channels,
            kernel_size=self.conv2.kernel_size,
            padding=self.conv2.padding,
            bias=True,
        )
        self.decoder1 = nn.ConvTranspose2d(
            self.conv1.out_channels,
            self.conv1.in_channels,
            kernel_size=self.conv1.kernel_size,
            padding=self.conv1.padding,
            bias=True,
        )

    def initialise_decoder_from_encoder(self) -> None:
        """Start the untied decoder from the matched encoder dictionary."""

        with torch.no_grad():
            self.decoder2.weight.copy_(self.conv2.weight)
            self.decoder1.weight.copy_(self.conv1.weight)
            self.decoder2.bias.zero_()
            self.decoder1.bias.zero_()

    def encoded_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_layer_features(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoded_maps(x)
        decoded = F.interpolate(
            encoded, scale_factor=2, mode="bilinear", align_corners=False
        )
        decoded = F.relu(self.decoder2(decoded))
        decoded = F.interpolate(
            decoded, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.decoder1(decoded)

    def auxiliary_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstruction = self.reconstruct(inputs).flatten(1)
        target = inputs.flatten(1)
        reconstruction = reconstruction - reconstruction.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        correlation = F.cosine_similarity(reconstruction, target, dim=1, eps=1e-8)
        return (1.0 - correlation).mean()


class PublishedCorrEncoder1D(nn.Module):
    """The published PPG-to-respiration Correncoder regression topology.

    This is intentionally separate from the image-classification adaptations
    above.  It mirrors the authors' public model: three Conv1d encoder layers
    with eight channels and kernels 150/75/50, followed by three mirrored
    ConvTranspose1d decoder layers.  A 288-sample input produces a 288-sample
    output, as in the published 9.6-second, 30-Hz protocol.
    """

    input_length = 288

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=150, padding=20)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=75, padding=20)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=50, padding=10)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.decoder3 = nn.ConvTranspose1d(8, 8, kernel_size=50, padding=10)
        self.decoder2 = nn.ConvTranspose1d(8, 8, kernel_size=75, padding=20)
        self.decoder1 = nn.ConvTranspose1d(8, 1, kernel_size=150, padding=20)
        # Optional and absent by default so legacy checkpoints and baseline
        # parameter counts remain unchanged.
        self.output_calibration: nn.Module | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(F.relu(self.conv2(x)))
        return self.dropout3(torch.sigmoid(self.conv3(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = torch.sigmoid(self.decoder3(x))
        x = F.relu(self.decoder2(x))
        x = self.decoder1(x)
        return self.output_calibration(x) if self.output_calibration is not None else x


class SharedConceptCNN(nn.Module):
    """CNN with a shared matched-template dictionary and sparse linear head.

    Unlike ``StandardCNN``, convolutional channels are not owned by a class.
    Every class can reuse the same concepts through a signed classifier weight,
    which also gives an exact per-concept logit decomposition.
    """

    stem_channels = 40
    concept_channels = 80

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(
            spec.input_channels, self.stem_channels, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv2d(
            self.stem_channels, self.concept_channels, kernel_size=3, padding=1
        )
        self.classifier = nn.Linear(self.concept_channels, spec.num_classes)

    def first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(F.relu(self.conv1(x)), 2)

    def second_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(F.relu(self.conv2(self.first_layer_features(x))), 2)

    def penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.second_layer_features(x), 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.penultimate_features(x))

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        # Unique prototype ids measure identity preservation without pretending
        # that a shared concept is owned by a single class.
        if layer_name == "conv1":
            return torch.arange(self.stem_channels)
        if layer_name == "conv2":
            return torch.arange(self.concept_channels)
        raise KeyError(layer_name)

    @property
    def anchor_layer_names(self):
        return ("conv1", "conv2")

    @property
    def output_layer(self):
        return self.classifier

    def sparsity_penalty(self) -> torch.Tensor:
        # Mean L1 norm per class keeps lambda interpretable when the concept
        # dictionary size changes (unlike a mean over all coefficients).
        return self.classifier.weight.abs().sum(dim=1).mean()

    def sparsity_metrics(self) -> dict[str, float]:
        weights = self.classifier.weight.detach()
        scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        relative = weights.abs() / scale
        return {
            "classifier_l1_mean": float(weights.abs().mean()),
            "fraction_exactly_zero": float((weights == 0).float().mean()),
            "fraction_below_1pct_class_max": float((relative < 0.01).float().mean()),
            "fraction_below_5pct_class_max": float((relative < 0.05).float().mean()),
            "mean_active_concepts_5pct": float((relative >= 0.05).float().sum(1).mean()),
        }

    def proximal_sparsity_step(self, step_size: float) -> None:
        """Apply the exact proximal operator for the mean per-class L1 norm."""

        if step_size <= 0:
            return
        threshold = step_size / self.spec.num_classes
        with torch.no_grad():
            weights = self.classifier.weight
            weights.copy_(weights.sign() * (weights.abs() - threshold).clamp_min(0.0))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(x)) + residual)


class SmallResNet(nn.Module):
    """Compact residual architecture for a genuine cross-architecture test."""

    stem_templates_per_class = 4

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        stem_channels = spec.num_classes * self.stem_templates_per_class
        self.conv1 = nn.Conv2d(
            spec.input_channels, stem_channels, kernel_size=5, padding=2, bias=True
        )
        self.block1 = ResidualBlock(stem_channels, stem_channels)
        self.block2 = ResidualBlock(stem_channels, 80, stride=2)
        self.classifier = nn.Linear(80, spec.num_classes)

    def stem_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(F.relu(self.conv1(x)), 2)

    def penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(self.stem_features(x))
        x = self.block2(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.penultimate_features(x))

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        if layer_name != "conv1":
            raise KeyError(layer_name)
        return torch.arange(self.spec.num_classes).repeat_interleave(
            self.stem_templates_per_class
        )

    @property
    def anchor_layer_names(self):
        return ("conv1",)

    @property
    def output_layer(self):
        return self.classifier


class StandardResNet18(nn.Module):
    """Torchvision ResNet-18 with an auditable matched stem and linear head.

    The canonical 64-channel, [2, 2, 2, 2] residual topology is retained. Only
    the input channel count and number of output classes are dataset-specific.
    """

    stem_channels = 64

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        backbone = resnet18(weights=None, num_classes=spec.num_classes)
        if spec.input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                spec.input_channels,
                self.stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.classifier = backbone.fc

    def stem_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))

    def penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.penultimate_features(x))

    def class_evidence(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def anchor_classes(self, layer_name: str) -> torch.Tensor:
        if layer_name != "conv1":
            raise KeyError(layer_name)
        base = self.stem_channels // self.spec.num_classes
        remainder = self.stem_channels % self.spec.num_classes
        counts = [
            base + int(class_id < remainder)
            for class_id in range(self.spec.num_classes)
        ]
        return torch.cat(
            [torch.full((count,), class_id) for class_id, count in enumerate(counts)]
        )

    @property
    def anchor_layer_names(self):
        return ("conv1",)

    @property
    def output_layer(self):
        return self.classifier


def build_model(model_name: str, dataset: str) -> nn.Module:
    if dataset not in SPECS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if model_name == "1layer":
        return MFOneLayer(SPECS[dataset])
    if model_name == "2layer":
        return MFTwoLayer(SPECS[dataset])
    if model_name == "standard":
        return StandardCNN(SPECS[dataset])
    if model_name == "rotation_invariant":
        return C4RotationInvariantCNN(SPECS[dataset])
    if model_name == "standard_half":
        return HalfWidthCNN(SPECS[dataset])
    if model_name == "standard_quarter":
        return QuarterWidthCNN(SPECS[dataset])
    if model_name == "correncoder":
        return CorrEncoderCNN(SPECS[dataset])
    if model_name == "correncoder_full":
        return FullCorrEncoderCNN(SPECS[dataset])
    if model_name == "concept":
        return SharedConceptCNN(SPECS[dataset])
    if model_name == "smallresnet":
        return SmallResNet(SPECS[dataset])
    if model_name == "resnet18":
        return StandardResNet18(SPECS[dataset])
    raise ValueError(f"Unknown model: {model_name}")
