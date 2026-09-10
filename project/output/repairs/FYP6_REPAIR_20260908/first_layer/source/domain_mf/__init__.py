"""Domain-informed matched-filter CNN experiments."""

from .models import (
    CorrEncoderCNN,
    HalfWidthCNN,
    MFOneLayer,
    MFTwoLayer,
    ModelSpec,
    QuarterWidthCNN,
    SharedConceptCNN,
    SmallResNet,
    StandardResNet18,
    StandardCNN,
    build_model,
)

__all__ = [
    "CorrEncoderCNN",
    "HalfWidthCNN",
    "MFOneLayer",
    "MFTwoLayer",
    "ModelSpec",
    "QuarterWidthCNN",
    "SharedConceptCNN",
    "SmallResNet",
    "StandardResNet18",
    "StandardCNN",
    "build_model",
]
