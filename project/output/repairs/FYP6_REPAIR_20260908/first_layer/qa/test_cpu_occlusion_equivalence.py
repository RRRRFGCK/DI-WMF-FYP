"""Reproduce the independent 56-case, CPU-only occlusion equivalence audit.

Run directly with Python. This test writes no results or checkpoint files and
does not use CUDA. Input fixtures are seeded so reruns use identical pixels.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import torch


FIRST_LAYER = Path(__file__).resolve().parents[1]
SHAPES = ((1, 2, 3), (3, 9, 13), (7, 1, 20, 31), (5, 3, 32, 17))
DTYPES = (torch.float32, torch.float64)
SEVERITIES = (0.0, 0.001, 0.05, 0.1, 0.2, 0.3, 1.0)
FIXTURE_SEED = 20260909
OCCLUSION_SEED = 19


def _load_module(name: str, path: Path):
    """Load the exact frozen/local files, independent of package search paths."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_occlusion_equivalence():
    reference = _load_module(
        "qa_frozen_corruptions", FIRST_LAYER / "source" / "domain_mf" / "corruptions.py"
    ).apply_corruption_batch
    replacement = _load_module(
        "qa_exact_occlusion", FIRST_LAYER / "exact_occlusion.py"
    ).vectorized_occlusion
    fixture_rng = torch.Generator(device="cpu").manual_seed(FIXTURE_SEED)
    checked = 0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for shape in SHAPES:
            for dtype in DTYPES:
                # Normal fixtures deliberately also exercise out-of-range
                # pixels and the original implementation's final clipping.
                pixels = torch.randn(shape, dtype=dtype, device="cpu", generator=fixture_rng)
                for severity in SEVERITIES:
                    original_rng = torch.Generator(device="cpu").manual_seed(OCCLUSION_SEED)
                    replacement_rng = torch.Generator(device="cpu").manual_seed(OCCLUSION_SEED)
                    expected = reference(pixels, "occlusion", severity, original_rng)
                    actual = replacement(pixels, severity, replacement_rng)
                    context = f"shape={shape}, dtype={dtype}, severity={severity}"
                    assert torch.equal(expected, actual), f"Output pixels differ: {context}"
                    assert torch.equal(original_rng.get_state(), replacement_rng.get_state()), (
                        f"Post-call generator states differ: {context}"
                    )
                    checked += 1
    finally:
        torch.set_num_threads(previous_threads)
    assert checked == 56, f"Unexpected case count: {checked}"


if __name__ == "__main__":
    test_cpu_occlusion_equivalence()
    print(json.dumps({
        "passed": True,
        "cpu_shape_dtype_cases_passed": 56,
        "pixels_bitwise_equal": True,
        "rng_state_bitwise_equal": True,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "fixture_seed": FIXTURE_SEED,
        "occlusion_seed": OCCLUSION_SEED,
        "shapes": SHAPES,
        "dtypes": [str(dtype) for dtype in DTYPES],
        "severities": SEVERITIES,
    }, indent=2))
