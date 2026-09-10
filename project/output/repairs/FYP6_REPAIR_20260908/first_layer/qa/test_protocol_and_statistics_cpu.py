"""Independent, read-only protocol and synthetic-statistics checks.

No training, data-loader iteration, checkpoint evaluation, or CUDA use. This
script checks the current frozen/original source relationship and exercises
the analyzer using synthetic rows, not partially completed repair results.
Run with the original CL2 Python environment and -B to suppress bytecode files.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.stats import t, ttest_rel


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]
TRAINING_METRICS = [
    "initial_val_accuracy", "validation_aulc", "test_accuracy", "conv1_drift"
]
CORRUPTION_METRICS = [
    "normalised_accuracy_auc", "mean_accuracy_drop", "mean_retention",
    "mean_consistency", "worst_accuracy",
]


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "independent_shared_first_layer_analyzer",
        HERE / "analyze_shared_first_layer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source_identity():
    counts = {}
    for directory in ("source", "evaluation_source"):
        files = sorted((HERE / directory).rglob("*.py"))
        assert files, directory
        for frozen in files:
            relative = frozen.relative_to(HERE / directory)
            original = ROOT / relative
            assert original.is_file(), str(original)
            assert hashlib.sha256(frozen.read_bytes()).digest() == hashlib.sha256(
                original.read_bytes()
            ).digest(), f"Source differs: {directory}/{relative}"
        counts[directory] = len(files)
    return counts


def check_historical_protocol(analyzer):
    paths = sorted((ROOT / "outputs_first_layer_cross_task").glob("*/config.json"))
    assert len(paths) == 140
    keys = set()
    for path in paths:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        keys.add((cfg["dataset"], cfg["seed"], cfg["init"]))
        for field, value in {
            "model": "standard", "init_layers": 1, "epochs": 10,
            "batch_size": 512, "learning_rate": .001, "train_fraction": .1,
            "val_fraction": .1, "num_workers": 0, "reg_schedule": "none",
            "no_logit_calibration": False, "logit_std": 1,
        }.items():
            assert cfg[field] == value, (str(path), field, cfg[field])
        for field, default in {
            "lr_scheduler": "none", "augmentation": "none",
            "sparsity_lambda": 0, "auxiliary_lambda": 0,
        }.items():
            assert cfg.get(field, default) == default, (str(path), field)
        if cfg["init"] == "lowrank_wmf":
            assert cfg["covariance_rank"] == 16
    expected = {
        (ds, seed, method)
        for ds in analyzer.DATASETS
        for seed in range(10 if ds == "sign" else 5)
        for method in analyzer.METHODS
    }
    assert keys == expected
    return len(paths)


def synthetic_rows(analyzer, metrics, rng, corruptions=("",)):
    return [
        {
            "dataset": ds, "method": method, "model_seed": seed,
            "corruption": corruption,
            **dict(zip(metrics, rng.normal(size=len(metrics)))),
        }
        for ds in analyzer.DATASETS
        for method in analyzer.METHODS
        for seed in range(10 if ds == "sign" else 5)
        for corruption in corruptions
    ]


def check_training_pairs(analyzer, rows, pairs):
    lookup = {
        (r["dataset"], r["method"], r["model_seed"]): r for r in rows
    }
    max_p_error = 0.0
    for pair in pairs:
        ds, method, metric = pair["dataset"], pair["method"], pair["metric"]
        seeds = range(10 if ds == "sign" else 5)
        candidate = np.array([lookup[(ds, method, s)][metric] for s in seeds])
        reference = np.array([lookup[(ds, "kaiming", s)][metric] for s in seeds])
        delta = candidate - reference
        p_error = abs(float(ttest_rel(candidate, reference).pvalue) - pair["raw_p"])
        max_p_error = max(max_p_error, p_error)
        assert p_error < 1e-12
        np.testing.assert_allclose(pair["mean_paired_difference"], delta.mean())
        expected_interval = t.interval(
            .95, len(delta) - 1, loc=delta.mean(),
            scale=delta.std(ddof=1) / np.sqrt(len(delta)),
        )
        np.testing.assert_allclose(
            [pair["ci95_low"], pair["ci95_high"]], expected_interval,
            rtol=1e-12, atol=1e-12,
        )
        # Neutral count is deliberate: positive drift/drop is not a "win".
        assert pair["method_greater_count"] == int((delta > 0).sum())
        assert "method_wins" not in pair
    assert set(Counter(r["correction_scope"] for r in pairs).values()) == {6}
    return max_p_error


def check_constant_differences(analyzer, rows):
    modified = [dict(row) for row in rows]
    for row in modified:
        row["initial_val_accuracy"] = float(row["model_seed"])
        if row["method"] == "random_stem":
            row["initial_val_accuracy"] += 2.0
    pairs = analyzer.pair(modified, ["initial_val_accuracy"], "constant_test")
    for row in pairs:
        mean = 2.0 if row["method"] == "random_stem" else 0.0
        assert row["mean_paired_difference"] == mean
        assert row["paired_difference_std"] == 0.0
        assert row["ci95_low"] == row["ci95_high"] == mean
        assert row["raw_p"] == (0.0 if mean else 1.0)
        assert row["method_greater_count"] == (row["paired_seeds"] if mean else 0)
    return len(pairs)


def main():
    analyzer = load_analyzer()
    source_counts = check_source_identity()
    historical_runs = check_historical_protocol(analyzer)
    rng = np.random.default_rng(88)
    training_rows = synthetic_rows(analyzer, TRAINING_METRICS, rng)
    training = analyzer.pair(training_rows, TRAINING_METRICS, "first_layer_training")
    assert len(training) == 72
    max_p_error = check_training_pairs(analyzer, training_rows, training)
    constant_pairs = check_constant_differences(analyzer, training_rows)

    known = [{"raw_p": p} for p in [.03, .002, .8, .009]]
    analyzer.holm(known, "adjusted")
    np.testing.assert_allclose([r["adjusted"] for r in known], [.06, .008, .8, .027])

    selectivity = analyzer.pair(
        synthetic_rows(analyzer, ["mean_best_class_selectivity"], rng),
        ["mean_best_class_selectivity"], "first_layer_selectivity",
    )
    corruption = analyzer.pair(
        synthetic_rows(analyzer, CORRUPTION_METRICS, rng, tuple(f"c{i}" for i in range(8))),
        CORRUPTION_METRICS, "first_layer_corruption", True,
    )
    assert len(selectivity) == 18 and len(corruption) == 720
    assert set(Counter(r["correction_scope"] for r in corruption).values()) == {48}
    registry = [
        row for row in training + selectivity + corruption
        if row["included_in_replacement_registry"]
    ]
    assert len(registry) == 378
    families = defaultdict(list)
    for row in registry:
        families[row["family"]].append(row)
    assert {k: len(v) for k, v in families.items()} == {
        "first_layer_training": 72, "first_layer_selectivity": 18,
        "first_layer_corruption": 288,
    }
    for family in families.values():
        analyzer.holm(family, "holm_p_family_wide")
    analyzer.holm(registry, "holm_p_all_378")
    for row in registry:
        assert 0 <= row["raw_p"] <= row["holm_p_scope"] <= row["holm_p_family_wide"] <= row["holm_p_all_378"] <= 1

    print(json.dumps({
        "passed": True, "device": "cpu_only_no_torch_import",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "frozen_source_files_equal_to_originals": source_counts,
        "historical_protocol_configs": historical_runs,
        "synthetic_training_contrasts": len(training),
        "maximum_paired_ttest_pvalue_error": max_p_error,
        "constant_difference_contrasts": constant_pairs,
        "known_holm_case": "passed",
        "synthetic_selectivity_contrasts": len(selectivity),
        "synthetic_corruption_contrasts": len(corruption),
        "synthetic_registry_hypotheses": len(registry),
        "method_greater_count_semantics": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
