"""Build an additive 902-row result-version index. No training or p-value edits.

The original registry is read as strings and copied verbatim by field. All
classification and replacement metadata are appended. Run from any directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPAIR = ROOT / "output/repairs/FYP6_REPAIR_20260908"
SOURCES = REPAIR / "index_sources"
REGISTRY = ROOT / "statistical_controls/all_core_hypotheses_holm.csv"
FIRST = ROOT / "first_layer_shared_results_20260908/statistical_registry_replacement_378.csv"
PHYSIO = REPAIR / "physiology_evaluation/physiology_paired_holm_patient46.csv"
MAIN = REPAIR / "thesis/submission_control_results/final_main_ten_seed_registry.csv"
CROSS = REPAIR / "thesis/submission_control_results/cross_architecture_post_aulc.csv"
FIRST_FAMILIES = {"first_layer_training": 72, "first_layer_selectivity": 18, "first_layer_corruption": 288}
PHYSIO_FAMILIES = {"correncoder_initialisation_objective": 24, "correncoder_layerwise_lag": 20, "correncoder_depth_covariance": 35}
SOURCE_MANIFEST = {}


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames), rows


def snapshot(path, name=None, *, allow_frozen_fallback=False):
    """Preserve source bytes; optionally verify an archived frozen substitute."""
    path = Path(path)
    dest = SOURCES / (name or Path(path).name)
    if not path.exists() and allow_frozen_fallback:
        manifest_path = REPAIR / "index_summary.json"
        if not dest.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing original source and verified frozen fallback: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("source_manifest", {}).get(relative(path), {})
        if expected.get("snapshot") != relative(dest) or not expected.get("sha256"):
            raise ValueError(f"Frozen fallback lacks matching provenance in index_summary.json: {dest}")
        if sha256(dest) != expected["sha256"]:
            raise ValueError(f"Frozen fallback SHA-256 mismatch: {dest}")
        SOURCE_MANIFEST[relative(path)] = {"snapshot": relative(dest), "sha256": expected["sha256"]}
        return dest
    content = path.read_bytes()
    if dest.exists() and dest.read_bytes() != content:
        raise ValueError(f"Snapshot differs from input; review explicitly: {dest}")
    if not dest.exists():
        dest.write_bytes(content)
    SOURCE_MANIFEST[relative(path)] = {"snapshot": relative(dest), "sha256": sha256(path)}
    return dest


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_method(method):
    return "lowrank_wmf" if method == "lowrank_wmf_rank16" else method


def first_key(row):
    return tuple(row.get(key, "") for key in ("family", "dataset", "metric", "corruption", "reference")) + (
        normalized_method(row.get("method", "")), row["correction_scope"]
    )


def actual_history_metrics(path):
    _, history = read_csv(path)
    epochs = [int(row["epoch"]) for row in history]
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epochs: {path}")
    values = [float(row["val_accuracy"]) for row in history]
    post = [float(row["val_accuracy"]) for row in history if int(row["epoch"]) >= 1]
    if 0 not in epochs or not post:
        raise ValueError(f"Missing Epoch 0 or post-update observations: {path}")
    return statistics.mean(values), statistics.mean(post), min(epochs), max(epochs), len(epochs)


def audit_efficiency(registry):
    """Use saved histories, not a current analyzer's name, to classify old AULC."""
    _, paired = read_csv(ROOT / "efficiency_results/paired.csv")
    roots = ["outputs_efficiency_width", "outputs_full20_cifar10", "outputs_full20_fashion", "outputs_full20_sign_bs64"]
    runs = defaultdict(dict)
    history_evidence = []
    for root_name in roots:
        root = ROOT / root_name
        summary = root / "summary.csv"
        SOURCE_MANIFEST[relative(summary)] = {"sha256": sha256(summary), "role": "efficiency_history_locator"}
        _, entries = read_csv(summary)
        for item in entries:
            # The historical width comparisons have paired seeds 0..4.
            if int(item["seed"]) not in range(5) or item["method"] not in ("kaiming", "lowrank_wmf"):
                continue
            key = (item["dataset"], item["model"], item["method"])
            seed = int(item["seed"])
            if seed in runs[key]:
                raise ValueError(f"Duplicate efficiency seed: {key}, {seed}")
            history = root / item["run"] / "history.csv"
            inclusive, post, first, last, count = actual_history_metrics(history)
            runs[key][seed] = (inclusive, post)
            history_evidence.append({"history_path": relative(history), "sha256": sha256(history), "dataset": item["dataset"],
                "model": item["model"], "method": item["method"], "seed": seed, "epoch_min": first,
                "epoch_max": last, "history_rows": count, "aulc_0T": inclusive, "aulc_1T": post})
    audits = {}
    for number, old in enumerate(registry, 1):
        if old["family"] != "parameter_efficiency" or old["metric"] != "validation_aulc":
            continue
        source_row = int(old["source_row"])
        pair = paired[source_row - 1]
        identity = json.loads(old["hypothesis"])
        for key in ("dataset", "contrast", "metric"):
            if pair[key] != identity[key]:
                raise ValueError(f"Efficiency identity mismatch: registry row {number}")
        if old["paired_difference"] != pair["mean_candidate_minus_reference"]:
            raise ValueError(f"Efficiency original effect mismatch: row {number}")
        reference = runs[(pair["dataset"], pair["reference_model"], pair["reference_method"])]
        candidate = runs[(pair["dataset"], pair["candidate_model"], pair["candidate_method"])]
        seeds = sorted(set(reference) & set(candidate))
        if seeds != list(range(5)) or int(pair["paired_seeds"]) != len(seeds):
            raise ValueError(f"Efficiency seed coverage mismatch: row {number}")
        values = {}
        for i, label in enumerate(("inclusive", "post_update")):
            values[f"{label}_reference_mean"] = statistics.mean(reference[s][i] for s in seeds)
            values[f"{label}_candidate_mean"] = statistics.mean(candidate[s][i] for s in seeds)
            values[f"{label}_paired_difference"] = statistics.mean(candidate[s][i] - reference[s][i] for s in seeds)
        def matches(label):
            return all(math.isclose(values[f"{label}_{key}"], float(pair[field]), rel_tol=1e-10, abs_tol=1e-9)
                for key, field in (("reference_mean", "reference_mean"), ("candidate_mean", "candidate_mean"),
                                   ("paired_difference", "mean_candidate_minus_reference")))
        inc, post = matches("inclusive"), matches("post_update")
        classification = "post_update_epoch0_excluded" if post and not inc else "inclusive_epoch0_included" if inc and not post else "unresolved"
        audits[number] = {"registry_row_1based": number, "source_row_1based": source_row,
            **{key: pair[key] for key in ("dataset", "contrast", "reference_model", "reference_method", "candidate_model", "candidate_method")},
            "paired_seed_ids": json.dumps(seeds), "recorded_paired_difference": old["paired_difference"],
            **values, "matches_inclusive": inc, "matches_post_update": post, "aulc_definition": classification}
    if len(audits) != 18:
        raise ValueError(f"Expected 18 efficiency AULC audits, got {len(audits)}")
    write_csv(SOURCES / "parameter_efficiency_aulc_evidence_18.csv", list(next(iter(audits.values()))), audits.values())
    write_csv(SOURCES / "parameter_efficiency_history_metrics.csv", list(history_evidence[0]), history_evidence)
    return audits


def main():
    fields, original = read_csv(REGISTRY)
    original_hash = sha256(REGISTRY)
    if len(original) != 902:
        raise ValueError(f"Expected exactly 902 registry rows, got {len(original)}")
    for family, count in (FIRST_FAMILIES | PHYSIO_FAMILIES).items():
        if sum(row["family"] == family for row in original) != count:
            raise ValueError(f"Unexpected original family count: {family}")
    SOURCES.mkdir(parents=True, exist_ok=True)
    registry_copy = snapshot(REGISTRY, "original_all_core_hypotheses_holm.csv")
    first_copy = snapshot(FIRST)
    physio_copy = snapshot(PHYSIO)
    main_copy = snapshot(MAIN, allow_frozen_fallback=True)
    cross_copy = snapshot(CROSS, allow_frozen_fallback=True)
    snapshot(ROOT / "efficiency_results/paired.csv", "original_efficiency_paired.csv")
    snapshot(ROOT / "analyze_efficiency.py", "analyze_efficiency_inspected.py")
    snapshot(ROOT / "domain_mf/metrics.py", "metrics_inspected.py")
    _, first_rows = read_csv(first_copy)
    _, physio_rows = read_csv(physio_copy)
    _, main_rows = read_csv(main_copy)
    _, cross_rows = read_csv(cross_copy)
    if len(first_rows) != 378 or len(physio_rows) != 120:
        raise ValueError("Replacement registry count mismatch")
    first_map = {}
    for number, row in enumerate(first_rows, 1):
        key = first_key(row)
        if key in first_map:
            raise ValueError(f"Duplicate first-layer replacement: {key}")
        first_map[key] = (number, row)
    efficiency = audit_efficiency(original)
    result = []
    first_used = set()
    added_fields = ["registry_row_1based", "registry_csv_line_1based", "registry_source_path", "registry_source_sha256",
        "registry_preserved_copy", "original_identity", "original_result_status", "result_version_status",
        "analysis_timing", "preregistered", "historical_metric_definition", "metric_definition_evidence",
        "replacement_level", "replacement_path", "replacement_sha256", "replacement_row_1based",
        "replacement_csv_line_1based", "replacement_row_count", "replacement_identity", "replacement_analysis_timing",
        "mapping_status", "status_reason", "current_use_note"]
    if set(fields) & set(added_fields):
        raise ValueError("Metadata columns collide with original registry")
    for number, old in enumerate(original, 1):
        hypothesis = json.loads(old["hypothesis"])
        entry = dict(old)
        entry.update(dict.fromkeys(added_fields, ""))
        entry.update(registry_row_1based=number, registry_csv_line_1based=number + 1,
            registry_source_path=relative(REGISTRY), registry_source_sha256=original_hash,
            registry_preserved_copy=relative(registry_copy),
            original_identity=json.dumps({"family": old["family"], "source": old["source"], "source_row": old["source_row"],
                "correction_scope": old["correction_scope"], "hypothesis": hypothesis}, sort_keys=True),
            original_result_status="historical_retrospective_registry_row",
            result_version_status="retained_historical_result_version_not_replaced",
            analysis_timing="retrospective", preregistered="False", mapping_status="not_replaced_within_bounded_index",
            historical_metric_definition="not_reaudited_in_this_index" if old["metric"] == "validation_aulc" else "not_applicable_to_aulc",
            current_use_note="Version status is not a blanket correctness certificate. Original p-values retain their original metric and correction scope.",
            status_reason="No replacement identified within the bounded result-version mapping. Historical values are retained, not independently recertified.")

        def replacement(path, rows, selected=None, level="one_to_one_result"):
            entry.update(replacement_level=level, replacement_path=relative(path), replacement_sha256=sha256(path),
                replacement_row_count=len(rows), replacement_analysis_timing="retrospective_not_preregistered")
            if selected is not None:
                row_number, target = selected
                entry.update(replacement_row_1based=row_number, replacement_csv_line_1based=row_number + 1,
                    replacement_identity=json.dumps(target, sort_keys=True), mapping_status="matched")

        if old["family"] in FIRST_FAMILIES:
            joined = {**hypothesis, "family": old["family"], "correction_scope": old["correction_scope"]}
            selected = first_map.get(first_key(joined))
            if selected is None:
                raise ValueError(f"Missing first-layer mapping at registry row {number}")
            if hypothesis.get("model", "standard") != selected[1].get("model", "standard"):
                raise ValueError(f"First-layer model mismatch at row {number}")
            first_used.add(selected[0])
            replacement(first_copy, first_rows, selected)
            entry.update(result_version_status="superseded_first_layer_shared_downstream_repair",
                status_reason="Shared-downstream control replacement with same task, method, reference, metric and inherited retrospective correction scope. Historical lowrank_wmf_rank16 is matched to lowrank_wmf only.")
        elif old["family"] in PHYSIO_FAMILIES:
            replacement(physio_copy, physio_rows, level="protocol_level_not_one_to_one")
            entry.update(result_version_status="superseded_physiology_protocol_repair", mapping_status="protocol_level_replacement_no_row_equivalence",
                replacement_identity="frozen_repaired_physiology_120_exploratory; 46 patients; 53 records; metric-specific/manual-RR masks",
                status_reason="79 historical hypotheses are superseded at protocol level by a 120-hypothesis repaired analysis. The 46-patient population and manual-RR/metric-specific masks changed; no one-to-one p-value or hypothesis equivalence is claimed.")
        elif old["family"] == "main_cross_task" and old["metric"] == "validation_aulc":
            selected = [(i, r) for i, r in enumerate(main_rows, 1) if r["dataset"] == old["dataset"] and r["metric"] == "Post-update AULC"]
            if len(selected) != 1:
                raise ValueError(f"Main AULC replacement ambiguity at row {number}")
            replacement(main_copy, main_rows, selected[0], level="same_contrast_changed_metric_definition")
            entry.update(result_version_status="superseded_aulc_metric_definition", historical_metric_definition="inclusive_epoch0_included",
                metric_definition_evidence="Historical convergence_results/paired.csv validation_aulc compared with archived post-update main registry; metric revised from AULC_0T to AULC_1T.",
                status_reason="Historical AULC includes Epoch 0. Current main result is the post-update AULC comparison; the replacement registry supplies estimates/CI, not replacement Holm p-values.")
        elif old["family"] == "cross_architecture" and old["metric"] == "validation_aulc":
            selected = [(i, r) for i, r in enumerate(cross_rows, 1) if r["dataset"] == old["dataset"]
                and r["architecture"] == old["model"] and r["method"] == normalized_method(hypothesis["method"])
                and r["reference"] == hypothesis["reference"]]
            entry.update(historical_metric_definition="inclusive_epoch0_included",
                metric_definition_evidence="Historical cross-architecture validation_aulc used AULC_0T; archived replacement explicitly records epoch0_excluded=True and AULC_1T.")
            if len(selected) == 1:
                replacement(cross_copy, cross_rows, selected[0], level="same_contrast_changed_metric_definition")
                entry.update(result_version_status="superseded_aulc_metric_definition",
                    status_reason="Historical AULC includes Epoch 0. Replacement excludes Epoch 0 and supplies its own retrospective correction scope; old p-values are not carried over to the revised metric.")
            elif not selected:
                replacement(cross_copy, cross_rows, level="related_current_family_not_a_matching_replacement")
                entry.update(result_version_status="historical_inclusive_aulc_replacement_unresolved", mapping_status="unresolved_matching_contrast_absent",
                    status_reason="Current cross-architecture AULC registry contains lowrank_wmf only. This historical di_wmf contrast has no matching post-update replacement; linked family file is context, not a substitute result.")
            else:
                raise ValueError(f"Ambiguous cross-architecture AULC replacement: {number}")
        elif number in efficiency:
            audit = efficiency[number]
            entry.update(historical_metric_definition=audit["aulc_definition"],
                metric_definition_evidence=f"{relative(SOURCES / 'parameter_efficiency_aulc_evidence_18.csv')} registry_row_1based={number}; means and paired difference independently recomputed from saved epoch histories.")
            if audit["aulc_definition"] == "unresolved":
                entry.update(result_version_status="historical_metric_definition_unresolved", mapping_status="unresolved_aulc_definition",
                    status_reason="Saved history recomputation does not uniquely identify inclusive versus post-update AULC; no definition was guessed.")
            else:
                entry.update(status_reason="Saved historical reference mean, candidate mean and paired difference match the independently recomputed AULC definition. No replacement is required by this bounded metric-definition check; other aspects are not recertified.")
        result.append(entry)
    if len(first_used) != 378:
        raise ValueError(f"First-layer mapping is not bijective: {len(first_used)} rows used")
    index = REPAIR / "current_result_index.csv"
    write_csv(index, fields + added_fields, result)
    _, check = read_csv(index)
    if len(check) != 902 or any(any(new[field] != old[field] for field in fields) for old, new in zip(original, check)):
        raise ValueError("Original registry field preservation check failed")
    if sha256(REGISTRY) != original_hash:
        raise ValueError("Original registry was modified")
    counts = dict(sorted(Counter(row["result_version_status"] for row in result).items()))
    summary = {"schema_version": 1, "index": relative(index), "index_sha256": sha256(index), "registry_rows": 902,
        "registry_source_sha256_before": original_hash, "registry_source_sha256_after": sha256(REGISTRY),
        "original_fields_preserved_exactly": True, "p_values_recomputed": False, "training_performed": False,
        "all_registry_analysis_timing": "retrospective", "all_registry_preregistered": False,
        "status_counts": counts, "family_counts": dict(sorted(Counter(row["family"] for row in result).items())),
        "first_layer_one_to_one_replacements": len(first_used), "physiology_historical_rows": 79,
        "physiology_protocol_replacement_hypotheses": len(physio_rows), "main_aulc_replacements": 3,
        "cross_architecture_aulc_matched_replacements": sum(row["family"] == "cross_architecture" and row["mapping_status"] == "matched" for row in result),
        "unresolved_registry_rows": [row["registry_row_1based"] for row in result if "unresolved" in row["mapping_status"]],
        "efficiency_aulc_definition_counts": dict(Counter(a["aulc_definition"] for a in efficiency.values())),
        "source_manifest": SOURCE_MANIFEST,
        "limitations": ["A retained version is not a correctness certificate.", "Original p-values must not be interpreted as tests of changed metrics or repaired protocols.",
            "Physiology replacement is protocol-level, not a 79-to-120 row join.", "The main post-update registry has no replacement Holm p-value column.",
            "Three full DI-WMF cross-architecture AULC contrasts are absent from the six-row low-rank replacement registry."]}
    (REPAIR / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status_table = "\n".join(f"| {status} | {count} |" for status, count in counts.items())
    (REPAIR / "current_result_index_README.md").write_text(f"""# Current result index

This additive index has exactly 902 rows, in the original order of
`statistical_controls/all_core_hypotheses_holm.csv`. All original columns,
including p-values and significance flags, are preserved as their exact CSV
field strings. The original file is unchanged (SHA-256 `{original_hash}`).

## Status counts

| Result-version status | Rows |
| --- | ---: |
{status_table}

## Reading the index

`registry_row_1based` starts at 1 for the first hypothesis. `registry_csv_line_1based`
adds the header line (first hypothesis is CSV line 2). Replacement row and line
pointers follow the same convention. All replacement paths are project-root-relative
and point to preserved files under `output/repairs/FYP6_REPAIR_20260908/index_sources/`,
so the index does not require the separately excluded `thesis/` directory.
`original_identity` includes the original source-row number as well as its hypothesis
JSON: some parameter-efficiency hypothesis JSON omits compact model width, so the
source-row identity must not be dropped when joining.

`result_version_status` is separate from `analysis_timing` and `preregistered`.
All 902 rows are retrospective and not preregistered. The original
`submission_stage_role` labels are preserved historical metadata, not timing claims.
A retained historical result is simply not replaced within this bounded mapping;
it is not a blanket correctness certificate.

## Replacement boundaries

- First-layer: 72 training + 18 selectivity + 288 corruption rows have a bijective
  378-row replacement under shared downstream controls. Matching preserves dataset,
  method/reference, metric, corruption and correction scope. Historical
  `lowrank_wmf_rank16` is normalized to `lowrank_wmf` only for the join. Study labels
  change intentionally. No p-values are recomputed or substituted in the old columns.
- Physiology: 24 + 20 + 35 historical rows are superseded at protocol level by 120
  repaired hypotheses using 46 patients, 53 records and metric-specific/manual-RR masks.
  There is no one-to-one correspondence. Blank replacement-row pointers are intentional.
- Main AULC: 3 historical inclusive-Epoch-0 rows point to the corresponding post-update
  AULC estimates in the preserved nine-row main registry. That registry has no replacement
  Holm p-values; historical significance flags do not transfer to the revised metric.
- Cross-architecture AULC: 6 low-rank rows match the six-row post-update replacement.
  The 3 historical full `di_wmf` contrasts are marked unresolved: the current file
  contains no matching contrast. Its link is related-family context only.
- Parameter-efficiency AULC: all 18 rows are checked against actual saved histories,
  not inferred from the current analyzer alone. Evidence compares reference means,
  candidate means and paired differences for both inclusive and post-update definitions.
  Classification counts: `{dict(Counter(a['aulc_definition'] for a in efficiency.values()))}`.
  The audit is in `index_sources/parameter_efficiency_aulc_evidence_18.csv` and
  history values/hashes are in `index_sources/parameter_efficiency_history_metrics.csv`.

Never treat the old p-values as tests of replacement metrics or repaired populations.
Replacement scopes and analyses remain retrospective. This is a result-version index,
not a fresh omnibus hypothesis test and not a new preregistration.

## Rebuild and checks

Run `python build_current_result_index.py` from the project root (standard library only).
No training, checkpoint loading or original result edits occur. Differing existing source
snapshots are rejected for explicit review instead of silently overwritten. When the
original main or cross-architecture registry under `thesis/` is absent from an archive,
the builder automatically uses its existing same-named `index_sources/` frozen copy.
Before reading that fallback, it verifies both its snapshot path and SHA-256 against
the original source's entry in the included `index_summary.json`. A missing manifest,
missing copy, unmatched provenance or changed hash stops rebuilding. The original
source path remains recorded as provenance; a verified fallback does not change the
index values, p-values or result-version statuses. Keep `index_summary.json` and both
frozen registry copies together when packaging without `thesis/`. The builder
checks 902-row preservation, original SHA-256, original field equality, a 378-row
bijective first-layer mapping, 120 replacement physiology rows and all 18 efficiency
AULC comparisons. Exact counts, hashes, preserved-source provenance and unresolved
row numbers are recorded in `index_summary.json`.
""", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("index", "registry_rows", "status_counts", "efficiency_aulc_definition_counts", "unresolved_registry_rows")}, indent=2))


if __name__ == "__main__":
    main()
