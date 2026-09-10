"""Independent audit of post-update AULC from saved original per-run histories.

Usage: python verify_aulc_from_histories.py --root PATH_TO_DOMAIN_MF_V2 --output OUTPUT
Requires Python >=3.10, numpy and scipy. Never modifies source experiment records.
All inferences are retrospective. Reconstructed CIs use paired Student t intervals;
Holm scopes are explicit in the exported paired table. Sign LOSO is unadjusted,
matching its original reported analysis, not implicitly added to the 902 registry.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

DATASETS = ("fashion", "cifar10", "sign")
STEMS = ("kaiming", "random_stem", "gabor", "pca", "kmeans", "di_wmf", "lowrank_wmf")


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(ps):
    result = [None] * len(ps)
    running = 0.0
    for rank, idx in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        running = max(running, (len(ps) - rank) * ps[idx])
        result[idx] = min(1.0, running)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs, checks, manifest = [], [], {}

    def record_file(path):
        path = Path(path)
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return rel

    def load_run(directory, study, dataset, method, unit, horizon):
        history = directory / "history.csv"
        rows = csv_read(history)
        epochs = [int(row["epoch"]) for row in rows]
        if sorted(epochs) != list(range(horizon + 1)):
            raise ValueError(f"Not exactly one observation for every epoch 0..{horizon}: {history}")
        by_epoch = {int(row["epoch"]): float(row["val_accuracy"]) for row in rows}
        post = float(np.mean([by_epoch[e] for e in range(1, horizon + 1)]))
        incl = float(np.mean([by_epoch[e] for e in range(horizon + 1)]))
        run = dict(study=study, dataset=dataset, method=method, unit=int(unit),
                   horizon=horizon, observed_epochs="0.." + str(horizon),
                   post_epochs="1.." + str(horizon), epoch0=by_epoch[0],
                   aulc_0T=incl, aulc_1T=post, history=record_file(history))
        for filename in ("config.json", "final_metrics.json", "control_summary.json", "loso_summary.json"):
            path = directory / filename
            if path.exists():
                record_file(path)
                if filename == "final_metrics.json":
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    value = stored.get("validation_aulc")
                    if value is not None:
                        run["stored_unqualified_aulc"] = value
                        run["stored_unqualified_definition"] = (
                            "inclusive_0T" if math.isclose(value, incl, abs_tol=1e-9) else
                            "post_update_1T" if math.isclose(value, post, abs_tol=1e-9) else "OTHER")
        runs.append(run)
        return run

    specs = [
        ("main50", "outputs_convergence50", "standard", {"kaiming", "lowrank_wmf"}, 50, 1.0),
        ("first_layer", "outputs_first_layer_cross_task", "standard", set(STEMS), 10, 0.1),
        ("resnet18", "outputs_resnet18_cross_task", "resnet18", {"kaiming", "di_wmf", "lowrank_wmf"}, 10, 0.1),
        ("correncoder", "outputs_correncoder_cross_task", "correncoder", {"kaiming", "lowrank_wmf"}, 10, 0.1),
    ]
    for study, folder, model, methods, horizon, fraction in specs:
        for cp in sorted((root / folder).glob("*/config.json")):
            c = json.loads(cp.read_text(encoding="utf-8"))
            seed_limit = 10 if study == "main50" or (study == "first_layer" and c["dataset"] == "sign") else 5
            if (c.get("dataset") not in DATASETS or c.get("model") != model or c.get("init") not in methods
                or int(c.get("epochs", 0)) != horizon or float(c.get("train_fraction", 0)) != fraction
                or int(c.get("seed", -1)) not in range(seed_limit)):
                continue
            load_run(cp.parent, study, c["dataset"], c["init"], c["seed"], horizon)
    for sp in sorted((root / "outputs_convergence50_fitted_head_control").glob("*/control_summary.json")):
        s = json.loads(sp.read_text(encoding="utf-8"))
        if int(s["seed"]) in range(10) and s["dataset"] in DATASETS:
            r = load_run(sp.parent, "fitted_head", s["dataset"], "kaiming_fitted_head", s["seed"], 50)
            assert math.isclose(r["aulc_1T"], float(s["post_training_aulc"]), abs_tol=1e-10)
    for sp in sorted((root / "sign_user_loso_results").glob("test_user_*/*/loso_summary.json")):
        s = json.loads(sp.read_text(encoding="utf-8"))
        r = load_run(sp.parent, "sign_loso", "sign", sp.parent.name, int(sp.parent.parent.name[-2:]), 50)
        assert math.isclose(r["aulc_1T"], float(s["post_training_aulc"]), abs_tol=1e-10)

    expected = {"main50": 60, "first_layer": 140, "resnet18": 45, "correncoder": 30, "fitted_head": 30, "sign_loso": 14}
    counts = dict(Counter(r["study"] for r in runs))
    if counts != expected:
        raise ValueError(f"Unexpected formal-run counts: {counts} vs {expected}")
    index = {(r["study"], r["dataset"], r["method"], r["unit"]): r for r in runs}
    assert len(index) == len(runs), "Duplicate formal run key"
    pairs = []

    def paired(study, dataset, candidate, reference, units, cand_study=None, ref_study=None, scope=None):
        ca = np.asarray([index[(cand_study or study, dataset, candidate, u)]["aulc_1T"] for u in units])
        re = np.asarray([index[(ref_study or study, dataset, reference, u)]["aulc_1T"] for u in units])
        delta = ca - re
        avg, sd = float(delta.mean()), float(delta.std(ddof=1))
        half = float(stats.t.ppf(0.975, len(delta) - 1)) * sd / math.sqrt(len(delta))
        p = float(stats.ttest_rel(ca, re).pvalue) if sd > 0 else (1.0 if avg == 0 else 0.0)
        row = dict(study=study, dataset=dataset, candidate=candidate, reference=reference,
                   metric="AULC_1T", n=len(delta), units=json.dumps(list(units)),
                   reference_mean=float(re.mean()), method_mean=float(ca.mean()),
                   mean_paired_difference=avg, paired_sd=sd, ci95_low=avg-half, ci95_high=avg+half,
                   p_value_raw=p, cohen_dz=(avg / sd if sd else None),
                   correction_scope=scope or "none_unadjusted_signer_comparison",
                   analysis_status="retrospective_not_preregistered")
        pairs.append(row)
        return row

    for ds in DATASETS:
        paired("main50", ds, "lowrank_wmf", "kaiming", range(10), scope="main50|AULC_1T|three_datasets")
        paired("head_fit_effect_on_kaiming_conv", ds, "kaiming_fitted_head", "kaiming", range(10),
               cand_study="fitted_head", ref_study="main50", scope="fitted_head|head_fit_effect|AULC_1T|three_datasets")
        paired("di_conv_effect_with_fitted_head", ds, "lowrank_wmf", "kaiming_fitted_head", range(10),
               cand_study="main50", ref_study="fitted_head", scope="fitted_head|di_conv_effect|AULC_1T|three_datasets")
        for method in STEMS[1:]:
            paired("first_layer", ds, method, "kaiming", range(10 if ds == "sign" else 5),
                   scope=f"first_layer|{ds}|AULC_1T|six_methods")
        for architecture in ("resnet18", "correncoder"):
            paired(architecture, ds, "lowrank_wmf", "kaiming", range(5), scope=f"{architecture}|lowrank|AULC_1T|three_datasets")
        paired("resnet18_diagonal_legacy", ds, "di_wmf", "kaiming", range(5), cand_study="resnet18", ref_study="resnet18",
               scope="resnet18|diagonal_legacy|AULC_1T|three_datasets")
    paired("sign_loso", "sign", "lowrank_wmf", "kaiming", (3, 4, 5, 6, 7, 9, 10))
    for scope in {r["correction_scope"] for r in pairs}:
        selected = [r for r in pairs if r["correction_scope"] == scope]
        adjusted = holm([r["p_value_raw"] for r in selected])
        for r, p in zip(selected, adjusted):
            r["scope_hypotheses"] = len(selected)
            r["p_value_holm"] = p if not scope.startswith("none_") else ""
            r["significant_0p05"] = p < 0.05

    def compare(row, path, predicate, mapping, tol=1e-9):
        record_file(path)
        source = [r for r in csv_read(path) if predicate(r)]
        assert len(source) == 1, (str(path), row, source)
        source = source[0]
        for recomputed, recorded in mapping.items():
            diff = float(row[recomputed]) - float(source[recorded])
            checks.append(dict(study=row["study"], dataset=row["dataset"], candidate=row["candidate"],
                               source=Path(path).relative_to(root).as_posix(), field=recorded,
                               recomputed=row[recomputed], recorded=source[recorded],
                               absolute_error=abs(diff), tolerance=tol, passed=abs(diff) <= tol))

    default_map = {k:k for k in ("mean_paired_difference", "ci95_low", "ci95_high", "p_value_raw")}
    for r in pairs:
        ds, method, study = r["dataset"], r["candidate"], r["study"]
        if study == "main50":
            compare(r, root / "submission_control_results/aulc_recomputed_paired.csv", lambda x:x["dataset"]==ds and x["metric"]=="aulc_1_to_50", default_map)
            compare(r, root / "submission_control_results/final_main_ten_seed_registry.csv", lambda x:x["dataset"]==ds and x["metric"]=="Post-update AULC", {k:k for k in ("mean_paired_difference", "ci95_low", "ci95_high")})
        elif study in ("head_fit_effect_on_kaiming_conv", "di_conv_effect_with_fitted_head"):
            compare(r, root / "fitted_head_training_control_results/training_control_paired.csv", lambda x:x["dataset"]==ds and x["contrast"]==study and x["metric"]=="post_training_aulc", default_map | {"p_value_holm":"p_value_holm_across_tasks"})
        elif study in ("resnet18_diagonal_legacy", "correncoder"):
            arch = "resnet18" if study.startswith("resnet18") else study
            compare(r, root / "submission_control_results/cross_architecture_post_aulc.csv", lambda x:x["dataset"]==ds and x["architecture"]==arch, default_map | {"p_value_holm":"p_value_holm_across_tasks"})
        elif study == "first_layer":
            stored_method = "lowrank_wmf_rank16" if method == "lowrank_wmf" else method
            compare(r, root / "first_layer_cross_task_results/paired_training.csv", lambda x:x["dataset"]==ds and x["method"]==stored_method and x["metric"]=="validation_aulc", {k:k for k in ("mean_paired_difference", "ci95_low", "ci95_high")}, tol=0.002)
            compare(r, root / "statistical_controls/all_core_hypotheses_holm.csv", lambda x:x["dataset"]==ds and x["family"]=="first_layer_training" and json.loads(x["hypothesis"]).get("method")==stored_method and x["metric"]=="validation_aulc", {"p_value_raw":"p_value_raw", "p_value_holm":"p_value_holm"})
        elif study == "sign_loso":
            compare(r, root / "sign_user_loso_results/sign_user_loso_paired.csv", lambda x:x["metric"]=="post_training_aulc", default_map)

    # Recompute all 902 Holm adjustments from archived raw p values; preserve the
    # registry as a historical record, and distinguish inclusive AULC rows.
    archived = csv_read(root / "statistical_controls/all_core_hypotheses_holm.csv")
    archive_check = []
    for field, key, adjusted_field in (("scope", "correction_scope", "p_value_holm"), ("family", "family", "p_value_holm_family_wide")):
        groups = defaultdict(list)
        for i, r in enumerate(archived):
            groups[r[key]].append(i)
        for group, ids in groups.items():
            vals = holm([float(archived[i]["p_value_raw"]) for i in ids])
            for i, p in zip(ids, vals):
                error = abs(p - float(archived[i][adjusted_field]))
                archive_check.append(dict(row=i+1, correction=field, group=group, recorded=archived[i][adjusted_field], recomputed=p, absolute_error=error, passed=error<1e-12))
    legacy_aulc = []
    legacy_raw_checks = []
    for i, r in enumerate(archived):
        if "aulc" not in r["metric"].lower():
            continue
        status = "unverified_other_tier"
        if r["family"] in {"main_cross_task", "cross_architecture"}:
            status = "legacy_inclusive_0T_not_evidence_for_post_update_claim"
            identity = json.loads(r["hypothesis"])
            study = "main50" if r["family"] == "main_cross_task" else r["model"]
            candidate = identity["method"].replace("_rank16", "")
            n = int(r["n_paired"])
            ca = np.asarray([index[(study, r["dataset"], candidate, u)]["aulc_0T"] for u in range(n)])
            re = np.asarray([index[(study, r["dataset"], "kaiming", u)]["aulc_0T"] for u in range(n)])
            delta = ca - re
            raw_p = float(stats.ttest_rel(ca, re).pvalue)
            for field, value in (("paired_difference", float(delta.mean())), ("p_value_raw", raw_p)):
                error = abs(value - float(r[field]))
                legacy_raw_checks.append(dict(registry_row=i+1, family=r["family"], dataset=r["dataset"],
                    model=r["model"], method=candidate, definition="AULC_0T", field=field,
                    recomputed=value, recorded=r[field], absolute_error=error, passed=error<1e-9))
        elif r["family"] == "first_layer_training":
            status = "post_update_1T_verified_against_raw_histories"
        legacy_aulc.append({"registry_row":i+1, "audit_metric_status":status, **r})

    replacement=[]
    for r in pairs:
        if r["study"] not in ("resnet18", "correncoder"):
            continue
        replacement.append(dict(architecture=r["study"], dataset=r["dataset"], method=r["candidate"], reference="kaiming", paired_seeds=r["n"],
            kaiming_mean_aulc_1_to_10=r["reference_mean"], di_mean_aulc_1_to_10=r["method_mean"], mean_paired_difference=r["mean_paired_difference"],
            ci95_low=r["ci95_low"], ci95_high=r["ci95_high"], p_value_raw=r["p_value_raw"], p_value_holm_across_tasks=r["p_value_holm"],
            holm_significant_0p05=r["significant_0p05"]))
    csv_write(out / "raw_history_aulc_rows.csv", runs)
    csv_write(out / "independent_post_update_paired.csv", pairs)
    csv_write(out / "stored_result_comparisons.csv", checks)
    csv_write(out / "archived_902_holm_verification.csv", archive_check)
    csv_write(out / "archived_aulc_metric_status.csv", legacy_aulc)
    csv_write(out / "archived_inclusive_aulc_raw_checks.csv", legacy_raw_checks)
    csv_write(out / "cross_architecture_post_aulc_CORRECTED_LOWRANK.csv", replacement)
    summary=dict(run_count=len(runs), runs_by_study=counts, post_update_epochs_verified=True,
                 recorded_comparisons=len(checks), comparisons_passed=sum(r["passed"] for r in checks),
                 archive_holm_checks=len(archive_check), archive_holm_checks_passed=sum(r["passed"] for r in archive_check),
                 archived_inclusive_raw_checks=len(legacy_raw_checks), archived_inclusive_raw_checks_passed=sum(r["passed"] for r in legacy_raw_checks),
                 discovered_issue="ResNet18 original post-update analysis selects diagonal di_wmf, while surrounding thesis and final results use lowrank_wmf.",
                 other_limits=["Historical unqualified validation_aulc fields can include epoch 0; current code alone cannot establish old-run metric meaning.",
                               "902-row registry retains inclusive main/cross-architecture AULC; its tests are historical, not the post-update tests.",
                               "Sign LOSO p=0.032479 is unadjusted; shared-training-fold dependence is a limitation.",
                               "Exact scipy t intervals differ slightly from legacy rounded t critical constants in first-layer CI endpoints."],
                 script_sha256=sha(__file__))
    (out / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "source_file_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(replacement, indent=2))
    if not all(r["passed"] for r in checks + archive_check + legacy_raw_checks):
        raise SystemExit("Some comparisons failed: inspect exported CSVs")


if __name__ == "__main__":
    main()
