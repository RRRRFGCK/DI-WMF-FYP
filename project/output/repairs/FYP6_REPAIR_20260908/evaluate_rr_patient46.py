"""Patient-group-corrected RR evaluation; CPU-only, no neural training.

The sibling JSON fixes the repair-stage contrasts and RR recomputation rules.
Original CNN FFT frequency estimation is deliberately unchanged. No original
experiment outputs are overwritten. See --stage prepare for training-free work.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import t as student_t, ttest_rel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from rr_cpu_legacy import (
    load_bidmc_subjects, respiratory_rate, bandpass, respiratory_envelope,
    verify_legacy_cpu_equivalence,
)

EXPERIMENTS = (
    "pretrained_mse", "random_mse", "matched_1d_mse", "pretrained_corr",
    "pretrained_spectral", "pretrained_corr_spectral",
    "matched_layerwise_calibrated_mse", "pretrained_hard_lag",
    "pretrained_soft_lag", "depth1", "depth2_diagonal", "depth3_diagonal",
    "depth3_lowrank",
)
CLASSICAL = ("bandpass_fft", "envelope_fft", "bandpass_autocorrelation_corrected")
METHODS = EXPERIMENTS + CLASSICAL
WAVEFORM_METRICS = ("test_mse", "test_mae", "waveform_correlation", "max_lag_waveform_correlation")
RR_METRICS = ("rr_manual_primary_mae_bpm", "rr_fft_secondary_mae_bpm")
FS, WINDOW, STRIDE, WAVEFORM_LENGTH = 30, 918, 30, 14388
N_WINDOWS = 450


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to emit empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def corrected_autocorrelation_rate(values, sampling_hz=30, low_hz=.08, high_hz=.8):
    """Return (bpm, status), with bounded interpolated *local* periodic peak.

    Adjacent integer support is needed at period=37.5 samples (48 bpm).
    The constrained quadratic maximum is explicitly clipped to the physical
    lag interval; a boundary sample on a descending zero-lag lobe is never a
    candidate because a genuine local maximum is required first.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if sampling_hz <= 0 or not 0 < low_hz < high_hz < sampling_hz / 2:
        raise ValueError("Invalid sampling rate or respiratory band")
    if x.size < math.ceil(2 * sampling_hz / low_hz):
        return float("nan"), "too_short_for_two_slowest_cycles"
    if not np.isfinite(x).all():
        return float("nan"), "nonfinite_input"
    x = x - x.mean()
    energy = float(x @ x)
    if energy <= np.finfo(np.float64).eps * max(1, x.size):
        return float("nan"), "constant_or_near_zero_energy"
    ac = np.correlate(x, x, mode="full")[x.size-1:] / energy
    minimum, maximum = sampling_hz / high_hz, sampling_hz / low_hz
    peaks, _ = find_peaks(ac)
    peaks = peaks[(peaks >= math.floor(minimum)) &
                  (peaks <= math.ceil(maximum)) & (peaks < x.size-1) & (ac[peaks] > 0)]
    candidates = []
    for peak in peaks:
        denominator = ac[peak-1] - 2 * ac[peak] + ac[peak+1]
        delta = .5 * (ac[peak-1] - ac[peak+1]) / denominator if denominator < 0 else 0.0
        delta = float(np.clip(delta, -.5, .5))
        refined = float(peak + delta)
        lag = float(np.clip(refined, minimum, maximum))
        # Evaluate the fitted quadratic at the constrained peak, not beyond it.
        offset = lag - peak
        height = float(ac[peak] + .5 * (ac[peak+1]-ac[peak-1]) * offset + .5 * denominator * offset**2)
        if height > 0:
            candidates.append((height, -lag, lag, refined != lag))
    if not candidates:
        return float("nan"), "no_positive_local_peak"
    _, _, lag, clipped = max(candidates)
    bpm = 60 * sampling_hz / lag
    assert 60 * low_hz <= bpm <= 60 * high_hz
    return float(bpm), "ok_boundary_constrained" if clipped else "ok"


def fft_rates(waveform):
    """Call the unchanged historical estimator once per exact 918-point window."""
    signal = np.asarray(waveform).reshape(-1)
    if signal.size != WAVEFORM_LENGTH or not np.isfinite(signal).all():
        raise ValueError("Expected finite 14388-sample reconstructed waveform")
    return np.asarray([respiratory_rate(signal[s:s+WINDOW], FS)
                       for s in range(0, WAVEFORM_LENGTH-WINDOW+1, STRIDE)])


def annotation_rates(samples, source_hz, starts):
    annotations = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not np.isfinite(annotations).all() or np.any(annotations < 1):
        raise ValueError("Annotations must be finite positive MATLAB sample indices")
    # The data contain two duplicated timestamps and two out-of-order lists.
    # Event-time set semantics make first/last chronological and avoid counting
    # one event twice. All changes are recorded in a separate source registry.
    annotations = np.unique(annotations)
    times = (annotations - 1) / source_hz
    rates, counts = [], []
    for start in starts:
        events = times[(times >= start) & (times < start + WINDOW / FS)]
        counts.append(len(events))
        rates.append(60 * (len(events)-1) / (events[-1]-events[0])
                     if len(events) >= 2 else np.nan)
    return np.asarray(rates), np.asarray(counts)


def unit_tests():
    time = np.arange(WINDOW) / FS
    rows = []
    for bpm in (4.8, 5, 6, 17, 18, 48):
        estimate, status = corrected_autocorrelation_rate(np.sin(2*np.pi*bpm/60*time))
        assert np.isfinite(estimate) and abs(estimate-bpm) <= .15
        assert 4.8 <= estimate <= 48
        rows.append({"case":f"pure_sine_{bpm:g}_bpm", "expected_bpm":bpm,
                     "estimate_bpm":estimate, "status":status, "passed":True})
    for label, values, expected in (
        ("constant",np.ones(WINDOW),"constant_or_near_zero_energy"),
        ("short_signal",np.sin(np.arange(100)),"too_short_for_two_slowest_cycles"),
        ("no_peak_ramp",np.arange(WINDOW,dtype=float),"no_positive_local_peak"),
        ("nonfinite",np.full(WINDOW,np.nan),"nonfinite_input"),
    ):
        estimate, status = corrected_autocorrelation_rate(values)
        assert np.isnan(estimate) and status == expected
        rows.append({"case":label,"expected_bpm":None,"estimate_bpm":None,"status":status,"passed":True})
    rate, counts = annotation_rates(np.array([1,126,251,376]),125,np.array([0.,2.,3.]))
    assert rate[0] == 60 and rate[1] == 60 and np.isnan(rate[2])
    assert counts.tolist() == [4,2,1]
    # Exclude events exactly at window end; no accidental inclusive extra cycle.
    rate, counts = annotation_rates(np.array([1,125*30.6+1]),125,np.array([0.]))
    assert counts[0] == 1 and np.isnan(rate[0])
    rate, counts = annotation_rates(np.array([251,1,126,126]),125,np.array([0.]))
    assert counts[0] == 3 and rate[0] == 60
    return rows


def prepare_classical(output):
    output.mkdir(parents=True, exist_ok=True)
    legacy_ast_checks = verify_legacy_cpu_equivalence(ROOT)
    write_json(output / "rr_unit_tests.json", {"passed":True,"tests":unit_tests()})
    raw_path = ROOT / "data_bidmc" / "bidmc_data.mat"
    raw = loadmat(raw_path, simplify_cells=True)["data"]
    subjects = load_bidmc_subjects(raw_path)
    patient_ids = np.asarray([record["fix"]["id"] for record in raw])
    assert len(subjects) == 53 and len(set(patient_ids)) == 46
    starts = np.arange(N_WINDOWS) * STRIDE / FS
    ann1, ann2, counts1, counts2, fft_reference, targets = [], [], [], [], [], []
    estimates = {method:[] for method in CLASSICAL}
    ac_status = []; annotation_registry = []
    for fold, (record, subject) in enumerate(zip(raw, subjects)):
        source_hz = float(record["ppg"]["fs"])
        for annotator in ("ann1","ann2"):
            a = np.asarray(record["ref"]["breaths"][annotator],dtype=float).reshape(-1)
            annotation_registry.append({"record_fold":fold,"patient_id":patient_ids[fold],
                "annotator":annotator,"original_event_count":len(a),
                "unique_event_count":len(np.unique(a)),"exact_duplicate_count":len(a)-len(np.unique(a)),
                "descending_adjacent_pairs":int((np.diff(a)<0).sum()),
                "raw_sample_sequence_sha256":hashlib.sha256(a.astype("<f8").tobytes()).hexdigest(),
                "canonical_sample_sequence_sha256":hashlib.sha256(np.unique(a).astype("<f8").tobytes()).hexdigest(),
                "hash_encoding":"contiguous_little_endian_float64_sample_indices",
                "canonicalisation":"sort_unique_sample_indices"})
        r1, n1 = annotation_rates(record["ref"]["breaths"]["ann1"],source_hz,starts)
        r2, n2 = annotation_rates(record["ref"]["breaths"]["ann2"],source_hz,starts)
        ann1.append(r1); ann2.append(r2); counts1.append(n1); counts2.append(n2)
        target = subject["respiration"][:WAVEFORM_LENGTH].astype(float)
        targets.append(target); fft_reference.append(fft_rates(target))
        # Filtering uses the whole original 14400-sample preprocessed PPG.
        bp = bandpass(subject["ppg"],FS,.08,.8)
        env = respiratory_envelope(subject["ppg"],FS)
        estimates["bandpass_fft"].append(fft_rates(bp[:WAVEFORM_LENGTH]))
        estimates["envelope_fft"].append(fft_rates(env[:WAVEFORM_LENGTH]))
        ac = [corrected_autocorrelation_rate(bp[s:s+WINDOW])
              for s in range(0,WAVEFORM_LENGTH-WINDOW+1,STRIDE)]
        estimates["bandpass_autocorrelation_corrected"].append([r[0] for r in ac])
        ac_status.append([r[1] for r in ac])
    payload = {"patient_ids":patient_ids,"ann1":np.asarray(ann1),"ann2":np.asarray(ann2),
               "ann1_counts":np.asarray(counts1),"ann2_counts":np.asarray(counts2),
               "fft_reference":np.asarray(fft_reference),"target_waveforms":np.asarray(targets),
               "autocorrelation_status":np.asarray(ac_status),
               **{method:np.asarray(values) for method,values in estimates.items()}}
    np.savez_compressed(output / "rr_classical_prepared.npz",**payload)
    write_csv(output / "annotation_canonicalisation_registry.csv",annotation_registry)
    manual_valid = np.isfinite(payload["ann1"]) & np.isfinite(payload["ann2"])
    manual_rates = ((payload["ann1"]+payload["ann2"])/2)[manual_valid]
    disagreement = abs(payload["ann1"]-payload["ann2"])[manual_valid]
    status_counts = dict(Counter(payload["autocorrelation_status"].reshape(-1)))
    manifest = {"status":"classical_prepared_not_final_all_method_evaluation",
                "data_mat_sha256":sha256(raw_path),
                "prepared_npz_sha256":sha256(output/"rr_classical_prepared.npz"),
                "cpu_helper_sha256":sha256(HERE/"rr_cpu_legacy.py"),
                "legacy_cpu_AST_equivalence_checks":legacy_ast_checks,
                "protocol_sha256":sha256(HERE / "rr_evaluation_protocol.json"),
                "script_sha256":sha256(__file__),"records":53,"patients":46,
                "total_windows":53*N_WINDOWS,"dual_annotation_valid_windows":int(manual_valid.sum()),
                "manual_reference_quality_descriptive_only":{
                    "minimum_bpm":float(manual_rates.min()),"maximum_bpm":float(manual_rates.max()),
                    "windows_below_4p8_bpm_retained":int((manual_rates<4.8).sum()),
                    "windows_above_48_bpm_retained":int((manual_rates>48).sum()),
                    "annotator_absolute_difference_median_bpm":float(np.median(disagreement)),
                    "annotator_absolute_difference_p95_bpm":float(np.percentile(disagreement,95))},
                "annotation_anomalies":[row for row in annotation_registry if row["exact_duplicate_count"] or row["descending_adjacent_pairs"]],
                "autocorrelation_status_counts":status_counts,
                "autocorrelation_finite_estimates_within_4p8_48_bpm":bool(np.all((payload[CLASSICAL[2]][np.isfinite(payload[CLASSICAL[2]])]>=4.8)&(payload[CLASSICAL[2]][np.isfinite(payload[CLASSICAL[2]])]<=48)))}
    write_json(output / "rr_preparation_manifest.json",manifest)
    print(json.dumps(manifest,indent=2),flush=True)


def mean_ci(values):
    values = np.asarray(values,dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid finite-patient statistics vector")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(student_t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values)))
    return {"mean":mean,"sd":sd,"ci95_low":mean-half,"ci95_high":mean+half}


def paired_row(method, reference, metric, first, second, n_windows="not_applicable", n_records=53,
               window_policy="not_applicable_waveform"):
    first, second = np.asarray(first,dtype=float), np.asarray(second,dtype=float)
    diff = first-second
    if len(diff) < 2 or not np.isfinite(diff).all():
        raise ValueError("Paired registry requires finite matched patient groups")
    direction = 1 if "correlation" in metric else -1
    sd = float(diff.std(ddof=1))
    p = float(ttest_rel(first,second).pvalue) if sd > 0 else (1.0 if diff.mean()==0 else 0.0)
    return {"family_id":"frozen_repaired_physiology_120_exploratory",
            "method":method,"reference":reference,"metric":metric,"n_patients":len(diff),
            "n_records":n_records,"n_windows":n_windows,"window_policy":window_policy,
            "effect_definition":"method_minus_reference","higher_is_better":direction==1,
            "method_mean":float(first.mean()),"reference_mean":float(second.mean()),
            **mean_ci(diff),"p_value_raw":p,"p_value_holm":None,
            "cohen_dz":float(diff.mean()/sd) if sd else None,
            "effect_size_status":"defined" if sd else "zero_difference_variance",
            "method_wins":int((direction*diff>0).sum()),
            "reference_wins":int((direction*diff<0).sum()),"ties":int((diff==0).sum()),
            "interpretation":"exploratory_descriptive_CV_dependence_single_pretrain_conditional"}


def export_latex_values(output, summaries, pairs, manifest):
    def format_interval(row, decimals=3):
        return f"{row['mean']:.{decimals}f} [{row['ci95_low']:.{decimals}f}, {row['ci95_high']:.{decimals}f}]"
    def format_p(value):
        if value == 0:
            return r"<10^{-300}"
        if value >= .001:
            return f"{value:.3f}"
        exponent = int(math.floor(math.log10(value)))
        return f"{value/10**exponent:.2f}" + r"\times10^{" + str(exponent) + "}"
    summary_map = {}
    for row in summaries:
        summary_map[row["method"]+"."+row["metric"]] = {
            **row,"mean_ci_latex":format_interval(row),
            "mean_latex":f"{row['mean']:.3f}",
            "ci95_half_width":(row["ci95_high"]-row["ci95_low"])/2}
    pair_map = {}
    for row in pairs:
        pair_map[row["method"]+"_vs_"+row["reference"]+"."+row["metric"]] = {
            **row,"effect_ci_latex":format_interval(row),
            "p_raw_latex":format_p(row["p_value_raw"]),
            "p_holm_latex":format_p(row["p_value_holm"])}
    write_json(output / "physiology_latex_values_patient46.json",{
        "status":"complete","protocol_id":manifest["protocol_id"],
        "family_id":manifest["inference_family_id"],"family_size":len(pairs),
        "n_patients":46,"interpretation":manifest["interpretation"],
        "summary":summary_map,"paired":pair_map})
    names = {"pretrained_mse":"Pretrained + MSE","bandpass_fft":"Band-pass FFT",
             "envelope_fft":"Envelope FFT","bandpass_autocorrelation_corrected":"Corrected autocorrelation"}
    lines = [r"% Generated from corrected patient-group predictions; see manifest and CSV.",
             r"\begin{tabular}{lrrr}",r"\toprule",
             r"Method & Manual RR MAE (95\% CI) & FFT RR MAE (95\% CI) & Coverage \\",r"\midrule"]
    for method in ("pretrained_mse",)+CLASSICAL:
        a = summary_map[method+"."+RR_METRICS[0]]
        b = summary_map[method+"."+RR_METRICS[1]]
        lines.append(names[method]+" & "+format_interval(a)+" & "+format_interval(b)+f" & {100*a['coverage']:.2f}\\% "+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",
        r"% Units: bpm. Patients have equal weight after averaging record means.",
        r"% Autocorrelation summary is coverage-conditioned; paired comparisons recompute",
        r"% pretrained MSE on identical available windows. Other methods keep all manual-valid windows.",
        r"% Intervals are descriptive patient-t intervals; shared CV training sets remain dependent."])
    (output/"classical_rr_table_patient46.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=[r"% Long-form frozen contrast registry; fragment intended for longtable/booktabs.",
           r"\begin{longtable}{lllrr}",r"\toprule",
           r"Method & Reference & Metric & Effect (95\% CI) & Holm $p$ \\",r"\midrule"]
    for row in pairs:
        escaped=lambda s:s.replace("_",r"\_")
        lines.append(" & ".join([escaped(row["method"]),escaped(row["reference"]),escaped(row["metric"]),
                                  format_interval(row),"$"+format_p(row["p_value_holm"])+"$"])+r" \\")
    lines.extend([r"\bottomrule",r"\end{longtable}"])
    (output/"physiology_paired_registry_patient46.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def evaluate_all(output, neural_root):
    # Refuse partial sets: final reporting and the frozen multiplicity family
    # must not depend on which methods happen to have completed first.
    incomplete = []
    for experiment in EXPERIMENTS:
        path = neural_root / experiment / "manifest.json"
        if not path.exists() or json.loads(path.read_text())["status"] != "complete":
            incomplete.append(experiment)
    if incomplete:
        raise RuntimeError("All 13 patient-group configurations must be complete before final RR comparison: " + ", ".join(incomplete))
    prep_manifest = json.loads((output / "rr_preparation_manifest.json").read_text())
    if prep_manifest["protocol_sha256"] != sha256(HERE / "rr_evaluation_protocol.json"):
        raise RuntimeError("Preparation rules changed; explicitly rerun prepare stage")
    if (prep_manifest["data_mat_sha256"] != sha256(ROOT/"data_bidmc"/"bidmc_data.mat") or
            prep_manifest["prepared_npz_sha256"] != sha256(output/"rr_classical_prepared.npz") or
            prep_manifest["cpu_helper_sha256"] != sha256(HERE/"rr_cpu_legacy.py")):
        raise RuntimeError("Source data or prepared reference cache changed")
    prepared = np.load(output / "rr_classical_prepared.npz",allow_pickle=False)
    ids = prepared["patient_ids"].tolist()
    unique_patients = sorted(set(ids)); counts = Counter(ids)
    rates = {method:prepared[method] for method in CLASSICAL}
    waveform_patients = {}; provenance = []; legacy_rr_replay_errors = []
    for experiment in EXPERIMENTS:
        rows = load_csv(neural_root / experiment / "record_metrics.csv")
        by_fold = {int(row["record_fold"]):row for row in rows}
        if set(by_fold) != set(range(53)) or len(rows) != 53:
            raise RuntimeError(f"Incomplete/duplicated record manifest for {experiment}")
        predictions = []
        for fold in range(53):
            row = by_fold[fold]
            reused = str(row["reused"]).lower() in {"true","1"}
            if row["patient_id"] != ids[fold] or reused != (counts[ids[fold]]==1):
                raise RuntimeError(f"Illegal identity/provenance at {experiment}/{fold}")
            wf_path = Path(row["waveforms"])
            if not wf_path.resolve().is_relative_to(neural_root.resolve()):
                raise RuntimeError("Only waveform paths in the repaired manifest workspace are accepted")
            group_path = neural_root / experiment / "groups" / ids[fold] / "group_manifest.json"
            group = json.loads(group_path.read_text())
            expected_test = {f for f in range(53) if ids[f]==ids[fold]}
            if (group.get("status") != "complete" or bool(group["reused"]) != reused or
                    set(group["test_record_folds"]) != expected_test or
                    set(group["train_record_folds"]) != set(range(53))-expected_test):
                raise RuntimeError(f"Incorrect complete patient-group exclusion manifest: {group_path}")
            if group.get("identity_intersection") or set(group["train_patient_ids"]) & set(group["test_patient_ids"]):
                raise RuntimeError(f"Patient leakage in group manifest {group_path}")
            if fold not in group["test_record_folds"]:
                raise RuntimeError("Waveform's record not held out by its patient-group model")
            with np.load(wf_path,allow_pickle=False) as waveforms:
                pred, target = waveforms["prediction"], waveforms["target"]
                if len(pred)!=WAVEFORM_LENGTH or target.shape != (WAVEFORM_LENGTH,):
                    raise RuntimeError(f"Unexpected waveform schema {wf_path}")
                if not np.allclose(target,prepared["target_waveforms"][fold],rtol=0,atol=1e-10):
                    raise RuntimeError(f"Target timing/value mismatch {wf_path}")
                pred_rates = fft_rates(pred)
                predictions.append(pred_rates)
                replayed_legacy_rr = float(np.mean(abs(pred_rates-fft_rates(target))))
                discrepancy = abs(replayed_legacy_rr-float(row["rr_mean_absolute_error_bpm_30p6s"]))
                if discrepancy > 1e-10:
                    raise RuntimeError(f"Unchanged FFT-vs-FFT RR replay mismatch: {wf_path}")
                legacy_rr_replay_errors.append(discrepancy)
            provenance.append({"method":experiment,"record_fold":fold,"patient_id":ids[fold],
                "reused":reused,"waveforms":str(wf_path),"waveform_sha256":sha256(wf_path),
                "group_manifest":str(group_path),"group_manifest_sha256":sha256(group_path)})
        rates[experiment] = np.asarray(predictions)
        patients = load_csv(neural_root / experiment / "patient_metrics.csv")
        pmap = {row["patient_id"]:row for row in patients}
        if set(pmap)!=set(unique_patients) or len(patients)!=46:
            raise RuntimeError(f"Expected exactly 46 patient rows: {experiment}")
        for metric in WAVEFORM_METRICS:
            values = []
            for patient in unique_patients:
                expected = np.mean([float(by_fold[f][metric]) for f in range(53) if ids[f]==patient])
                observed = float(pmap[patient][metric])
                if not np.isclose(expected,observed,rtol=1e-9,atol=1e-12):
                    raise RuntimeError(f"Patient aggregation mismatch: {experiment}/{patient}/{metric}")
                values.append(observed)
            waveform_patients[(experiment,metric)] = np.asarray(values)

    manual = (prepared["ann1"]+prepared["ann2"])/2
    fft_reference = prepared["fft_reference"]
    for method in EXPERIMENTS + CLASSICAL[:2]:
        if not np.isfinite(rates[method]).all():
            raise RuntimeError(f"Nonfinite FFT estimates must be investigated, not silently masked: {method}")
    if not np.isfinite(fft_reference).all():
        raise RuntimeError("Nonfinite impedance FFT reference")
    common = np.isfinite(manual)
    full_grid = np.ones_like(common,dtype=bool)
    if np.any(common.sum(1)==0):
        raise RuntimeError("A record has no jointly valid evaluation windows")
    record_rows, patient_rows, summary_rows = [], [], []
    patient_vectors = dict(waveform_patients)
    descriptive_metric = "rr_fft_full_grid_sensitivity_mae_bpm"
    def patient_values_on_mask(method, reference, mask):
        record_values = []
        for fold in range(53):
            error = abs(rates[method][fold][mask[fold]]-reference[fold][mask[fold]])
            record_values.append(float(error.mean()) if len(error) else np.nan)
        pvalues = []
        for patient in unique_patients:
            subset = np.asarray([record_values[f] for f in range(53) if ids[f]==patient])
            pvalues.append(float(np.mean(subset[np.isfinite(subset)])) if np.isfinite(subset).any() else np.nan)
        return np.asarray(record_values),np.asarray(pvalues)
    for method in METHODS:
        for metric, reference, reference_mask in (
            (RR_METRICS[0],manual,common), (RR_METRICS[1],fft_reference,common),
            (descriptive_metric,fft_reference,full_grid),
        ):
            mask = reference_mask & np.isfinite(rates[method])
            values_by_record,pvalues = patient_values_on_mask(method,reference,mask)
            for fold in range(53):
                error = abs(rates[method][fold][mask[fold]]-reference[fold][mask[fold]])
                record_rows.append({"method":method,"metric":metric,"record_fold":fold,
                    "patient_id":ids[fold],"n_windows":int(mask[fold].sum()),
                    "reference_valid_windows":int(reference_mask[fold].sum()),
                    "coverage":float(mask[fold].sum()/reference_mask[fold].sum()),
                    "mean_absolute_error_bpm":values_by_record[fold],
                    "median_absolute_error_bpm":float(np.median(error)) if len(error) else np.nan})
            for patient,value in zip(unique_patients,pvalues):
                folds = [f for f in range(53) if ids[f]==patient]
                patient_rows.append({"method":method,"metric":metric,"patient_id":patient,
                    "record_count":len(folds),"record_folds":";".join(map(str,folds)),
                    "n_windows":int(mask[folds].sum()),"value":value})
            patient_vectors[(method,metric)] = pvalues
            finite_patients = pvalues[np.isfinite(pvalues)]
            summary_rows.append({"method":method,"metric":metric,"n_patients":len(finite_patients),
                "n_records":int(np.isfinite(values_by_record).sum()),
                "n_windows":int(mask.sum()),"reference_valid_windows":int(reference_mask.sum()),
                "coverage":float(mask.sum()/reference_mask.sum()),
                "window_policy":(
                    "full_grid_sensitivity" if method!=CLASSICAL[2] else "full_grid_autocorrelation_coverage_conditioned"
                ) if metric==descriptive_metric else (
                    "reference_only_common_mask" if method!=CLASSICAL[2] else "autocorrelation_coverage_conditioned"
                ),
                **mean_ci(finite_patients),
                "evidence_role":"primary_manual_reference" if metric==RR_METRICS[0] else "secondary_sensitivity",
                "inference_caveat":"descriptive_patient_t_interval_CV_training_overlap"})
    for (method,metric), values in waveform_patients.items():
        summary_rows.append({"method":method,"metric":metric,"n_patients":46,"n_records":53,
            "n_windows":"not_applicable","reference_valid_windows":"not_applicable","coverage":1.0,
            "window_policy":"not_applicable_waveform",**mean_ci(values),"evidence_role":"corrected_waveform_endpoint",
            "inference_caveat":"descriptive_patient_t_interval_CV_training_overlap"})
        for patient, value in zip(unique_patients,values):
            folds=[f for f in range(53) if ids[f]==patient]
            patient_rows.append({"method":method,"metric":metric,"patient_id":patient,
                "record_count":len(folds),"record_folds":";".join(map(str,folds)),
                "n_windows":"not_applicable","value":float(value)})
    pairs = []
    protocol = json.loads((HERE/"rr_evaluation_protocol.json").read_text())
    neural_contrasts = protocol["neural_contrasts"]
    assert len(neural_contrasts)==19 and len(set(tuple(pair) for pair in neural_contrasts))==19
    for metric in WAVEFORM_METRICS + RR_METRICS:
        for first, second in neural_contrasts:
            pairs.append(paired_row(first,second,metric,patient_vectors[first,metric],patient_vectors[second,metric],
                n_windows=int(common.sum()) if metric in RR_METRICS else "not_applicable",
                window_policy="reference_only_common_mask" if metric in RR_METRICS else "not_applicable_waveform"))
    ac_paired_patient_rows = []
    for method in CLASSICAL:
        pair_mask = common & np.isfinite(rates[method]) & np.isfinite(rates["pretrained_mse"])
        for metric, reference in ((RR_METRICS[0],manual),(RR_METRICS[1],fft_reference)):
            _, first = patient_values_on_mask(method,reference,pair_mask)
            _, second = patient_values_on_mask("pretrained_mse",reference,pair_mask)
            selected = np.isfinite(first) & np.isfinite(second)
            policy = "reference_only_common_mask" if method!=CLASSICAL[2] else "paired_complete_autocorrelation_windows"
            pairs.append(paired_row(method,"pretrained_mse",metric,first[selected],second[selected],
                n_windows=int(pair_mask.sum()),n_records=int((pair_mask.sum(1)>0).sum()),window_policy=policy))
            for patient,a,b,included in zip(unique_patients,first,second,selected):
                folds=[f for f in range(53) if ids[f]==patient]
                ac_paired_patient_rows.append({"method":method,"reference":"pretrained_mse","metric":metric,
                    "patient_id":patient,"included":bool(included),"window_policy":policy,
                    "n_windows":int(pair_mask[folds].sum()),"method_value":a,"reference_value":b})
    assert len(pairs)==120
    ordering = np.argsort([row["p_value_raw"] for row in pairs],kind="stable")
    previous = 0.0
    for position, index in enumerate(ordering):
        corrected = max(previous,min(1.0,(len(pairs)-position)*pairs[index]["p_value_raw"]))
        pairs[index]["p_value_holm"] = corrected; previous=corrected
    window_rows = []
    for fold in range(53):
        for window in range(N_WINDOWS):
            row = {"record_fold":fold,"patient_id":ids[fold],"window_index":window,
                "start_seconds":window,"stop_seconds":window+WINDOW/FS,
                "ann1_event_count":int(prepared["ann1_counts"][fold,window]),
                "ann2_event_count":int(prepared["ann2_counts"][fold,window]),
                "ann1_rate_bpm":prepared["ann1"][fold,window],"ann2_rate_bpm":prepared["ann2"][fold,window],
                "manual_reference_bpm":manual[fold,window],"fft_reference_bpm":fft_reference[fold,window],
                "primary_common_valid":bool(common[fold,window]),
                "autocorrelation_finite":bool(np.isfinite(rates[CLASSICAL[2]][fold,window])),
                "autocorrelation_status":prepared["autocorrelation_status"][fold,window]}
            row.update({method+"_predicted_bpm":rates[method][fold,window] for method in METHODS})
            window_rows.append(row)
    write_csv(output / "rr_windows_patient46.csv",window_rows)
    write_csv(output / "rr_record_metrics_patient46.csv",record_rows)
    write_csv(output / "physiology_patient_metrics_patient46.csv",patient_rows)
    write_csv(output / "physiology_summary_patient46.csv",summary_rows)
    write_csv(output / "physiology_paired_holm_patient46.csv",pairs)
    write_csv(output / "classical_paired_patient_values_patient46.csv",ac_paired_patient_rows)
    write_csv(output / "waveform_provenance_patient46.csv",provenance)
    final = {"status":"complete","protocol_id":"patient46_dual_annotation_rr_v1_20260908",
        "evaluation_completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha256(HERE/"rr_evaluation_protocol.json"),"script_sha256":sha256(__file__),
        "preparation_manifest":prep_manifest,"methods":list(METHODS),"neural_configurations":13,
        "records_per_method":53,"patient_groups":46,"total_windows":53*N_WINDOWS,
        "dual_annotation_valid_windows":int(np.isfinite(manual).sum()),
        "primary_common_valid_windows":int(common.sum()),"fft_full_grid_windows":int(full_grid.sum()),
        "autocorrelation_valid_on_primary_mask":int((common & np.isfinite(rates[CLASSICAL[2]])).sum()),
        "inference_family_size":len(pairs),"inference_family_id":"frozen_repaired_physiology_120_exploratory",
        "reused_record_predictions":sum(row["reused"] for row in provenance),
        "new_group_record_predictions":sum(not row["reused"] for row in provenance),
        "neural_legacy_fft_rr_replay_max_absolute_error":max(legacy_rr_replay_errors),
        "interpretation":"Exploratory, conditional on one fixed CapnoBase pretraining checkpoint; patient t summaries do not remove CV dependence.",
        "table_hashes":{name:sha256(output/name) for name in (
            "rr_windows_patient46.csv", "rr_record_metrics_patient46.csv",
            "physiology_patient_metrics_patient46.csv", "physiology_summary_patient46.csv",
            "physiology_paired_holm_patient46.csv", "classical_paired_patient_values_patient46.csv",
            "waveform_provenance_patient46.csv")}}
    export_latex_values(output,summary_rows,pairs,final)
    artifact_names = tuple(final["table_hashes"]) + (
        "physiology_latex_values_patient46.json", "classical_rr_table_patient46.tex",
        "physiology_paired_registry_patient46.tex")
    # QA reports must not be hashed here: QA is run afterward and would create
    # a self-referential/stale hash on a second evaluation in the same folder.
    final["generated_artifact_hashes"] = {name:sha256(output/name) for name in artifact_names}
    write_json(output / "physiology_evaluation_manifest_patient46.json",final)
    print(json.dumps(final,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","evaluate","all"),default="all")
    parser.add_argument("--output-root",type=Path,default=HERE / "physiology_evaluation")
    parser.add_argument("--neural-root",type=Path,default=ROOT / "outputs_correncoder_patient46_20260908")
    args = parser.parse_args()
    if args.stage in {"prepare","all"}:
        prepare_classical(args.output_root)
    if args.stage in {"evaluate","all"}:
        evaluate_all(args.output_root,args.neural_root)


if __name__ == "__main__":
    main()
