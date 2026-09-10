"""Redraw only the thesis F14 PNG from manifest-admitted patient46 outputs.

No fitting, no changes to training code, no PDF output. Selection is fixed before
plotting: median-nearest patient for pretrained_corr, mean-nearest record within
that patient, then maximum target-variance 918-sample window (one-sample stride).
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
REPAIR = ROOT / "output/repairs/FYP6_REPAIR_20260908"
sys.path.insert(0, str(REPAIR))
from rr_cpu_legacy import load_bidmc_subjects, verify_legacy_cpu_equivalence, respiratory_rate

OUTPUT = ROOT / "outputs_correncoder_patient46_20260908"
FIGURE = REPAIR / "thesis/figures/main/F14_correncoder_representative_waveform.png"
JSON_FILE = FIGURE.with_suffix(".json")
EXPERIMENTS = ["pretrained_mse", "pretrained_corr", "matched_1d_mse"]
LABELS = {
    "pretrained_mse": "Pretrained + MSE",
    "pretrained_corr": "Pretrained + correlation",
    "matched_1d_mse": "1-D stem initializer + MSE",
}
COLORS = {"pretrained_mse": "#1479AC", "pretrained_corr": "#C25B20", "matched_1d_mse": "#008676"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    source_checks = verify_legacy_cpu_equivalence(ROOT)
    inputs = {}

    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return path

    track(__file__)
    track(REPAIR / "rr_cpu_legacy.py")
    track(ROOT / "run_correncoder_regression.py")
    global_manifest = json.loads(track(OUTPUT / "manifest.json").read_text())
    assert len(global_manifest["experiments"]) == 13
    assert all(r["status"] == "complete" for r in global_manifest["experiments"])

    patients = csv_read(track(OUTPUT / "pretrained_corr/patient_metrics.csv"))
    assert len(patients) == 46 and len({r["patient_id"] for r in patients}) == 46
    values = np.array([float(r["waveform_correlation"]) for r in patients])
    median = float(np.median(values))
    patient_distances = [{"patient_id": r["patient_id"], "waveform_correlation": float(r["waveform_correlation"]),
                          "distance_to_median": abs(float(r["waveform_correlation"]) - median)} for r in patients]
    closest_distance = min(r["distance_to_median"] for r in patient_distances)
    tied_patients = [r for r in patient_distances if abs(r["distance_to_median"] - closest_distance) <= 1e-12]
    chosen_patient = min(tied_patients, key=lambda r: r["patient_id"])
    patient = chosen_patient["patient_id"]
    records = csv_read(track(OUTPUT / "pretrained_corr/record_metrics.csv"))
    candidates = [r for r in records if r["patient_id"] == patient]
    patient_mean = float(np.mean([float(r["waveform_correlation"]) for r in candidates]))
    assert abs(patient_mean - chosen_patient["waveform_correlation"]) < 1e-12
    chosen_record = min(candidates, key=lambda r: (abs(float(r["waveform_correlation"]) - patient_mean), int(r["record_fold"])))
    fold = int(chosen_record["record_fold"])

    selected = {}
    arrays = {}
    group_records = {}
    for experiment in EXPERIMENTS:
        directory = OUTPUT / experiment
        manifest = json.loads(track(directory / "manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["records_completed"] == 53
        assert manifest["patients_completed"] == 46 and manifest["new_patient_models"] == 3
        rows = csv_read(track(directory / "record_metrics.csv"))
        assert len(rows) == 53 and {int(r["record_fold"]) for r in rows} == set(range(53))
        row = next(r for r in rows if int(r["record_fold"]) == fold)
        assert row["patient_id"] == patient
        group_ref = next(r for r in manifest["group_manifests"] if r["patient_id"] == patient)
        group_path = track(group_ref["group_manifest"])
        assert group_path == (directory / "groups" / patient / "group_manifest.json").resolve()
        group = json.loads(group_path.read_text())
        assert group["status"] == "complete" and group["identity_intersection"] == []
        assert group["test_patient_ids"] == [patient] and patient not in group["train_patient_ids"]
        assert fold in group["test_record_folds"]
        waveform = track(row["waveforms"])
        assert waveform == (directory / "records" / f"record_{fold:02d}" / "waveforms.npz").resolve()
        metric_path = track(row["metrics_file"])
        assert str(metric_path) in group["record_metrics"]
        record_metric = json.loads(metric_path.read_text())
        assert record_metric["record_fold"] == fold and record_metric["patient_id"] == patient
        assert (row["reused"].lower() == "true") == group["reused"]
        with np.load(waveform) as saved:
            arrays[experiment] = {key: saved[key].copy() for key in ("prediction", "target")}
        assert arrays[experiment]["prediction"].shape == arrays[experiment]["target"].shape == (14388,)
        assert all(np.isfinite(v).all() for v in arrays[experiment].values())
        selected[experiment] = {"label": LABELS[experiment], "waveforms": str(waveform), "waveforms_sha256": inputs[str(waveform)],
                                "record_metrics": str(metric_path), "group_manifest": str(group_path),
                                "provenance": "reused_singleton" if group["reused"] else "new_patient_group"}
        group_records[experiment] = group["test_record_folds"]

    target = arrays["pretrained_corr"]["target"]
    assert all(np.array_equal(target, arrays[e]["target"]) for e in EXPERIMENTS)
    subjects = load_bidmc_subjects(track(ROOT / "data_bidmc/bidmc_data.mat"))
    assert np.array_equal(target, subjects[fold]["respiration"][:len(target)])
    ppg = subjects[fold]["ppg"][:len(target)]
    hz, width = 30, 918
    # All sample-aligned windows, not selection on model prediction quality.
    target64 = target.astype(np.float64)
    sums = np.concatenate(([0.], np.cumsum(target64)))
    squares = np.concatenate(([0.], np.cumsum(target64 ** 2)))
    means = (sums[width:] - sums[:-width]) / width
    variances = (squares[width:] - squares[:-width]) / width - means ** 2
    start = int(np.argmax(variances))
    end = start + width
    assert abs(float(np.var(target64[start:end])) - variances[start]) < 1e-12
    direct_variances = np.var(np.lib.stride_tricks.sliding_window_view(target64, width), axis=1)
    assert int(np.argmax(direct_variances)) == start
    assert np.max(np.abs(direct_variances - variances)) < 1e-11
    reference = target[start:end]
    predictions = {e: arrays[e]["prediction"][start:end] for e in EXPERIMENTS}
    correlations = {e: float(np.corrcoef(predictions[e], reference)[0, 1]) for e in EXPERIMENTS}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
                         "axes.titlesize": 19, "axes.labelsize": 18, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.2, 13.3), facecolor="white", layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0, 1, .825), h_pad=.14, w_pad=.12, hspace=.035)
    grid = fig.add_gridspec(5, 1, height_ratios=[.87, 1, 1, 1, 1.24])
    axes = [fig.add_subplot(grid[i, 0]) for i in range(5)]
    fig.suptitle("Selected patient-level LOSO case", fontsize=25, fontweight="bold", x=.55, y=.988).set_in_layout(False)
    fig.text(.55, .943, f"Patient {patient}  |  recording {fold + 1:02d}  |  {start / hz:.2f}–{end / hz:.2f} s", ha="center", fontsize=18)
    fig.text(.55, .914, "Illustrative selected window — not a population summary", ha="center", fontsize=17, color="#555555")
    handles = [Line2D([0], [0], color="#232323", lw=2.6, ls="--", label="Reference respiration")]
    handles += [Line2D([0], [0], color=COLORS[e], lw=2.6, label=LABELS[e]) for e in EXPERIMENTS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .891), ncol=2,
               frameon=False, fontsize=17, handlelength=2.7, columnspacing=1.8, labelspacing=.55).set_in_layout(False)
    time = np.arange(width) / hz
    axes[0].plot(time, ppg[start:end], color="#676B73", lw=1.5)
    axes[0].set_title("A  PPG input", loc="left", pad=9, fontweight="bold")
    axes[0].set_ylabel("PPG\n(z-score)")
    min_resp = min(float(reference.min()), *(float(v.min()) for v in predictions.values()))
    max_resp = max(float(reference.max()), *(float(v.max()) for v in predictions.values()))
    margin = .12 * max(max_resp - min_resp, .1)
    for index, experiment in enumerate(EXPERIMENTS, start=1):
        axis = axes[index]
        axis.plot(time, predictions[experiment], color=COLORS[experiment], lw=2.6, zorder=2)
        axis.plot(time, reference, color="#232323", ls="--", lw=2.35, zorder=3)
        axis.set_ylim(min_resp - margin, max_resp + margin)
        axis.set_title(f"{'BCD'[index - 1]}  {LABELS[experiment]}   |   segment r = {correlations[experiment]:.2f}",
                       loc="left", pad=9, fontweight="bold")
        axis.set_ylabel("Respiration\n(normalized)")
    for index, axis in enumerate(axes[:4]):
        axis.set_xlim(0, width / hz)
        axis.set_xticks([0, 5, 10, 15, 20, 25, 30])
        axis.grid(axis="both", color="#D9DEE4", lw=.7, alpha=.8)
        if index != 3:
            axis.tick_params(axis="x", labelbottom=False)
    axes[3].set_xlabel("Time within selected window (s)", labelpad=7)

    frequency = np.fft.rfftfreq(width, d=1 / hz)
    band = (frequency >= .08) & (frequency <= .8)
    spectra = {}
    series = {"reference": reference, **predictions}
    for key, values in series.items():
        # Float32 centering matches the saved-waveform RR estimator exactly.
        power = np.abs(np.fft.rfft(values - values.mean())) ** 2
        power = power[band] / max(float(power[band].max()), 1e-12)
        spectra[key] = power.tolist()
        axes[4].plot(60 * frequency[band], power, color="#232323" if key == "reference" else COLORS[key],
                     lw=2.6, ls="--" if key == "reference" else "-", marker="o", markersize=3.3,
                     zorder=4 if key == "reference" else 2)
    axes[4].set_title("E  Respiratory-band spectrum (peak-normalized)",
                      loc="left", pad=10, fontweight="bold")
    axes[4].set_xlabel("Frequency (breaths/min)", labelpad=7)
    axes[4].set_ylabel("Relative power")
    axes[4].set_xlim(4.8, 48)
    axes[4].set_ylim(-.03, 1.08)
    axes[4].set_xticks([5, 10, 20, 30, 40, 48])
    axes[4].set_yticks([0, .5, 1])
    axes[4].grid(color="#D9DEE4", lw=.7, alpha=.8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    previous_png_sha = sha(FIGURE) if FIGURE.exists() else None
    fig.savefig(FIGURE, dpi=210, facecolor="white")
    plt.close(fig)

    caption = (
        "Illustrative patient-level LOSO case, not a population summary. The pretrained-plus-correlation patient "
        "nearest the median correlation across 46 patients was selected (ties by patient ID), followed by the record "
        "nearest that patient's mean correlation and its maximum-reference-variance 30.6-s window at one-sample stride. "
        f"This gives patient {patient}, recording {fold + 1}, {start / hz:.2f}--{end / hz:.2f} s. "
        "A: unchanged record-standardized PPG input. B--D: three manifest-admitted, patient-disjoint predictions "
        "and the same record-min--max-normalized respiration reference; displayed correlations describe this window only. "
        "E: mean-removed, unpadded 918-point FFT power in 0.08--0.8 Hz, each curve divided by its own band maximum. "
        "The spectrum depicts shape, not absolute power or manual-reference respiratory-rate accuracy; "
        "this selected case does not establish overall method superiority."
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": str(FIGURE), "figure_sha256": sha(FIGURE), "replaced_working_copy_png_sha256": previous_png_sha,
        "selection_rule": {
            "configuration": "pretrained_corr", "patient": "minimum absolute distance to median of 46 patient mean record correlations",
            "patient_tie_break": "lexicographic patient_id among distances within 1e-12 of the minimum", "record": "minimum absolute distance to selected patient mean record correlation",
            "record_tie_break": "lowest original zero-based fold", "window": "maximum reference variance among all 918-sample windows",
            "window_stride_samples": 1, "window_tie_break": "earliest start sample", "variance_ddof": 0,
            "uses_prediction_fit_to_select_window": False,
        },
        "selection_patient_candidates": patient_distances, "patient_median_correlation": median,
        "tied_nearest_patients": tied_patients,
        "selected_patient": chosen_patient, "selected_patient_record_candidates": candidates,
        "selected_record_fold": fold, "selected_record_number": fold + 1,
        "selected_record_correlation": float(chosen_record["waveform_correlation"]),
        "segment_start_sample": start, "segment_end_sample_exclusive": end,
        "segment_length_samples": width, "sampling_hz": hz,
        "segment_start_seconds": start / hz, "segment_end_seconds_exclusive": end / hz,
        "candidate_windows": int(len(variances)), "target_window_variance": float(variances[start]),
        "window_variance_independent_direct_check": "same argmax; maximum variance-array difference below 1e-11",
        "candidate_variances_float64_sha256": hashlib.sha256(variances.tobytes()).hexdigest(),
        "segment_correlations": correlations,
        "window_fft_peak_rates_bpm_not_manual_reference": {key: respiratory_rate(value) for key, value in series.items()},
        "spectrum": {"mean_removed": True, "window": "rectangular", "n_fft": width,
                     "zero_padding": False, "peak_interpolation": False, "band_hz": [.08, .8],
                     "normalization": "each curve divided by its own band maximum", "bpm": (frequency[band] * 60).tolist(),
                     "relative_power": spectra},
        "preprocessing": "unchanged 30-Hz resample_poly; whole-record PPG z-score and respiration min-max; no plot-only detrending or waveform rescaling",
        "preprocessing_ast_equivalence": source_checks, "selected_inputs": selected,
        "input_sha256": inputs, "caption": caption,
        "interpretation": "Selected case only. Target-variance window selection emphasizes reference variation; no overall superiority claim.",
        "pdf_written": False, "training_performed": False,
    }
    JSON_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"patient": patient, "fold": fold, "start_sample": start,
                      "window_seconds": [start / hz, end / hz], "segment_correlations": correlations,
                      "figure": str(FIGURE), "json": str(JSON_FILE), "caption": caption}, indent=2))


if __name__ == "__main__":
    main()
