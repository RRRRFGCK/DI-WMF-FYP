"""Independent arithmetic/schema checks of completed patient46 RR artifacts."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel

HERE=Path(__file__).resolve().parent
OUT=HERE/"physiology_evaluation"


def read_csv(name):
    with (OUT/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def statistics(a):
    a=np.asarray(a,dtype=float)
    mean=float(np.mean(a)); sd=float(np.std(a,ddof=1))
    half=float(t.ppf(.975,len(a)-1)*sd/math.sqrt(len(a)))
    return {"mean":mean,"sd":sd,"ci95_low":mean-half,"ci95_high":mean+half}


def equal(a,b,tolerance=1e-10):
    if not math.isclose(float(a),float(b),rel_tol=tolerance,abs_tol=tolerance):
        raise AssertionError((a,b))


def main():
    manifest=json.loads((OUT/"physiology_evaluation_manifest_patient46.json").read_text())
    assert manifest["status"]=="complete"
    assert manifest["patient_groups"]==46 and manifest["records_per_method"]==53
    assert manifest["reused_record_predictions"]==559 and manifest["new_group_record_predictions"]==130
    assert manifest["inference_family_size"]==120
    assert manifest["primary_common_valid_windows"]==23838
    for name,digest in manifest["generated_artifact_hashes"].items():
        observed=hashlib.sha256((OUT/name).read_bytes()).hexdigest()
        assert observed==digest,(name,observed,digest)
    patients=read_csv("physiology_patient_metrics_patient46.csv")
    vectors=defaultdict(dict)
    for row in patients:
        vectors[row["method"],row["metric"]][row["patient_id"]]=float(row["value"])
    summary=read_csv("physiology_summary_patient46.csv")
    for row in summary:
        values=list(vectors[row["method"],row["metric"]].values())
        values=[v for v in values if np.isfinite(v)]
        assert len(values)==int(row["n_patients"])
        for name,value in statistics(values).items():
            equal(value,row[name])
    classical=read_csv("classical_paired_patient_values_patient46.csv")
    cvectors=defaultdict(dict)
    for row in classical:
        if row["included"].lower()=="true":
            cvectors[row["method"],row["metric"]][row["patient_id"]]=(
                float(row["method_value"]),float(row["reference_value"]))
    pairs=read_csv("physiology_paired_holm_patient46.csv")
    assert len(pairs)==120
    protocol=json.loads((HERE/"rr_evaluation_protocol.json").read_text())
    expected={(a,b,m) for a,b in protocol["neural_contrasts"] for m in
              protocol["waveform_metrics"]+protocol["inferential_rr_metrics"]}
    expected.update((method,"pretrained_mse",metric) for method in
                    ("bandpass_fft","envelope_fft","bandpass_autocorrelation_corrected")
                    for metric in protocol["inferential_rr_metrics"])
    assert {(r["method"],r["reference"],r["metric"]) for r in pairs}==expected
    for row in pairs:
        key=(row["method"],row["metric"])
        if key in cvectors:
            ordered=sorted(cvectors[key])
            a,b=np.asarray([cvectors[key][p] for p in ordered]).T
        else:
            first,second=vectors[key],vectors[row["reference"],row["metric"]]
            assert set(first)==set(second)
            ordered=sorted(first)
            a=np.array([first[p] for p in ordered]);b=np.array([second[p] for p in ordered])
        assert len(a)==int(row["n_patients"])
        equal(a.mean(),row["method_mean"]);equal(b.mean(),row["reference_mean"])
        difference=a-b
        for name,value in statistics(difference).items():
            equal(value,row[name])
        equal(ttest_rel(a,b).pvalue,row["p_value_raw"])
        direction=1 if row["higher_is_better"].lower()=="true" else -1
        assert int((direction*difference>0).sum())==int(row["method_wins"])
        assert int((direction*difference<0).sum())==int(row["reference_wins"])
    raw=np.array([float(r["p_value_raw"]) for r in pairs])
    reported=np.array([float(r["p_value_holm"]) for r in pairs])
    order=np.argsort(raw)
    corrected=np.minimum(1,np.maximum.accumulate(raw[order]*np.arange(len(raw),0,-1)))
    assert np.max(abs(reported[order]-corrected))<1e-12
    assert np.all(reported>=raw-1e-15)
    windows=read_csv("rr_windows_patient46.csv")
    assert len(windows)==23850
    valid=sum(r["primary_common_valid"].lower()=="true" for r in windows)
    assert valid==23838
    for row in windows:
        expected_valid=int(row["ann1_event_count"])>=2 and int(row["ann2_event_count"])>=2
        assert (row["primary_common_valid"].lower()=="true")==expected_valid
        if expected_valid:
            equal(row["manual_reference_bpm"],(float(row["ann1_rate_bpm"])+float(row["ann2_rate_bpm"]))/2)
        ac=float(row["bandpass_autocorrelation_corrected_predicted_bpm"])
        if np.isfinite(ac):
            assert 4.8<=ac<=48
    result={"status":"passed","summary_rows_recomputed":len(summary),
            "patient_metric_rows":len(patients),"paired_tests_recomputed":len(pairs),
            "holm_max_absolute_error":float(np.max(abs(reported[order]-corrected))),
            "window_rows_validated":len(windows),"reference_only_valid_windows":valid,
            "reused_record_predictions":559,"new_group_record_predictions":130,
            "artifact_hashes_verified":len(manifest["generated_artifact_hashes"]),
            "source_data_or_original_outputs_modified":False}
    (OUT/"rr_independent_qa_patient46.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
