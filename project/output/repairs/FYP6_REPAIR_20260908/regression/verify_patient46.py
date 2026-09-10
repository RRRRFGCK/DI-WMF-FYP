"""Verify the patient46 repair manifest, lineage, identity, metrics and outputs.

Read-only with respect to experiment outputs; writes this repair QA directory.
No fitting and no GPU inference. Partial mode allows QA while jobs finish.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from run_correncoder_regression import load_bidmc_subjects, respiratory_rate_errors, lag_robust_waveform_correlation


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--partial",action="store_true");a=p.parse_args()
    out=ROOT/"outputs_correncoder_patient46_20260908"
    subjects=load_bidmc_subjects(ROOT/"data_bidmc/bidmc_data.mat")
    data=loadmat(ROOT/"data_bidmc/bidmc_data.mat",simplify_cells=True)["data"]
    ids=[r["fix"]["id"] for r in data]
    rows=[];issues=[];new=0;reused=0;records=0
    registry=ROOT/"output/pdf/FYP3_RECORDS_REVISION_20260906/audit_work/regression/regression_experiment_registry.csv"
    with registry.open(newline="",encoding="utf-8") as f:
        experiments=list(csv.DictReader(f))
    for entry in experiments:
        exp=entry["experiment"];ep=out/exp
        for pid in dict.fromkeys(ids):
            gp=ep/"groups"/pid;mp=gp/"group_manifest.json"
            if not mp.exists():
                if not a.partial:issues.append(f"Missing {exp}/{pid}")
                continue
            m=json.loads(mp.read_text());status=m.get("status")
            if status!="complete":
                if not a.partial:issues.append(f"Incomplete {exp}/{pid}")
                continue
            expected_test=[i for i,v in enumerate(ids) if v==pid]
            expected_train=[i for i,v in enumerate(ids) if v!=pid]
            assert m["test_record_folds"]==expected_test,(exp,pid,"test")
            assert m["train_record_folds"]==expected_train,(exp,pid,"train order")
            assert not({ids[i] for i in expected_train}&{pid}),(exp,pid,"overlap")
            assert m["train_patient_ids"]==sorted(set(ids[i] for i in expected_train))
            assert m["test_patient_ids"]==[pid] and m["identity_intersection"]==[]
            assert m["group_seed"]==55+min(expected_test)
            assert m["train_segments"]==50*len(expected_train)
            with (gp/"history.csv").open(newline="",encoding="utf-8") as f:
                history=list(csv.DictReader(f))
            assert [int(h["epoch"]) for h in history]==list(range(1,81))
            is_reused=m["reused"]
            if is_reused:
                reused+=1;assert len(expected_test)==1
                for copy in m["copied_files"]:
                    assert digest(copy["source"])==copy["sha256"]==digest(copy["destination"])
                source_metrics_path=Path(m["source_fold_directory"])/"final_metrics.json"
                assert digest(source_metrics_path)==m["source_metrics_sha256"]
                source_metrics=json.loads(source_metrics_path.read_text())
            else:
                new+=1;assert len(expected_test)>1 and m["epochs_completed"]==80
                assert digest(m["checkpoint"])==m["checkpoint_sha256"]
                cfg=m["configuration"]
                assert cfg["epochs"]==80 and cfg["batch_size"]==30 and cfg["learning_rate"]==.001
                expected_frozen=[int(cfg["safe_calibration_head"] and int(h["epoch"])<=cfg["calibration_warmup_epochs"]) for h in history]
                assert [int(h["base_frozen"]) for h in history]==expected_frozen
                assert all(np.isfinite(float(h["train_objective"])) for h in history)
                for source,sha in m["source_codes_sha256"].items():
                    if digest(ROOT/source)!=sha:
                        issues.append(f"Source changed after {exp}/{pid}: {source}")
            for fold in expected_test:
                rp=ep/"records"/f"record_{fold:02d}";r=json.loads((rp/"final_metrics.json").read_text())
                assert r["patient_id"]==pid and r["record_fold"]==fold and r["reused"]==is_reused
                assert Path(r["checkpoint"]).resolve()==(gp/"checkpoint_final.pt").resolve()
                if is_reused:
                    assert Path(r["source_metrics"]).resolve()==source_metrics_path.resolve()
                    assert r["source_metrics_sha256"]==m["source_metrics_sha256"]
                    for metric in ("test_mse","test_mae","waveform_correlation",
                                   "rr_mean_absolute_error_bpm_30p6s","rr_median_absolute_error_bpm_30p6s"):
                        assert r[metric]==source_metrics[metric],(exp,pid,metric,"reused metric changed")
                arrays=np.load(rp/"waveforms.npz")
                pred,target=arrays["prediction"],arrays["target"]
                assert pred.shape==target.shape==(14388,) and np.isfinite(pred).all() and np.isfinite(target).all()
                assert np.array_equal(target,subjects[fold]["respiration"][:14388]),(exp,fold,"target differs")
                errors=respiratory_rate_errors(pred,target)
                assert len(errors)==450 and np.array_equal(errors,arrays["rr_absolute_errors"])
                assert abs(float(errors.mean())-r["rr_mean_absolute_error_bpm_30p6s"])<1e-10
                corr=lag_robust_waveform_correlation(pred,target,max_lag_samples=0)[0]
                maxcorr=lag_robust_waveform_correlation(pred,target,max_lag_samples=30)[0]
                assert abs(corr-r["waveform_correlation"])<1e-8
                assert abs(maxcorr-r["max_lag_waveform_correlation"])<1e-8
                records+=1
            rows.append({"experiment":exp,"patient_id":pid,"reused":is_reused,
                         "records":len(expected_test),"train_patients":len(m["train_patient_ids"]),
                         "train_records":len(expected_train),"seed":m["group_seed"],
                         "epochs":80,"identity_check":"pass","lineage_and_metric_check":"pass"})
    if not a.partial:
        assert len(rows)==598 and new==39 and reused==559 and records==689
    dest=Path(__file__).parent
    with (dest/"patient46_integrity_rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"complete":not a.partial,"groups_checked":len(rows),"new_fits_checked":new,
             "reused_fits_checked":reused,"record_outputs_checked":records,
             "issues":issues,"training_or_inference_performed":False}
    (dest/"patient46_integrity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    if issues:raise SystemExit(1)


if __name__=="__main__":
    main()
