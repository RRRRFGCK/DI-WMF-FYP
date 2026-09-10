"""Versioned BIDMC patient-group LOSO repair, preserving historical outputs.

Protocol v1: record-balanced training, 50 windows/record, final Epoch 80;
patient groups use raw MAT fix.id. Singleton fits are reused only when their
ordered training records, preprocessing, seed and optimisation policy agree.
Multiple-record patients receive one new fit using seed 55+min(original fold).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.models import PublishedCorrEncoder1D
from domain_mf.regression import (
    calibrate_regression_output, correncoder_regression_loss,
    fit_safe_calibration_head, initialise_1d_matched_filter,
    initialise_1d_layerwise_matched_filter,
)
from run_correncoder_regression import (
    ROOT, _evaluate, _non_overlapping_segments, _sliding_segments,
    lag_robust_waveform_correlation, load_bidmc_subjects, overlap_average,
    respiratory_rate_errors, seed_everything,
)

VERSION = "bidmc_patient_group_loso_v1_20260908"
OUTPUT = ROOT / "outputs_correncoder_patient46_20260908"
WORK = ROOT / "output/repairs/FYP6_REPAIR_20260908/regression"
REGISTRY = ROOT / "output/pdf/FYP3_RECORDS_REVISION_20260906/audit_work/regression/regression_experiment_registry.csv"
METRICS = ("test_mse", "test_mae", "waveform_correlation",
           "max_lag_waveform_correlation", "rr_mean_absolute_error_bpm_30p6s",
           "rr_median_absolute_error_bpm_30p6s")
DEFAULTS = dict(epochs=80, batch_size=30, learning_rate=.001, seed=55,
    num_workers=0, matched_max_patches=50000, matched_deep_max_patches=10000,
    matched_shrinkage=.1, matched_depth=3, matched_deep_covariance="diagonal",
    matched_covariance_rank=16, affine_calibration=False,
    safe_calibration_head=False, calibration_min_gain=.05,
    calibration_max_gain=10., calibration_warmup_epochs=3,
    correlation_lambda=0., correlation_mode="zero_lag", max_lag_samples=30,
    lag_step=3, lag_temperature=.05, spectral_lambda=0.,
    initialisation="pretrained", pretrained_capnobase=None)


def utc():
    return datetime.now(timezone.utc).isoformat()


@lru_cache(None)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def csv_write(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def provenance():
    paths=[Path(__file__),ROOT/"run_correncoder_regression.py",
           ROOT/"domain_mf/regression.py",ROOT/"domain_mf/models.py"]
    return {str(p.relative_to(ROOT)):sha(str(p)) for p in paths}


def read_registry():
    with REGISTRY.open(newline="",encoding="utf-8") as f:
        return {r["experiment"]:r for r in csv.DictReader(f)}


def configuration(entry):
    source=ROOT/entry["path"]/"experiment_config.json"
    raw=json.loads(source.read_text()) if source.exists() else {}
    cfg={**DEFAULTS,**raw}
    if entry["experiment"]=="pretrained_mse":
        cfg["initialisation"]="pretrained"
    cfg={k:cfg[k] for k in DEFAULTS}
    if cfg["initialisation"]=="pretrained":
        cfg["pretrained_capnobase"]=str(ROOT/"outputs_correncoder_pretrain/checkpoint_capnobase_all.pt")
    assert cfg["epochs"]==80 and cfg["batch_size"]==30 and cfg["seed"]==55
    assert cfg["learning_rate"]==.001
    return cfg, {"source_configuration":str(source) if source.exists() else None,
                 "source_configuration_sha256":sha(str(source)) if source.exists() else None,
                 "fields_using_explicit_repair_defaults":[k for k in DEFAULTS if k not in raw],
                 "historical_config_missing":not source.exists(),
                 "interpretation":"Defaults are declared repair execution choices from available source/protocol; not fabricated historical launch metadata."}


def load_data():
    raw=loadmat(ROOT/"data_bidmc/bidmc_data.mat",simplify_cells=True)["data"]
    subjects=load_bidmc_subjects(ROOT/"data_bidmc/bidmc_data.mat")
    ids=[r["fix"]["id"] for r in raw]
    groups=defaultdict(list)
    for i,pid in enumerate(ids):
        assert isinstance(pid,str) and pid==pid.strip()
        assert "/"+pid+"/" in raw[i]["fix"]["source"]["waves_data_file"]
        groups[pid].append(i)
        subjects[i]["patient_id"]=pid
        subjects[i]["record_fold"]=i
    assert len(ids)==53 and len(groups)==46
    assert sorted(len(v) for v in groups.values() if len(v)>1)==[2,4,4]
    return subjects,ids,dict(groups)


def group_identity(pid, ids, groups):
    test=groups[pid]
    train=[i for i in range(len(ids)) if ids[i]!=pid]
    train_ids=sorted({ids[i] for i in train})
    assert not(set(train_ids)&{pid})
    assert set(train)|set(test)==set(range(53)) and not(set(train)&set(test))
    return {"patient_id":pid,"test_record_folds":test,
            "train_record_folds":train,"train_patient_ids":train_ids,
            "test_patient_ids":[pid],"identity_intersection":[],
            "identity_assertion_passed":True,"train_records":len(train),
            "train_patients":len(train_ids),"train_segments":50*len(train),
            "group_seed":55+min(test),"seed_source_original_fold":min(test)}


def source_fold_dir(entry,fold):
    p=ROOT/entry["path"]/f"bidmc_fold{fold:02d}_seed{55+fold}"
    assert p.exists(),p
    return p


def prepare_experiment(entry,ids,groups):
    experiment=entry["experiment"]; dest=OUTPUT/experiment
    cfg,cfg_prov=configuration(entry)
    config_payload={"protocol_version":VERSION,"experiment":experiment,
        "training_policy":"record-balanced; 50 non-overlapping 288-sample windows per training record, unchanged original record order",
        "test_policy":"patient-disjoint; per-record overlapping inference, record means averaged within patient, then 46 equal patient weights",
        "sampling_hz":30,"preprocessing":"unchanged whole-record PPG z-score and reference respiration min-max before windowing",
        "checkpoint_rule":"fixed_final_epoch_80_no_validation",
        "singleton_reuse_conditions":"Same original ordered training records, per-record preprocessing, seed, batch and optimisation settings; not patient-balanced training",
        "configuration":cfg,"configuration_provenance":cfg_prov,
        "source_codes_sha256":provenance(),
        "data_mat_sha256":sha(str(ROOT/"data_bidmc/bidmc_data.mat")),
        "external_checkpoint_sha256":sha(cfg["pretrained_capnobase"]) if cfg["pretrained_capnobase"] else None}
    config_file=dest/"experiment_config.json"
    if config_file.exists():
        old=json.loads(config_file.read_text())
        assert old["configuration"]==cfg,"Refusing configuration overwrite"
    else:
        json_write(config_file,config_payload)
    for pid,folds in groups.items():
        if len(folds)!=1:
            continue
        fold=folds[0]; source=source_fold_dir(entry,fold)
        gp=dest/"groups"/pid; rp=dest/"records"/f"record_{fold:02d}"
        if (gp/"group_manifest.json").exists():
            manifest=json.loads((gp/"group_manifest.json").read_text())
            assert manifest["status"]=="complete" and manifest["reused"]
            continue
        gp.mkdir(parents=True,exist_ok=True);rp.mkdir(parents=True,exist_ok=True)
        identity=group_identity(pid,ids,groups)
        # Explicit ordered-index equality is what allows original tensor/seed reuse.
        assert identity["train_record_folds"]==[i for i in range(53) if i!=fold]
        copied=[]
        for name,target in [("checkpoint_final.pt",gp/"checkpoint_final.pt"),
                            ("history.csv",gp/"history.csv"),
                            ("waveforms.npz",rp/"waveforms.npz")]:
            src=source/name
            shutil.copy2(src,target)
            source_sha=sha(str(src))
            assert source_sha==sha(str(target))
            copied.append({"source":str(src),"destination":str(target),"sha256":source_sha})
        original_metrics=json.loads((source/"final_metrics.json").read_text())
        metrics={**original_metrics,"fold":fold,"record_fold":fold,
                 "record_number":fold+1,"patient_id":pid,"experiment":experiment,
                 "protocol":VERSION,"reused":True,"source_metrics":str(source/"final_metrics.json"),
                 "source_metrics_sha256":sha(str(source/"final_metrics.json")),
                 "checkpoint":str(gp/"checkpoint_final.pt"),
                 "waveforms":str(rp/"waveforms.npz")}
        if "max_lag_waveform_correlation" not in metrics:
            arrays=np.load(rp/"waveforms.npz")
            corr,lag=lag_robust_waveform_correlation(arrays["prediction"],arrays["target"])
            metrics.update(max_lag_waveform_correlation=corr,best_waveform_lag_samples=lag)
        json_write(rp/"final_metrics.json",metrics)
        json_write(gp/"group_manifest.json",{"protocol_version":VERSION,
            "experiment":experiment,"status":"complete","reused":True,
            **identity,"source_fold_directory":str(source),"copied_files":copied,
            "source_metrics_sha256":metrics["source_metrics_sha256"],
            "checkpoint":str(gp/"checkpoint_final.pt"),"record_metrics":[str(rp/"final_metrics.json")],
            "reuse_ordered_training_index_assertion":True,"created_utc":utc()})
    return cfg


def make_model(cfg,train_x,train_y,device,seed):
    model=PublishedCorrEncoder1D(dropout=.5).to(device)
    start=time.perf_counter();report={"method":cfg.initialisation}
    if cfg.initialisation=="pretrained":
        model.load_state_dict(torch.load(cfg.pretrained_capnobase,map_location="cpu",weights_only=True))
        report["checkpoint"]=cfg.pretrained_capnobase
    elif cfg.initialisation=="matched_1d":
        report=initialise_1d_matched_filter(model,train_x,train_y,
            max_patches=cfg.matched_max_patches,shrinkage=cfg.matched_shrinkage,seed=seed).to_dict()
    elif cfg.initialisation=="matched_layerwise":
        report=initialise_1d_layerwise_matched_filter(model,train_x,train_y,
            stem_max_patches=cfg.matched_max_patches,deep_max_patches=cfg.matched_deep_max_patches,
            shrinkage=cfg.matched_shrinkage,depth=cfg.matched_depth,
            deep_covariance=cfg.matched_deep_covariance,covariance_rank=cfg.matched_covariance_rank,seed=seed)
    elif cfg.initialisation!="random":
        raise ValueError(cfg.initialisation)
    if cfg.affine_calibration:
        report["affine_calibration"]=calibrate_regression_output(model,train_x,train_y).to_dict()
    if cfg.safe_calibration_head:
        report["safe_calibration_head"]=fit_safe_calibration_head(model,train_x,train_y,
            minimum_absolute_gain=cfg.calibration_min_gain,
            maximum_absolute_gain=cfg.calibration_max_gain).to_dict()
    return model,report,time.perf_counter()-start


def evaluate_record(model,subject,cfg,device,shared,dest):
    fold=subject["record_fold"]
    x,starts=_sliding_segments(subject["ppg"])
    y,target_starts=_sliding_segments(subject["respiration"])
    assert np.array_equal(starts,target_starts)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[:,None,:]),torch.from_numpy(y[:,None,:])),
        batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True)
    result=_evaluate(model,loader,device)
    prediction=overlap_average(result["prediction"],starts)
    target=overlap_average(result["target"],starts)
    errors=respiratory_rate_errors(prediction,target)
    corr=lag_robust_waveform_correlation(prediction,target,max_lag_samples=0)[0]
    maxcorr,lag=lag_robust_waveform_correlation(prediction,target,max_lag_samples=cfg.max_lag_samples,lag_step=1)
    rp=dest/"records"/f"record_{fold:02d}";rp.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(rp/"waveforms.npz",prediction=prediction,target=target,rr_absolute_errors=errors)
    metrics={**shared,"fold":fold,"record_fold":fold,"record_number":fold+1,
        "test_subject":fold+1,"patient_id":subject["patient_id"],
        "test_mse":result["mse"],"test_mae":result["mae"],"waveform_correlation":corr,
        "max_lag_waveform_correlation":maxcorr,"best_waveform_lag_samples":lag,
        "rr_median_absolute_error_bpm_30p6s":float(np.median(errors)),
        "rr_mean_absolute_error_bpm_30p6s":float(np.mean(errors)),"rr_windows":int(len(errors)),
        "test_segments":len(x),"waveforms":str(rp/"waveforms.npz")}
    assert all(np.isfinite(float(metrics[k])) for k in METRICS)
    json_write(rp/"final_metrics.json",metrics)
    return str(rp/"final_metrics.json")


def train_group(entry, cfg_dict, pid, subjects,ids,groups,device):
    experiment=entry["experiment"];dest=OUTPUT/experiment;gp=dest/"groups"/pid
    gp.mkdir(parents=True,exist_ok=True)
    manifest_file=gp/"group_manifest.json"
    if manifest_file.exists():
        existing=json.loads(manifest_file.read_text())
        if existing.get("status")=="complete":
            assert not existing["reused"] and existing["protocol_version"]==VERSION
            print(f"SKIP complete {experiment}/{pid}",flush=True)
            return
    identity=group_identity(pid,ids,groups)
    assert len(identity["test_record_folds"])>1
    cfg=SimpleNamespace(**cfg_dict);seed=identity["group_seed"]
    seed_everything(seed)
    train_x=torch.from_numpy(np.concatenate([_non_overlapping_segments(subjects[i]["ppg"])
        for i in identity["train_record_folds"]])[:,None,:])
    train_y=torch.from_numpy(np.concatenate([_non_overlapping_segments(subjects[i]["respiration"])
        for i in identity["train_record_folds"]])[:,None,:])
    assert len(train_x)==identity["train_segments"]
    train_loader=DataLoader(TensorDataset(train_x,train_y),batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,pin_memory=True,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    torch.cuda.reset_peak_memory_stats(device)
    started=utc();wall=time.perf_counter()
    run_record={"protocol_version":VERSION,"experiment":experiment,"status":"running",
        "reused":False,**identity,"configuration":cfg_dict,
        "source_codes_sha256":provenance(),"started_utc":started,
        "torch_version":torch.__version__,"torch_num_threads":torch.get_num_threads(),
        "cuda_version":torch.version.cuda,"device_name":torch.cuda.get_device_name(device)}
    json_write(gp/"run_config.json",run_record)
    model,init_report,init_seconds=make_model(cfg,train_x,train_y,device,seed)
    optimiser=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    calibration_parameters=set(model.output_calibration.parameters()) if model.output_calibration is not None else set()
    base_parameters=[p for p in model.parameters() if p not in calibration_parameters]
    history=[];start=time.perf_counter()
    print(f"START {experiment}/{pid} seed={seed} train_records={len(identity['train_record_folds'])} test_records={identity['test_record_folds']}",flush=True)
    for epoch in range(1,cfg.epochs+1):
        frozen=bool(cfg.safe_calibration_head and epoch<=cfg.calibration_warmup_epochs)
        for p in base_parameters:
            p.requires_grad_(not frozen)
        model.train();totals=dict(train_objective=0.,train_mse=0.,train_correlation_loss=0.,train_spectral_loss=0.);n=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            output=model(x)
            loss,components=correncoder_regression_loss(output,y,
                correlation_lambda=cfg.correlation_lambda,spectral_lambda=cfg.spectral_lambda,
                correlation_mode=cfg.correlation_mode,max_lag_samples=cfg.max_lag_samples,
                lag_step=cfg.lag_step,lag_temperature=cfg.lag_temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite objective: {experiment}/{pid}/epoch{epoch}")
            loss.backward();optimiser.step();b=y.shape[0]
            totals["train_objective"]+=float(loss.detach())*b
            totals["train_mse"]+=float(components["mse"].detach())*b
            totals["train_correlation_loss"]+=float(components["correlation_loss"].detach())*b
            totals["train_spectral_loss"]+=float(components["spectral_loss"].detach())*b
            n+=b
        torch.cuda.synchronize(device)
        row={"epoch":epoch,**{k:v/n for k,v in totals.items()},"base_frozen":int(frozen),"elapsed_seconds":time.perf_counter()-start}
        history.append(row)
        if epoch==1 or epoch%10==0:
            csv_write(gp/"history.partial.csv",history)
            print(f"EPOCH {experiment}/{pid} {epoch}/80 objective={row['train_objective']:.6f} mse={row['train_mse']:.6f} seconds={row['elapsed_seconds']:.1f}",flush=True)
    csv_write(gp/"history.csv",history)
    checkpoint=gp/"checkpoint_final.pt";torch.save(model.state_dict(),checkpoint)
    shared={"protocol":VERSION,"experiment":experiment,"reused":False,"seed":seed,
        "initialisation":cfg.initialisation,"initialisation_report":init_report,
        "initialisation_seconds":init_seconds,"affine_calibration":cfg.affine_calibration,
        "safe_calibration_head":cfg.safe_calibration_head,
        "final_calibration_gain":float(model.output_calibration.gain().detach()) if model.output_calibration is not None else None,
        "final_calibration_offset":float(model.output_calibration.offset.detach()) if model.output_calibration is not None else None,
        "correlation_lambda":cfg.correlation_lambda,"correlation_mode":cfg.correlation_mode,
        "spectral_lambda":cfg.spectral_lambda,"max_lag_samples":cfg.max_lag_samples,
        "lag_step":cfg.lag_step,"lag_temperature":cfg.lag_temperature,
        "calibration_warmup_epochs":cfg.calibration_warmup_epochs,
        "train_segments":len(train_x),"training_seconds":history[-1]["elapsed_seconds"],
        "parameter_count":sum(p.numel() for p in model.parameters()),"device":str(device),
        "device_name":torch.cuda.get_device_name(device),"checkpoint":str(checkpoint),
        "checkpoint_sha256":sha(str(checkpoint)),"group_train_test_identity":identity}
    paths=[evaluate_record(model,subjects[i],cfg,device,shared,dest) for i in identity["test_record_folds"]]
    final={**run_record,"status":"complete","completed_utc":utc(),
        "initialisation_seconds":init_seconds,"training_seconds":history[-1]["elapsed_seconds"],
        "wall_seconds":time.perf_counter()-wall,"epochs_completed":len(history),
        "checkpoint":str(checkpoint),"checkpoint_sha256":sha(str(checkpoint)),
        "record_metrics":paths,"gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "initialisation_report":init_report}
    json_write(manifest_file,final)
    print(f"COMPLETE {experiment}/{pid} training_seconds={final['training_seconds']:.1f} wall_seconds={final['wall_seconds']:.1f} peak_allocated_MB={final['gpu_peak_allocated_bytes']/2**20:.1f}",flush=True)
    del model,optimiser,train_loader,train_x,train_y
    torch.cuda.empty_cache()


def aggregate_experiment(entry,groups):
    dest=OUTPUT/entry["experiment"];rows=[];patients=[];group_manifests=[]
    for pid,folds in groups.items():
        gp=dest/"groups"/pid/"group_manifest.json"
        if not gp.exists():
            continue
        manifest=json.loads(gp.read_text())
        if manifest.get("status")!="complete":
            continue
        group_manifests.append({"patient_id":pid,"reused":manifest["reused"],"group_manifest":str(gp)})
        record_rows=[]
        for fold in folds:
            p=dest/"records"/f"record_{fold:02d}"/"final_metrics.json"
            r=json.loads(p.read_text());assert r["patient_id"]==pid
            compact={"experiment":entry["experiment"],"patient_id":pid,"record_fold":fold,
                "reused":r["reused"],"seed":r["seed"],**{k:r[k] for k in METRICS},
                "waveforms":r["waveforms"],"checkpoint":r["checkpoint"],"metrics_file":str(p)}
            rows.append(compact);record_rows.append(compact)
        patients.append({"experiment":entry["experiment"],"patient_id":pid,
            "record_count":len(folds),"record_folds":";".join(map(str,folds)),
            "reused":manifest["reused"],
            **{k:float(np.mean([r[k] for r in record_rows])) for k in METRICS}})
    csv_write(dest/"record_metrics.csv",rows);csv_write(dest/"patient_metrics.csv",patients)
    complete=len(rows)==53 and len(patients)==46
    result={"protocol_version":VERSION,"experiment":entry["experiment"],
        "status":"complete" if complete else "partial","records_completed":len(rows),
        "patients_completed":len(patients),"reused_patient_models":sum(r["reused"] for r in patients),
        "new_patient_models":sum(not r["reused"] for r in patients),
        "group_manifests":group_manifests,"patient_aggregation":"unweighted mean of record metrics within patient, then equal patient weights",
        "metric_definition":"legacy waveform errors and respiratory-band FFT-peak discrepancy; independent reference/estimator sensitivity belongs to separate evaluation outputs",
        "uncertainty":"two-sided Student-t over patient means; shared LOSO training sets and single external pretraining instance remain conditional limitations"}
    if complete:
        result["aggregate"]={}
        for metric in METRICS:
            a=np.array([r[metric] for r in patients]);h=float(student_t.ppf(.975,45)*a.std(ddof=1)/np.sqrt(46))
            result["aggregate"][metric]={"mean":float(a.mean()),"ci95_half_width":h,"ci95_low":float(a.mean()-h),"ci95_high":float(a.mean()+h)}
    json_write(dest/"manifest.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments",nargs="+")
    parser.add_argument("--patients",nargs="+")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--aggregate-only",action="store_true")
    parser.add_argument("--num-threads",type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(args.num_threads)
    registry=read_registry();labels=args.experiments or list(registry)
    assert all(v in registry for v in labels)
    subjects,ids,groups=load_data()
    repeated=[pid for pid,folds in groups.items() if len(folds)>1]
    selected=args.patients or repeated
    assert all(pid in repeated for pid in selected)
    WORK.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    if not args.prepare_only and not args.aggregate_only:
        assert torch.cuda.is_available(),"Original CUDA runtime required"
    for label in labels:
        entry=registry[label]
        cfg=prepare_experiment(entry,ids,groups)
        if not args.prepare_only and not args.aggregate_only:
            for pid in selected:
                train_group(entry,cfg,pid,subjects,ids,groups,torch.device("cuda"))
                aggregate_experiment(entry,groups)
        aggregate_experiment(entry,groups)
    all_results=[]
    for label in registry:
        p=OUTPUT/label/"manifest.json"
        if p.exists():
            r=json.loads(p.read_text())
            all_results.append({k:r[k] for k in ["experiment","status","records_completed","patients_completed","reused_patient_models","new_patient_models"]})
    json_write(OUTPUT/"manifest.json",{"protocol_version":VERSION,"updated_utc":utc(),
        "source_codes_sha256":provenance(),"experiments":all_results,
        "expected_experiments":13,"expected_new_fits":39,"expected_reused_fits":559,
        "patient_groups":groups})
    print(json.dumps(all_results,indent=2),flush=True)


if __name__=="__main__":
    main()
