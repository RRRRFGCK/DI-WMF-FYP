"""Independent CUDA reload/inference QA for one new patient per configuration.

No training. All inputs originate from original raw BIDMC, all checkpoints from
the new patient-group output root. Selected patient is s03386 (record fold 5).
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from domain_mf.models import PublishedCorrEncoder1D
from domain_mf.regression import SafeAffineCalibration
from run_correncoder_regression import load_bidmc_subjects, _sliding_segments, _evaluate, overlap_average, seed_everything


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(16);seed_everything(60)
    device=torch.device("cuda")
    subjects=load_bidmc_subjects(ROOT/"data_bidmc/bidmc_data.mat")
    x,starts=_sliding_segments(subjects[5]["ppg"])
    y,_=_sliding_segments(subjects[5]["respiration"])
    loader=DataLoader(TensorDataset(torch.from_numpy(x[:,None,:]),torch.from_numpy(y[:,None,:])),
        batch_size=30,shuffle=False,num_workers=0,pin_memory=True)
    output=ROOT/"outputs_correncoder_patient46_20260908"
    rows=[]
    for exp_dir in sorted(p for p in output.iterdir() if p.is_dir()):
        manifest=json.loads((exp_dir/"manifest.json").read_text())
        assert manifest["status"]=="complete"
        run=json.loads((exp_dir/"groups/s03386/group_manifest.json").read_text())
        assert not run["reused"] and run["identity_intersection"]==[]
        cfg=run["configuration"]
        state=torch.load(exp_dir/"groups/s03386/checkpoint_final.pt",map_location="cpu",weights_only=True)
        model=PublishedCorrEncoder1D(dropout=.5)
        if cfg["safe_calibration_head"]:
            model.output_calibration=SafeAffineCalibration(
                float(state["output_calibration.gain_parameter"]),float(state["output_calibration.offset"]),
                minimum_absolute_gain=cfg["calibration_min_gain"],maximum_absolute_gain=cfg["calibration_max_gain"])
        model.load_state_dict(state,strict=True);model=model.to(device)
        evaluation=_evaluate(model,loader,device)
        prediction=overlap_average(evaluation["prediction"],starts)
        target=overlap_average(evaluation["target"],starts)
        saved=np.load(exp_dir/"records/record_05/waveforms.npz")
        metrics=json.loads((exp_dir/"records/record_05/final_metrics.json").read_text())
        row={"experiment":exp_dir.name,"patient_id":"s03386","record_fold":5,
             "prediction_max_abs_difference":float(np.max(abs(prediction-saved["prediction"]))),
             "target_max_abs_difference":float(np.max(abs(target-saved["target"]))),
             "mse_absolute_difference":abs(evaluation["mse"]-metrics["test_mse"]),
             "mae_absolute_difference":abs(evaluation["mae"]-metrics["test_mae"])}
        assert row["prediction_max_abs_difference"]<1e-6,row
        assert row["target_max_abs_difference"]==0,row
        assert row["mse_absolute_difference"]<1e-8 and row["mae_absolute_difference"]<1e-8,row
        rows.append(row);print(json.dumps(row),flush=True)
        del model,state;torch.cuda.empty_cache()
    assert len(rows)==13
    dest=Path(__file__).parent
    with (dest/"new_patient_cuda_replay.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (dest/"new_patient_cuda_replay_summary.json").write_text(json.dumps({
        "checkpoints_replayed":len(rows),"all_passed":True,"training_performed":False,
        "max_prediction_difference":max(r["prediction_max_abs_difference"] for r in rows),
        "max_mse_difference":max(r["mse_absolute_difference"] for r in rows),
        "max_mae_difference":max(r["mae_absolute_difference"] for r in rows)},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
