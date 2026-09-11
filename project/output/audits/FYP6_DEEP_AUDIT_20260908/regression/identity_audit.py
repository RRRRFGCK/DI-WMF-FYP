"""Audit BIDMC patient identity grouping without training or changing outputs."""
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np
from scipy.io import loadmat
from scipy.stats import t, ttest_rel

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def ci(x):
    a=np.asarray(x,dtype=float)
    h=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return {"mean":float(a.mean()),"ci95_low":float(a.mean()-h),"ci95_high":float(a.mean()+h)}


def main():
    data=loadmat(ROOT/"data_bidmc/bidmc_data.mat",simplify_cells=True)["data"]
    ids=[s["fix"]["id"] for s in data]
    counts=Counter(ids)
    rows=[]
    for i,s in enumerate(data):
        overlap=[j for j,sid in enumerate(ids) if sid==ids[i] and j!=i]
        rows.append({"fold_zero_based":i,"reported_test_subject":i+1,
                     "actual_patient_id":ids[i],"same_patient_training_folds":";".join(map(str,overlap)),
                     "same_patient_training_records":len(overlap),
                     "same_patient_training_segments":len(overlap)*50,
                     "source_recording":s["fix"]["data_segment"],
                     "source_waveform":s["fix"]["source"]["waves_data_file"]})
    with (OUT/"bidmc_identity_registry.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    eligible=[i for i,sid in enumerate(ids) if counts[sid]==1]
    exp={}
    registry=ROOT/"output/pdf/FYP3_RECORDS_REVISION_20260906/audit_work/regression/regression_experiment_registry.csv"
    with registry.open(newline="",encoding="utf-8") as f:
        experiments=list(csv.DictReader(f))
    summaries=[]
    metrics=["test_mse","test_mae","waveform_correlation","rr_mean_absolute_error_bpm_30p6s"]
    for r in experiments:
        with (ROOT/r["path"]/"fold_metrics.csv").open(newline="",encoding="utf-8") as f:
            records={int(v["fold"]):v for v in csv.DictReader(f)}
        exp[r["experiment"]]=records
        for metric in metrics:
            for label,folds in [("all_53_records",list(range(53))),
                                ("43_no_same_patient_training_overlap",eligible)]:
                summaries.append({"experiment":r["experiment"],"subset":label,"metric":metric,
                                  "n":len(folds),**ci([records[i][metric] for i in folds])})
    with (OUT/"identity_sensitivity_aggregate.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    pairs=[]
    contrasts=[("pretrained_corr","pretrained_mse"),
               ("matched_layerwise_calibrated_mse","matched_1d_mse"),
               ("pretrained_hard_lag","pretrained_corr"),
               ("pretrained_soft_lag","pretrained_mse")]
    for a,b in contrasts:
        for metric in metrics:
            aa=np.asarray([float(exp[a][i][metric]) for i in eligible])
            bb=np.asarray([float(exp[b][i][metric]) for i in eligible])
            pairs.append({"method":a,"reference":b,"metric":metric,"n":len(eligible),
                          "effect_definition":"method_minus_reference",**ci(aa-bb),
                          "p_raw_exploratory":float(ttest_rel(aa,bb).pvalue)})
    with (OUT/"identity_sensitivity_paired.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    # Patient weighting for training-free classical estimates can be repaired
    # without new fits. Grouping neural predictions does NOT cure training leak.
    with (ROOT/"classical_rr_results/classical_rr_subject_rows.csv").open(newline="",encoding="utf-8") as f:
        classical=list(csv.DictReader(f))
    classical_groups=[]
    for method in sorted({r["method"] for r in classical}):
        subset={int(r["fold"]):float(r["rr_mean_absolute_error_bpm_30p6s"]) for r in classical if r["method"]==method}
        patient_values=[np.mean([subset[i] for i,v in enumerate(ids) if v==sid]) for sid in counts]
        classical_groups.append({"method":method,"records":53,"patient_groups":46,
                                 "old_record_macro_mean":float(np.mean(list(subset.values()))),
                                 "patient_macro":ci(patient_values)})
    summary={"records":len(data),"unique_patient_ids":len(counts),
             "id_types":sorted({type(sid).__name__ for sid in ids}),
             "all_ids_equal_to_stripped_ids":all(sid==sid.strip() for sid in ids),
             "all_source_waveforms_in_matching_patient_directory":all('/'+ids[i]+'/' in s['fix']['source']['waves_data_file'] for i,s in enumerate(data)),
             "source_segments_unique":len({s['fix']['data_segment'] for s in data}),
             "source_waveform_files_unique":len({s['fix']['source']['waves_data_file'] for s in data}),
             "duplicate_groups":{sid:[i for i,v in enumerate(ids) if v==sid] for sid,n in counts.items() if n>1},
             "leaky_test_folds":sum(counts[sid]>1 for sid in ids),
             "eligible_unique_test_folds":eligible,
             "affected_training_windows_per_fold":sorted({r["same_patient_training_segments"] for r in rows if r["same_patient_training_records"]}),
             "neural_repair_minimum":"Keep 43 unaffected fits per configuration only under the original record-balanced window training protocol (50 windows/record, existing subject order/seed/batching, not 46-patient-balanced sampling). For these singleton patients, excluding the held-out patient yields exactly the old 52-record training tensors and existing seeded trajectory. Replace 10 contaminated record fits by 3 patient-group fits and evaluate every held-out recording; aggregate at 46 patient groups. This requires 3 new fits/configuration (39 for 13 configurations). Changing patient-level sampling weights, seeds, or order invalidates this minimal reuse argument.",
             "classical_patient_macro":classical_groups,
             "limitation":"The 43-fold deletion sensitivity is not a corrected full 46-patient LOSO result. The existing 10 contaminated predictions cannot be repaired by merely averaging them per patient.",
             "pretrained_mse_rr_sensitivity":[r for r in summaries if r["experiment"]=="pretrained_mse" and r["metric"]=="rr_mean_absolute_error_bpm_30p6s"]}
    (OUT/"identity_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
