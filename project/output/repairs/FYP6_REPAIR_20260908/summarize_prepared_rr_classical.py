"""Training-free classical summaries from the frozen RR preparation only."""
import json
from pathlib import Path
import numpy as np
from evaluate_rr_patient46 import CLASSICAL, RR_METRICS, mean_ci, write_csv, write_json, sha256

HERE = Path(__file__).resolve().parent
OUT = HERE / "physiology_evaluation"
data = np.load(OUT / "rr_classical_prepared.npz",allow_pickle=False)
ids = data["patient_ids"]
manual = (data["ann1"] + data["ann2"]) / 2
common = np.isfinite(manual)
rows = []
for method in CLASSICAL:
    mask = common & np.isfinite(data[method])
    for metric,reference in ((RR_METRICS[0],manual),(RR_METRICS[1],data["fft_reference"])):
        record_values = np.array([np.mean(abs(data[method][i,mask[i]]-reference[i,mask[i]])) for i in range(53)])
        patient_values = [np.mean(record_values[ids==patient]) for patient in sorted(set(ids))]
        rows.append({"method":method,"metric":metric,"n_patients":46,"n_records":53,
                     "n_windows":int(mask.sum()),"reference_valid_windows":int(common.sum()),
                     "coverage":float(mask.sum()/common.sum()),**mean_ci(patient_values)})
write_csv(OUT / "classical_prepared_summary_patient46.csv",rows)
write_json(OUT / "classical_prepared_summary_manifest.json",{
    "status":"training_free_classical_summary_only_not_full_neural_comparison",
    "input_sha256":sha256(OUT / "rr_classical_prepared.npz"),
    "protocol_sha256":sha256(HERE / "rr_evaluation_protocol.json"),"rows":rows})
print(json.dumps(rows,indent=2))
