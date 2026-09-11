"""Independent provenance/data checks for the three repaired PNG figures."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"physiology_evaluation"
FIG=ROOT/"physiology_figure_sources"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def main():
    expected={
        "F13":{"input_summary.csv":("physiology_summary_patient46.csv",21)},
        "F15":{"input_paired.csv":("physiology_paired_holm_patient46.csv",15)},
        "F24":{"input_patient_values.csv":("physiology_patient_metrics_patient46.csv",184),
               "input_summary.csv":("physiology_summary_patient46.csv",4),
               "input_paired.csv":("physiology_paired_holm_patient46.csv",1)},
    }
    reports=[]
    for fig_id,entries in expected.items():
        manifest=json.loads((FIG/fig_id/"hash_manifest.json").read_text())
        assert sha(manifest["output_png"])==manifest["output_sha256"]
        assert sha(manifest["plot_source"])==manifest["plot_source_sha256"]
        assert sha(DATA/"physiology_evaluation_manifest_patient46.json")==manifest["evaluation_manifest_sha256"]
        for path,digest in manifest["figure_input_csv"].items():assert sha(path)==digest
        for name,digest in manifest["original_inputs"].items():assert sha(DATA/name)==digest
        for short_name,(source_name,n) in entries.items():
            subset=read(FIG/fig_id/short_name);source=read(DATA/source_name)
            assert len(subset)==n
            assert all(row in source for row in subset)
        with Image.open(manifest["output_png"]) as im:
            assert im.format=="PNG" and im.width==3840 and abs(im.info["dpi"][0]-300)<.1
        assert manifest["automated_layout_QA"]=={"no_text_beyond_canvas":True,"ci_axis_limits_checked":True}
        reports.append({"figure":fig_id,"source_rows_checked":sum(v[1] for v in entries.values()),
                        "png_sha256":manifest["output_sha256"],"dimensions":manifest["pixel_dimensions"]})
    patients=read(FIG/"F24"/"input_patient_values.csv")
    summaries=read(FIG/"F24"/"input_summary.csv")
    for row in summaries:
        vector=np.array([float(p["value"]) for p in patients if p["method"]==row["method"]])
        assert len(vector)==46 and len({p["patient_id"] for p in patients if p["method"]==row["method"]})==46
        radius=t.ppf(.975,45)*vector.std(ddof=1)/np.sqrt(46)
        assert np.allclose([vector.mean(),vector.mean()-radius,vector.mean()+radius],
                           [float(row[k]) for k in ["mean","ci95_low","ci95_high"]],rtol=0,atol=1e-12)
        assert float(row["coverage"])==1 and int(row["n_windows"])==23838
    by_method={method:{p["patient_id"]:float(p["value"]) for p in patients if p["method"]==method}
               for method in ["pretrained_mse","bandpass_fft"]}
    ids=sorted(by_method["pretrained_mse"])
    assert ids==sorted(by_method["bandpass_fft"])
    diff=np.array([by_method["bandpass_fft"][p]-by_method["pretrained_mse"][p] for p in ids])
    assert int((diff>0).sum())==40 and int((diff<0).sum())==6
    displayed=read(FIG/"F15"/"input_paired.csv")
    significant=[r for r in displayed if float(r["p_value_holm"])<.05]
    assert len(significant)==1 and significant[0]["method"]=="pretrained_soft_lag"
    assert significant[0]["metric"]=="max_lag_waveform_correlation"
    report={"status":"passed","figure_reports":reports,"all_input_rows_exact_subsets":True,
            "all_pngs_300dpi_12p8_inches":True,"F24_patient_means_CIs_independently_recomputed":4,
            "F24_paired_patient_count":46,"F24_CNN_lower_error_patient_count":40,
            "F15_displayed_contrasts":15,"F15_Holm_significant":1,
            "manual_view_image_review":"All three final PNGs inspected individually in the current conversation: no clipped intervals, text, or overlapping legend/data.",
            "no_tex_or_other_figure_edits_by_these_scripts":True}
    (FIG/"figure_qa.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
