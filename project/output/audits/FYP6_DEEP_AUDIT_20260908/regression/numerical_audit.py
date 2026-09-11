"""Read-only regression audit: synthetic checks and saved-waveform sensitivity.

No training, source edits, original-output writes, or checkpoint replay.
Run with the original CL2 Python from the repository root.
"""
from pathlib import Path
import csv
import json
import sys
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import t, ttest_rel
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_correncoder_regression import (
    _normalise_pair, build_fold_tensors, load_bidmc_subjects, respiratory_rate,
)
from evaluate_classical_rr_baselines import (
    autocorrelation_rate, bandpass, respiratory_envelope,
)
from domain_mf.regression import (
    lag_tolerant_pearson_loss, soft_lag_pearson_loss, spectral_shape_loss,
    SafeAffineCalibration,
)


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ci(values):
    a = np.asarray(values)
    h = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return {"mean": float(a.mean()), "ci95_low": float(a.mean()-h),
            "ci95_high": float(a.mean()+h)}


def fft_rates(x, nfft=918, hann=False):
    windows = np.lib.stride_tricks.sliding_window_view(x, 918)[::30]
    windows = windows - windows.mean(axis=-1, keepdims=True)
    if hann:
        windows = windows*np.hanning(918)
    frequencies = np.fft.rfftfreq(nfft, 1/30)
    band = (frequencies >= .08) & (frequencies <= .8)
    power = np.abs(np.fft.rfft(windows, n=nfft, axis=-1))**2
    return 60*frequencies[band][power[:, band].argmax(axis=-1)]


def annotation_rr(samples, n_windows):
    # Explicit audit estimator, not claimed to be the authors' historical code:
    # average cycle frequency between first/last annotated breaths inside each
    # exact evaluation window. MATLAB sample indices are converted to seconds.
    times = (np.asarray(samples, dtype=float)-1)/125
    result = []
    for start in range(n_windows):
        included = times[(times >= start) & (times < start+30.6)]
        result.append(60*(len(included)-1)/(included[-1]-included[0])
                      if len(included) >= 2 else np.nan)
    return np.asarray(result)


def local_peak_rate(x):
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    # Minimal bounded audit repair: enforce in-band true local maxima; retain
    # the historical biased autocorrelation itself to isolate boundary argmax.
    peaks, _ = find_peaks(ac)
    peaks = peaks[(peaks >= int(np.ceil(30/.8))) &
                  (peaks <= int(np.floor(30/.08)))]
    return float(1800/peaks[np.argmax(ac[peaks])]) if len(peaks) else np.nan


def main():
    torch.set_num_threads(2)
    summary = {"no_training": True, "checkpoint_replay": False,
               "original_outputs_modified": False}
    # Failure independent of physiology/noise: global lag argmax includes the
    # descending zero-lag lobe, and floor allows an estimate outside the band.
    tt = np.arange(918)/30
    synthetic = []
    for bpm in [4.8, 5, 6, 8, 10, 12, 15, 20, 30, 48]:
        x = np.sin(2*np.pi*bpm/60*tt)
        synthetic.append({"true_bpm": bpm, "old_ac_bpm": autocorrelation_rate(x,30),
                          "local_peak_bpm": local_peak_rate(x),
                          "fft_bpm": respiratory_rate(x,30)})
    summary["autocorrelation_pure_sines"] = synthetic
    # Shared-bin errors need not track actual frequency separation.
    examples = []
    for a,b in [(17,18),(17.5,18.5),(17.7,17.8),(19,20)]:
        aa,bb = [np.sin(2*np.pi*f/60*tt) for f in (a,b)]
        examples.append({"true_rates": [a,b], "true_error": abs(a-b),
                         "old_fft_error": abs(respiratory_rate(aa)-respiratory_rate(bb))})
    summary["fft_quantisation_examples"] = examples
    summary["fft_bin_spacing_bpm"] = 60/30.6
    # Inputs do not depend on labels: per-record target normalisation defines
    # the reference scale, but the held-out target is not fed into the model.
    ppg = np.sin(np.arange(100)/7)
    target = np.cos(np.arange(100)/9)
    xa, ya = _normalise_pair(ppg.copy(), target.copy())
    xb, yb = _normalise_pair(ppg.copy(), 3*target+10)
    summary["normalisation"] = {"max_input_change_on_target_affine_transform": float(np.max(abs(xa-xb))),
                                "max_normalized_target_change": float(np.max(abs(ya-yb)))}
    # Lag losses are differentiable on ordinary non-tie cases; no claim that
    # hard-max is differentiable at ties. The straight-through head deliberately
    # differs from the derivative of its forward physical scale.
    torch.manual_seed(7)
    y = torch.randn(2,1,288, dtype=torch.double)
    grad_checks = {}
    for name, loss in [("hard_lag",lag_tolerant_pearson_loss),
                       ("soft_lag",soft_lag_pearson_loss),
                       ("spectral",spectral_shape_loss)]:
        x = torch.randn_like(y, requires_grad=True)
        val = loss(x,y)
        g, = torch.autograd.grad(val,x)
        idx = (0,0,100)
        eps = 1e-6
        xp,xm = x.detach().clone(),x.detach().clone()
        xp[idx]+=eps; xm[idx]-=eps
        fd = float((loss(xp,y)-loss(xm,y))/(2*eps))
        const = torch.zeros_like(y, requires_grad=True)
        cv = loss(const,y)
        cg, = torch.autograd.grad(cv,const)
        grad_checks[name] = {"finite_difference":fd,"autograd":float(g[idx]),
                            "absolute_difference":abs(fd-float(g[idx])),
                            "constant_loss":float(cv.detach()),
                            "constant_gradient_finite":bool(torch.isfinite(cg).all()),
                            "constant_gradient_norm":float(cg.norm())}
    head = SafeAffineCalibration(.0001,.5)
    xx = torch.ones(1, requires_grad=True)
    head(xx).backward()
    grad_checks["safe_head"] = {"forward_gain":float(head.gain().detach()),
                                "backward_gain":float(xx.grad)}
    summary["gradients"] = grad_checks
    # Check raw source metadata and all records, not only one example.
    raw = loadmat(ROOT/"data_bidmc/bidmc_data.mat", simplify_cells=True)["data"]
    summary["bidmc_raw_shapes_rates"] = sorted({
        (len(s["ppg"]["v"]),len(s["ref"]["resp_sig"]["imp"]["v"]),
         s["ppg"]["fs"],s["ref"]["resp_sig"]["imp"]["fs"])
        for s in raw})
    cap = []
    for p in sorted((ROOT/"data_capnobase").glob("*_8min.mat")):
        with h5py.File(p) as h:
            cap.append({"file":p.name,"ppg_fs":float(h["param/samplingrate/pleth"][0,0]),
                        "co2_fs":float(h["param/samplingrate/co2"][0,0]),
                        "ppg_shape":list(h["signal/pleth/y"].shape),
                        "co2_shape":list(h["signal/co2/y"].shape)})
    summary["capno_metadata"] = cap
    subjects = load_bidmc_subjects(ROOT/"data_bidmc/bidmc_data.mat")
    tr_x,tr_y,tx,ty,starts = build_fold_tensors(subjects,0)
    summary["fold_shapes"] = {"train":list(tr_x.shape),"test":list(tx.shape),
                              "reconstructed_samples":int(starts[-1]+288),
                              "rr_windows":len(range(0,int(starts[-1]+288)-918+1,30))}
    del tr_x,tr_y,tx,ty
    rows=[]
    all_schemes = ["old_fft", "zero_pad_8192", "hann_zero_pad_8192", "manual_ann1", "manual_ann2", "manual_mean"]
    baseline_boundary=0; ac_windows=0
    for fold, subject in enumerate(subjects):
        wf = ROOT/"outputs_correncoder_regression_full"/f"bidmc_fold{fold:02d}_seed{55+fold}"/"waveforms.npz"
        arrays=np.load(wf)
        pred=arrays["prediction"]; ref=arrays["target"]
        proxies={"correncoder":pred,
                 "bandpass_fft":bandpass(subject["ppg"],30,.08,.8),
                 "envelope_fft":respiratory_envelope(subject["ppg"],30)}
        rates={}
        for scheme,nfft,hann in [("old_fft",918,False),("zero_pad_8192",8192,False),
                                 ("hann_zero_pad_8192",8192,True)]:
            reference=fft_rates(ref,nfft,hann)
            for method,proxy in proxies.items():
                est=fft_rates(proxy,nfft,hann)[:len(reference)]
                rates[(scheme,method)]=est
                err=abs(est-reference)
                rows.append({"fold":fold,"scheme":scheme,"method":method,
                             "mae":float(err.mean()),"windows":len(err),
                             "zero_error_fraction":float(np.mean(err==0))})
        n=len(rates[("old_fft","correncoder")])
        ann1=annotation_rr(raw[fold]["ref"]["breaths"]["ann1"],n)
        ann2=annotation_rr(raw[fold]["ref"]["breaths"]["ann2"],n)
        for scheme,reference in [("manual_ann1",ann1),("manual_ann2",ann2),
                                 ("manual_mean",(ann1+ann2)/2)]:
            for method in proxies:
                est=rates[("old_fft",method)]
                err=abs(est-reference)
                rows.append({"fold":fold,"scheme":scheme,"method":method,
                             "mae":float(np.nanmean(err)),"windows":int(np.isfinite(err).sum()),
                             "zero_error_fraction":float(np.mean(err==0))})
            ref_error=abs(fft_rates(ref)-reference)
            rows.append({"fold":fold,"scheme":scheme,"method":"impedance_fft_vs_annotations",
                         "mae":float(np.nanmean(ref_error)),"windows":int(np.isfinite(ref_error).sum()),
                         "zero_error_fraction":float(np.mean(ref_error==0))})
        windows=np.lib.stride_tricks.sliding_window_view(proxies["bandpass_fft"],918)[::30]
        oldac=np.asarray([autocorrelation_rate(w,30) for w in windows])
        newac=np.asarray([local_peak_rate(w) for w in windows])
        baseline_boundary+=int(np.sum(oldac>48)); ac_windows+=len(oldac)
        reference=fft_rates(ref)
        for method,rr in [("historical_autocorrelation",oldac),("local_peak_autocorrelation",newac)]:
            err=abs(rr-reference)
            rows.append({"fold":fold,"scheme":"old_fft","method":method,
                         "mae":float(np.nanmean(err)),"windows":int(np.isfinite(err).sum()),
                         "zero_error_fraction":float(np.mean(err==0))})
    with (OUT/"rr_sensitivity_subject_rows.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for scheme,method in sorted({(r["scheme"],r["method"]) for r in rows}):
        subset=[r for r in rows if r["scheme"]==scheme and r["method"]==method]
        aggregate.append({"scheme":scheme,"method":method,"subjects":len(subset),
                          **ci([r["mae"] for r in subset]),
                          "total_windows":sum(r["windows"] for r in subset),
                          "zero_error_fraction":float(np.mean([r["zero_error_fraction"] for r in subset]))})
    paired=[]
    for scheme in all_schemes:
        for method in ["bandpass_fft","envelope_fft"]:
            a=np.asarray([r["mae"] for r in rows if r["scheme"]==scheme and r["method"]==method])
            b=np.asarray([r["mae"] for r in rows if r["scheme"]==scheme and r["method"]=="correncoder"])
            paired.append({"scheme":scheme,"comparison":method+"_minus_correncoder",
                           **ci(a-b),"p_raw":float(ttest_rel(a,b).pvalue),"wins":int(np.sum(a>b))})
    summary["rr_sensitivity_aggregate"]=aggregate
    summary["rr_sensitivity_paired"]=paired
    summary["autocorrelation_outside_band"]={"windows_above_48_bpm":baseline_boundary,
                                              "total_windows":ac_windows,
                                              "fraction":baseline_boundary/ac_windows}
    # Recover actual experimental confounding from authoritative raw configs.
    labels={"stem":"outputs_correncoder_initialisation/matched_1d_mse",
            "layerwise_affine":"outputs_correncoder_extensions/matched_layerwise_calibrated_mse",
            "depth1_safe":"outputs_correncoder_depth_ste/depth1",
            "depth3_safe":"outputs_correncoder_depth_ste/depth3_diagonal"}
    config_rows={}; metrics={}
    for label,rel in labels.items():
        config=json.loads((ROOT/rel/"experiment_config.json").read_text())
        config_rows[label]={k:v for k,v in config.items() if k!="folds"}
        with (ROOT/rel/"fold_metrics.csv").open(newline="",encoding="utf-8") as f:
            records=list(csv.DictReader(f))
        metrics[label]=np.asarray([float(r["waveform_correlation"]) for r in records])
    summary["layerwise_configs"]=config_rows
    summary["layerwise_effects"]={}
    for a,b in [("layerwise_affine","stem"),("depth3_safe","depth1_safe")]:
        d=metrics[a]-metrics[b]
        summary["layerwise_effects"][a+"_minus_"+b]={**ci(d),
            "p_raw":float(ttest_rel(metrics[a],metrics[b]).pvalue),
            "cohen_dz":float(d.mean()/d.std(ddof=1))}
    write_json("numerical_audit_results.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["capno_metadata","layerwise_configs"]},indent=2))


if __name__=="__main__":
    main()
