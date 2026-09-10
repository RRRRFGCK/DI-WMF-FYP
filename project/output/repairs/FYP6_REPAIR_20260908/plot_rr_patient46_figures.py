"""Redraw only F13/F15/F24 from the frozen patient46 evaluation; PNG only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "physiology_evaluation"
DEST = HERE / "thesis" / "figures" / "main"
SOURCES = HERE / "physiology_figure_sources"
NAMES = {
    "F13": "F13_correncoder_regression_results.png",
    "F15": "F15_correncoder_depth_lag_forest.png",
    "F24": "F24_classical_rr_baselines.png",
}
COLORS = ["#697888", "#C77C16", "#276BA0", "#278061", "#8054A0", "#B54B49", "#23858B"]
F13_METHODS = ["random_mse", "matched_1d_mse", "pretrained_mse", "pretrained_corr",
               "pretrained_spectral", "pretrained_corr_spectral", "matched_layerwise_calibrated_mse"]
F13_LABELS = ["Random + MSE", "Matched stem + MSE", "Pretrained + MSE",
              "Pretrained + MSE + correlation", "Pretrained + MSE + spectral", "Pretrained + MSE + corr. + spectral",
              "Layerwise + affine + MSE (compound)"]
F15_PAIRS = [
    ("matched_layerwise_calibrated_mse", "matched_1d_mse", "Layerwise + affine vs stem\n(compound treatment)"),
    ("depth3_diagonal", "depth1", "Depth 3 diagonal vs depth 1\n(same safe head)"),
    ("depth3_lowrank", "depth3_diagonal", "Depth 3 low-rank vs diagonal\n(same safe head)"),
    ("pretrained_hard_lag", "pretrained_mse", "Hard-lag vs pretrained MSE"),
    ("pretrained_soft_lag", "pretrained_mse", "Soft-lag vs pretrained MSE"),
]
F15_METRICS = ["test_mse", "max_lag_waveform_correlation", "rr_manual_primary_mae_bpm"]
F24_METHODS = ["pretrained_mse", "bandpass_fft", "envelope_fft", "bandpass_autocorrelation_corrected"]
MANUAL = "rr_manual_primary_mae_bpm"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def num(row, key="mean"):
    return float(row[key])


def ci_error(row):
    return np.array([[num(row)-num(row,"ci95_low")], [num(row,"ci95_high")-num(row)]])


def style_axes(ax, grid="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A3ABB2")
    ax.tick_params(colors="#26313A", length=4)
    ax.grid(axis=grid, color="#E8EBED", lw=.8)
    ax.set_axisbelow(True)


def save_figure(fig, fig_id, csv_paths, source_names, selection):
    output = DEST / NAMES[fig_id]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text().strip():
            bb = item.get_window_extent(renderer)
            if bb.width and bb.height and (bb.x0 < -1 or bb.y0 < -1 or
                    bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1):
                outside.append(item.get_text())
    assert not outside, (fig_id, "text beyond canvas", outside)
    fig.savefig(output, dpi=300, facecolor="white", metadata={"Software":"Matplotlib; frozen patient46 RR evaluation"})
    plt.close(fig)
    with Image.open(output) as im:
        assert im.width == 3840, (fig_id, im.size)
        assert abs(im.info["dpi"][0]-300)<.1
        size = list(im.size)
        dpi = list(im.info["dpi"])
    registry = {
        "figure":fig_id, "output_png":str(output), "output_sha256":sha(output),
        "pixel_dimensions":size, "dpi":dpi, "width_inches":12.8,
        "plot_source":str(Path(__file__).resolve()), "plot_source_sha256":sha(__file__),
        "font":FONT, "font_file":font_manager.findfont(FONT, fallback_to_default=False),
        "original_inputs":{name:sha(DATA/name) for name in source_names},
        "figure_input_csv":{str(path):sha(path) for path in csv_paths},
        "evaluation_manifest_sha256":sha(DATA/"physiology_evaluation_manifest_patient46.json"),
        "selection":selection,
        "patient_count":46, "record_count":53, "manual_valid_windows":23838,
        "ci":"unadjusted patient t 95% CI; descriptive due to shared CV training sets",
        "p_adjustment":"one frozen repair-stage exploratory Holm family of 120 tests",
        "outputs":"PNG only; no PDF generated",
        "automated_layout_QA":{"no_text_beyond_canvas":True,"ci_axis_limits_checked":True},
    }
    (SOURCES/fig_id/"hash_manifest.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    print(json.dumps({"figure":fig_id,"path":str(output),"sha256":sha(output),"pixels":size}),flush=True)


def f13():
    folder = SOURCES/"F13"; folder.mkdir(parents=True,exist_ok=True)
    selected = [r for r in SUMMARY if r["method"] in F13_METHODS and
                r["metric"] in ["test_mae","waveform_correlation",MANUAL]]
    assert len(selected)==21
    inputs = folder/"input_summary.csv"; write_csv(inputs,selected)
    rows = {(r["method"],r["metric"]):r for r in load_csv(inputs)}
    fig,(left,right) = plt.subplots(1,2,figsize=(12.8,7.0),gridspec_kw={"width_ratios":[1.15,1]})
    fig.subplots_adjust(left=.085,right=.975,top=.855,bottom=.31,wspace=.27)
    fig.suptitle("Waveform and human-reference RR endpoints",fontsize=17,fontweight="bold",y=.97)
    fig.text(.5,.918,"46-patient means with 95% patient-t intervals",ha="center",fontsize=13,color="#4C5963")
    positions = [(.232,.132),(.272,.150),(.272,.216),(.218,.278),(.270,.289),(.226,.345),(.251,.346)]
    for i,(method,color,position) in enumerate(zip(F13_METHODS,COLORS,positions),start=1):
        x,y,r = rows[method,"test_mae"],rows[method,"waveform_correlation"],rows[method,MANUAL]
        left.errorbar(num(x),num(y),xerr=ci_error(x),yerr=ci_error(y),fmt="none",ecolor=color,
                      alpha=.48,elinewidth=1.5,capsize=3,zorder=1)
        left.scatter(num(x),num(y),s=72,color=color,edgecolors="white",linewidths=1,zorder=3)
        left.annotate(str(i),(num(x),num(y)),xytext=position,textcoords="data",ha="center",va="center",
                      color=color,fontweight="bold",fontsize=13,
                      bbox={"boxstyle":"circle,pad=.23","fc":"white","ec":color,"lw":1.1},
                      arrowprops={"arrowstyle":"-","color":color,"lw":1},zorder=4)
        yy=7-i
        right.errorbar(num(r),yy,xerr=ci_error(r),fmt="o",color=color,markersize=7,
                       capsize=4,elinewidth=2,zorder=3)
    left.set(xlim=(.205,.287),ylim=(.07,.399),xlabel="Test-window MAE (normalized units)",
             ylabel="Fused-waveform zero-lag correlation")
    left.set_xticks([.21,.23,.25,.27]); left.set_yticks([.1,.2,.3,.4])
    left.set_title("A  Waveform endpoints",loc="left",pad=12,fontweight="bold")
    right.set(xlim=(.90,2.55),ylim=(-.65,6.65),xlabel="Manual-reference RR MAE (bpm)")
    right.set_yticks(range(7),list(map(str,range(7,0,-1))))
    right.set_ylabel("Configuration number")
    right.set_xticks([1,1.5,2,2.5]);right.set_title("B  Human-reference RR",loc="left",pad=12,fontweight="bold")
    for method in F13_METHODS:
        for ax,metric,dimension in [(left,"test_mae","x"),(left,"waveform_correlation","y"),(right,MANUAL,"x")]:
            low,high = ax.get_xlim() if dimension=="x" else ax.get_ylim()
            assert low < num(rows[method,metric],"ci95_low") < num(rows[method,metric],"ci95_high") < high
    style_axes(left);style_axes(right,grid="x")
    handles=[Line2D([],[],marker="o",linestyle="none",markersize=7,color=c,label=f"{i}  {label}")
             for i,(label,c) in enumerate(zip(F13_LABELS,COLORS),start=1)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.51,.055),ncol=2,
               frameon=False,fontsize=12.5,columnspacing=2.8,handletextpad=.55,labelspacing=.60)
    fig.text(.5,.022,"RR: 23,838 shared manual-valid windows. Intervals are not simultaneous; comparisons are exploratory.",
             ha="center",fontsize=11.8,color="#4C5963")
    save_figure(fig,"F13",[inputs],["physiology_summary_patient46.csv"],
                "Seven main init/loss configurations; window MAE versus fused zero-lag correlation; separate manual RR panel")


def f15():
    folder=SOURCES/"F15";folder.mkdir(parents=True,exist_ok=True)
    selected=[r for r in PAIRED if (r["method"],r["reference"]) in [(a,b) for a,b,_ in F15_PAIRS]
              and r["metric"] in F15_METRICS]
    assert len(selected)==15
    assert [(r["method"],r["metric"]) for r in selected if num(r,"p_value_holm")<.05] == [
        ("pretrained_soft_lag","max_lag_waveform_correlation")]
    inputs=folder/"input_paired.csv";write_csv(inputs,selected)
    rows={(r["method"],r["reference"],r["metric"]):r for r in load_csv(inputs)}
    fig,axes=plt.subplots(1,3,figsize=(12.8,6.6),sharey=True)
    fig.subplots_adjust(left=.30,right=.98,bottom=.185,top=.82,wspace=.27)
    fig.suptitle("Selected patient-paired contrasts",fontsize=17,fontweight="bold",y=.968)
    fig.text(.64,.913,"Method minus reference; 46 patients per contrast",ha="center",fontsize=13,color="#4C5963")
    titles=["A  Window MSE", "B  Maximum-lag correlation", "C  Manual RR MAE"]
    labels=["Δ MSE\n(normalized squared units)","Δ correlation\n(unitless)","Δ RR MAE\n(bpm)"]
    limits=[(-.018,.013),(-.047,.083),(-.22,.24)]
    ticks=[[-.015,0,.010],[-.04,0,.04,.08],[-.2,0,.2]]
    for ax,metric,title,label,limit,xticks in zip(axes,F15_METRICS,titles,labels,limits,ticks):
        style_axes(ax,grid="x");ax.axvline(0,color="#677581",lw=1,ls="--")
        for i,(first,second,_) in enumerate(F15_PAIRS):
            row=rows[first,second,metric]; y=4-i; p=num(row,"p_value_holm")
            assert limit[0] < num(row,"ci95_low") < num(row,"ci95_high") < limit[1], (metric,first)
            color="#276BA0" if p<.05 else "#71818E"
            ax.errorbar(num(row),y,xerr=ci_error(row),fmt="D" if p<.05 else "o",color=color,
                         markersize=7,capsize=4,elinewidth=2,zorder=3)
            ptext=f"Holm p = {p:.3f}" + (" *" if p<.05 else "")
            ax.text(.98,y-.29,ptext,transform=ax.get_yaxis_transform(),ha="right",va="center",
                    fontsize=11.7,color=color,fontweight="bold" if p<.05 else "normal")
        ax.set(xlim=limit,ylim=(-.58,4.55),xlabel=label);ax.set_xticks(xticks)
        ax.set_title(title,loc="left",pad=16,fontsize=12.8,fontweight="bold")
    axes[0].set_yticks(range(4,-1,-1),[x[2] for x in F15_PAIRS])
    axes[0].tick_params(axis="y",labelsize=12.7,pad=12,length=0)
    for ax in axes[1:]:ax.tick_params(axis="y",length=0)
    fig.text(.64,.064,"Bars: unadjusted 95% patient-t CIs. Holm p: frozen family of 120 tests.",
             ha="center",fontsize=12,color="#4C5963")
    fig.text(.64,.027,"Only the shown soft-lag correlation contrast passes Holm; shared-CV dependence remains.",
             ha="center",fontsize=11.6,color="#4C5963")
    save_figure(fig,"F15",[inputs],["physiology_paired_holm_patient46.csv"],
                "Five already-frozen contrasts across window MSE, maximum-lag correlation, and primary manual RR")


def f24():
    folder=SOURCES/"F24";folder.mkdir(parents=True,exist_ok=True)
    patients=[r for r in PATIENTS if r["method"] in F24_METHODS and r["metric"]==MANUAL]
    summaries=[r for r in SUMMARY if r["method"] in F24_METHODS and r["metric"]==MANUAL]
    paired=[r for r in PAIRED if r["method"]=="bandpass_fft" and r["reference"]=="pretrained_mse" and r["metric"]==MANUAL]
    assert len(patients)==184 and len(summaries)==4 and len(paired)==1
    assert all(num(r,"coverage")==1 for r in summaries)
    paths=[folder/"input_patient_values.csv",folder/"input_summary.csv",folder/"input_paired.csv"]
    for path,rows in zip(paths,[patients,summaries,paired]):write_csv(path,rows)
    patients,summaries,paired=[load_csv(path) for path in paths]
    pmap={(r["method"],r["patient_id"]):num(r,"value") for r in patients}
    ids=sorted({r["patient_id"] for r in patients});assert len(ids)==46
    smap={r["method"]:r for r in summaries};contrast=paired[0]
    xx=np.array([pmap["pretrained_mse",p] for p in ids]);yy=np.array([pmap["bandpass_fft",p] for p in ids])
    assert int((xx<yy).sum())==int(contrast["reference_wins"])==40
    assert np.all((xx>=0)&(xx<13)) and np.all((yy>=0)&(yy<13))
    fig,(left,right)=plt.subplots(1,2,figsize=(12.8,7.0),gridspec_kw={"width_ratios":[1,1.17]})
    fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.21,wspace=.26)
    fig.suptitle("Manual-reference RR: patient-level comparisons",fontsize=17,fontweight="bold",y=.97)
    fig.text(.5,.92,"46 patients, 53 records, 23,838 shared windows; 100% estimator coverage",ha="center",fontsize=13,color="#4C5963")
    left.plot([0,13],[0,13],color="#8896A0",ls="--",lw=1.4,zorder=1,label="Equal error")
    left.scatter(xx,yy,s=47,color="#276BA0",alpha=.80,edgecolor="white",lw=.7,zorder=3)
    left.set(xlim=(0,13),ylim=(0,13),xlabel="Pretrained MSE: patient RR MAE (bpm)",
             ylabel="Band-pass FFT: patient RR MAE (bpm)")
    left.set_aspect("equal",adjustable="box");left.set_xticks([0,3,6,9,12]);left.set_yticks([0,3,6,9,12])
    left.set_title("A  Paired patient errors",loc="left",pad=12,fontweight="bold")
    left.text(7.7,9.1,"Equal error",rotation=45,ha="center",va="center",fontsize=12,color="#647580")
    for i,method in enumerate(F24_METHODS):
        values=np.array([pmap[method,p] for p in ids])
        # Deterministic offsets depend on patient identity, never on the outcome.
        jitter=np.array([(int(hashlib.sha256((method+p).encode()).hexdigest()[:8],16)/0xFFFFFFFF-.5)*.26 for p in ids])
        color="#276BA0" if i==0 else "#83949F"
        right.scatter(i+jitter-.06,values,s=28,color=color,alpha=.70,edgecolor="white",linewidth=.4,zorder=2)
        row=smap[method]
        assert values.min()>=0 and max(values.max(),num(row,"ci95_high"))<14.4 and num(row,"ci95_low")>0
        right.errorbar(i+.22,num(row),yerr=ci_error(row),fmt="D",color="#162D40",mfc="white",mew=1.7,
                        markersize=7.5,capsize=5,elinewidth=2,zorder=4)
    right.set_xticks(range(4),["Pretrained\nMSE","Band-pass\nFFT","Envelope\nFFT","Corrected\nautocorr."])
    right.set(xlim=(-.48,3.49),ylim=(0,14.4),ylabel="Patient RR MAE (bpm)")
    right.set_yticks([0,3,6,9,12]);right.set_title("B  Patient distributions and mean CIs",loc="left",pad=12,fontweight="bold")
    style_axes(left);style_axes(right,grid="y")
    fig.text(.25,.104,"Band-pass FFT minus pretrained MSE: 1.709 [0.744, 2.673] bpm",ha="center",fontsize=11.7)
    fig.text(.25,.063,"Holm p = 0.100 (not significant); CNN lower in 40/46 patients",ha="center",fontsize=11.7)
    handles=[Line2D([],[],marker="o",linestyle="none",markersize=6,color="#83949F",label="One patient"),
             Line2D([],[],marker="D",linestyle="-",markersize=6,mfc="white",color="#162D40",label="Mean and 95% CI")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.735,.066),ncol=2,frameon=False,fontsize=12,columnspacing=1.1)
    fig.text(.5,.020,"Unadjusted patient-t CIs; one frozen exploratory Holm family of 120 comparisons; shared CV training remains dependent.",
             ha="center",fontsize=11.5,color="#4C5963")
    save_figure(fig,"F24",paths,["physiology_patient_metrics_patient46.csv","physiology_summary_patient46.csv","physiology_paired_holm_patient46.csv"],
                "Primary dual-human annotation reference only; headline CNN versus three classical estimators; scatter is patient-paired")


def main():
    global SUMMARY,PAIRED,PATIENTS,FONT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures",nargs="+",choices=list(NAMES),default=list(NAMES))
    args=parser.parse_args()
    manifest=json.loads((DATA/"physiology_evaluation_manifest_patient46.json").read_text())
    assert manifest["status"]=="complete" and manifest["patient_groups"]==46
    assert manifest["primary_common_valid_windows"]==23838 and manifest["inference_family_size"]==120
    for name in ["physiology_summary_patient46.csv","physiology_paired_holm_patient46.csv","physiology_patient_metrics_patient46.csv"]:
        assert sha(DATA/name)==manifest["table_hashes"][name], name
    SUMMARY=load_csv(DATA/"physiology_summary_patient46.csv")
    PAIRED=load_csv(DATA/"physiology_paired_holm_patient46.csv")
    PATIENTS=load_csv(DATA/"physiology_patient_metrics_patient46.csv")
    FONT="Arial";font_manager.findfont(FONT,fallback_to_default=False)
    plt.rcParams.update({"font.family":FONT,"font.size":13,"axes.labelsize":13,"axes.titlesize":14,
                         "xtick.labelsize":12,"ytick.labelsize":12,"axes.linewidth":.8,
                         "figure.facecolor":"white","savefig.facecolor":"white","axes.unicode_minus":True})
    DEST.mkdir(parents=True,exist_ok=True);SOURCES.mkdir(parents=True,exist_ok=True)
    for name in args.figures:{"F13":f13,"F15":f15,"F24":f24}[name]()


if __name__=="__main__":main()
