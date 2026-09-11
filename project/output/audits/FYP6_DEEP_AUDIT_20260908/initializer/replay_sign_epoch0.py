"""No-gradient replay of two existing seed-0 Sign initializers; prints only.

Reads original saved configs and filters, uses the existing training split,
and does not optimise or write any original experiment artifact.
"""
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from domain_mf.data import build_loaders
from domain_mf.initializers import initialise_model, calibrate_logit_scale
import domain_mf.initializers as initializers
from domain_mf.models import build_model
from run_experiment import seed_everything

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []
for method in ("kaiming", "lowrank_wmf"):
    run_name = "sign_standard_" + method + "_layers3_seed0_frac1p0_epochs50_none0p1_shrink0p1_logit1p0"
    if method == "lowrank_wmf":
        run_name += "_rank16"
    run_name += "_lrcosine1e-05"
    run = ROOT / "outputs_convergence50" / run_name
    cfg = json.loads((run / "config.json").read_text())
    seed_everything(cfg["seed"])
    _, fit, _, _ = build_loaders(
        "sign", Path(cfg["data_root"]), Path(cfg["sign_root"]),
        cfg["batch_size"], cfg["seed"], cfg["train_fraction"],
        cfg["val_fraction"], num_workers=0)
    model = build_model("standard", "sign").to(device)
    covariance_stats = []
    original_lowrank = initializers._lowrank_discriminants
    def audited_lowrank(means, background, covariance, *args, **kwargs):
        diag = covariance.diag()
        covariance_stats.append({"dimension":len(diag),
            "zero_diagonal_entries":int((diag==0).sum()),
            "negative_diagonal_entries":int((diag<0).sum()),
            "minimum_raw_variance":float(diag.min()),
            "all_dimension_target":float(diag.mean()),
            "positive_dimension_target":float(diag[diag>0].mean())})
        return original_lowrank(means, background, covariance, *args, **kwargs)
    initializers._lowrank_discriminants = audited_lowrank
    report = initialise_model(model, fit, method, device,
        layers=3, shrinkage=cfg["shrinkage"], covariance_rank=cfg["covariance_rank"])
    initializers._lowrank_discriminants = original_lowrank
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x.to(device)) for i,(x,y) in enumerate(fit) if i < 10]).double()
    common = logits.mean(1, keepdim=True)
    centered = logits-common
    scale = calibrate_logit_scale(model, fit, device)
    saved = torch.load(run / "filters_initial.pt", map_location="cpu", weights_only=True)
    replay = model.state_dict()
    rows = {
        "method":method,
        "device":str(device),
        "covariance_diagonal_stats":covariance_stats,
        "raw_logit_std_before":float(logits.std()),
        "class_centered_std_after_raw_matching":float(centered.std())*scale,
        "common_mode_fraction_of_total_variance":float((common.expand_as(logits)-logits.mean()).square().sum()/(logits-logits.mean()).square().sum()),
        "replay_scale":scale,
        "saved_scale":cfg["initialisation_report"]["output_scale"],
        "saved_initial_filter_max_abs_difference":{k:float((v-replay[k+".weight"].cpu()).abs().max()) for k,v in saved.items()},
    }
    results.append(rows)

print(json.dumps(results, indent=2))
