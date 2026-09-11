"""Inspect original ResNet checkpoints and initial stem banks; no writes."""
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
rows = []
for run in sorted((ROOT / "outputs_resnet18_cross_task").glob("*lowrank_wmf*")):
    initial_path = run / "filters_initial.pt"
    if not initial_path.exists():
        continue
    cfg = json.loads((run / "config.json").read_text())
    w = torch.load(initial_path, map_location="cpu", weights_only=True)["conv1"]
    checkpoint = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    target = math.sqrt(2 / w[0].numel())
    rows.append({"run":run.name,"requested_rank":cfg["covariance_rank"],
        "stem_shape":list(w.shape),"stored_stem_std":float(w.std()),
        "kaiming_target_std":target,"std_target_ratio":float(w.std())/target,
        "checkpoint_has_conv1_bias":"conv1.bias" in checkpoint,
        "checkpoint_has_bn1_bias":"bn1.bias" in checkpoint})
print(json.dumps(rows, indent=2))
