"""Read-only mechanism checks. No training or writes outside this audit folder."""
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from domain_mf.models import C4RotationInvariantCNN, SPECS, StandardCNN
from evaluate_template_semantics import exact_and_gradcam

torch.set_num_threads(2)
torch.manual_seed(92831)
result = {}
rotation = []
for task in ("fashion", "cifar10", "sign"):
    model = C4RotationInvariantCNN(SPECS[task]).double().eval()
    x = torch.randn(2, SPECS[task].input_channels, SPECS[task].image_size,
                    SPECS[task].image_size, dtype=torch.float64)
    with torch.no_grad():
        y = model(x)
        errors = [(model(torch.rot90(x, k, (-2, -1))) - y).abs().max().item()
                  for k in (1, 2, 3)]
    rotation.append({"task": task, "quarter_turn_max_errors": errors})
result["c4_formal_input_shapes"] = rotation

model = StandardCNN(SPECS["fashion"]).double().eval()
x = torch.randn(4, 1, 20, 20, dtype=torch.float64)
targets = torch.arange(4)
logits, cam, gradcam = exact_and_gradcam(model, x, targets)
with torch.no_grad():
    features = model.second_layer_features(x)
    weights = model.classifier.weight[targets]
    raw = (features * weights[:, :, None, None]).sum(1)
    bias = model.classifier.bias[targets]
    selected = logits[torch.arange(4), targets]
    result["cam"] = {
        "normalised_cam_gradcam_max_diff": (cam-gradcam).abs().max().item(),
        "raw_signed_completeness_max_error": (raw.mean((1,2))+bias-selected).abs().max().item(),
        "rectified_completeness_max_error": (raw.relu().mean((1,2))+bias-selected).abs().max().item(),
        "normalised_completeness_max_error": (cam.mean((1,2))+bias-selected).abs().max().item(),
        "note": "Synthetic numerical identity check, not a new trained-model performance estimate."
    }

# Exact expectation for five iid uniform ten-class labels, not a data-derived null.
purities = []
for labels in itertools.product(range(10), repeat=5):
    counts = [labels.count(c) for c in set(labels)]
    entropy = -sum((n/5)*math.log(n/5) for n in counts)
    purities.append(1-entropy/math.log(10))
result["five_label_entropy_measure"] = {
    "floor": min(purities), "iid_uniform_null_exact_mean": sum(purities)/len(purities),
    "warning": "Uniform iid diagnostic, not permutation-adjusted empirical significance."
}
groups = defaultdict(lambda: {"channels": 0, "zero_top_activation": 0,
                              "purity_sum": 0.0, "active_purity_sum": 0.0})
for path in sorted((ROOT/"template_semantics_results").glob("*/template_channel_rows.csv")):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (path.parent.name, row["dataset"], row["method"])
            group = groups[key]
            dead = float(row["mean_top_activation"]) == 0.0
            group["channels"] += 1
            group["zero_top_activation"] += int(dead)
            group["purity_sum"] += float(row["semantic_purity"])
            if not dead:
                group["active_purity_sum"] += float(row["semantic_purity"])
rows = []
for key, group in groups.items():
    n = group["channels"]
    active = n-group["zero_top_activation"]
    rows.append(dict(tier=key[0], dataset=key[1], method=key[2], channels=n,
                     zero_top_activation=group["zero_top_activation"],
                     mean_purity=group["purity_sum"]/n,
                     active_only_mean_purity=group["active_purity_sum"]/active if active else None))
result["saved_semantic_records"] = rows
(HERE/"mechanism_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
