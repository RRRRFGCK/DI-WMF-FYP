"""Read-only source audit: synthetic numerical tests, no training or source edits."""
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from domain_mf.initializers import (
    _class_feature_full_stats, _class_feature_stats, _collect_salient_patches,
    _copy_parameters, _diagonal_discriminants, _lowrank_discriminants,
    _rescale_bank, _stabilise_diagonal_variance, calibrate_logit_scale,
    fit_classifier_head, initialise_model,
)
from domain_mf.models import ModelSpec, StandardCNN

torch.set_num_threads(1)
torch.manual_seed(608)
DT = torch.float64
CPU = torch.device("cpu")
out = {}

# LDA / arbitrary common Gaussian reference: scores differ only by common term.
d, c = 7, 4
means = torch.randn(c, d, dtype=DT)
background = torch.randn(d, dtype=DT)
variance = torch.rand(d, dtype=DT) + 0.4
priors = torch.arange(1, c + 1, dtype=DT); priors /= priors.sum()
x = torch.randn(31, d, dtype=DT)
w, b, v = _diagonal_discriminants(means, background, variance, priors, .1, 1e-6)
ref = torch.distributions.MultivariateNormal(background, torch.diag(v)).log_prob(x)
llr = torch.stack([
    torch.distributions.MultivariateNormal(means[k], torch.diag(v)).log_prob(x)
    - ref + priors[k].log() for k in range(c)
], dim=1)
z = x @ w.T + b
lda = x @ (means / v).T - .5 * (means.square() / v).sum(1) + priors.log()
out["diagonal_llr"] = {
    "maximum_abs_error": float((z - llr).abs().max()),
    "softmax_vs_lda_max_abs_error": float((z.softmax(1)-lda.softmax(1)).abs().max()),
    "all_argmax_equal": bool((z.argmax(1)==lda.argmax(1)).all()),
}

# Full pooled within-class moments agree with direct residual construction.
labels = torch.tensor([0]*5 + [1]*7 + [2]*4)
features = torch.randn(len(labels), 9, dtype=DT) + labels[:,None]
loader = DataLoader(TensorDataset(features, labels), batch_size=5)
mu, bg, cov, p = _class_feature_full_stats(loader, lambda x:x, 3, CPU)
m2, b2, var, p2 = _class_feature_stats(loader, lambda x:x, 3, CPU)
residual = features - mu[labels]
direct = residual.T @ residual / (len(labels)-3)
out["pooled_stats"] = {
    "full_covariance_max_abs_error": float((cov-direct).abs().max()),
    "diagonal_max_abs_error": float((var-direct.diag()).abs().max()),
    "means_max_abs_error": float((mu-m2).abs().max()),
}

# Woodbury implementation vs explicit inverse of EXACT represented D+U Lambda U'.
woodbury = []
for d, latent, rank in [(25, 25, 16), (75, 8, 16), (80, 30, 16), (360, 20, 16)]:
    a = torch.randn(d, latent, dtype=DT)
    s = a @ a.T / latent
    tau = s.diag().mean().clamp_min(1e-6)
    shrunk = .9*s + .1*tau*torch.eye(d,dtype=DT)
    vals, vecs = torch.linalg.eigh(shrunk)
    vals, vecs = vals[-rank:].clamp_min(1e-6), vecs[:,-rank:]
    diag = (shrunk.diag()-(vecs.square()*vals).sum(1)).clamp_min(max(1e-6,float(tau)*1e-6))
    represented = torch.diag(diag) + (vecs*vals) @ vecs.T
    precision, _, reported_diag, effective = _lowrank_discriminants(
        torch.eye(d,dtype=DT), torch.zeros(d,dtype=DT), s,
        torch.ones(d,dtype=DT), .1, 1e-6, rank)
    exact = torch.linalg.inv(represented)
    woodbury.append({"d":d,"input_rank":latent,"retained_rank":effective,
        "min_eigenvalue_covariance":float(torch.linalg.eigvalsh(represented).min()),
        "min_eigenvalue_precision":float(torch.linalg.eigvalsh(precision).min()),
        "relative_inverse_error":float(torch.linalg.norm(precision-exact)/torch.linalg.norm(exact)),
        "inverse_residual_max_abs":float((represented @ precision-torch.eye(d,dtype=DT)).abs().max()),
        "reported_diagonal_max_abs_error":float((reported_diag-diag).abs().max())})
out["woodbury"] = woodbury

# Full-bank positive rescale preserves EVERY detector zero set, not only argmax.
weight = torch.randn(12, 3, 5, 5, dtype=DT)
bias = torch.randn(12,dtype=DT)
ww, bb = _rescale_bank(weight, bias)
scale = math.sqrt(2/75) / float(weight.std())
out["bank_rescale"] = {
    "std":float(ww.std()),"target_std":math.sqrt(2/75),
    "bias_same_scale_max_abs":float((bb.double()-bias*scale).abs().max()),
}

# Exact fitting-head stage equivalence, using small synthetic 2-class model.
spec = ModelSpec("synthetic", 8, 1, 2, 4, 2)
im = torch.randn(24,1,8,8)
lab = torch.arange(24) % 2
dl = DataLoader(TensorDataset(im,lab),batch_size=8)
model = StandardCNN(spec)
initialise_model(model, dl, "lowrank_wmf", CPU, layers=3, covariance_rank=3)
pre = {k:v.clone() for k,v in model.state_dict().items()}
fit_classifier_head(model, dl, "lowrank_wmf", CPU, covariance_rank=3)
out["classifier_stage_equivalence"] = {
    "largest_parameter_difference":max(float((v-pre[k]).abs().max()) for k,v in model.state_dict().items()),
    "convolutions_unchanged":all(torch.equal(v,pre[k]) for k,v in model.state_dict().items() if k.startswith("conv")),
}

# Same per-class cap does not rebalance an unequal small dataset.
ulab = torch.tensor([0]*3+[1]*8)
udl = DataLoader(TensorDataset(torch.randn(11,1,8,8),ulab),batch_size=4)
patches = _collect_salient_patches(udl,lambda x:x,2,3,1,2,CPU,max_patches_per_class=100)
out["per_class_cap_not_general_balancing"] = [len(p) for p in patches]

# Source's bias-copy behavior in a bias-free convolution.
conv = nn.Conv2d(1,1,1,bias=False)
_copy_parameters(conv,torch.tensor([[[[1.0]]]]),torch.tensor([-2.0]))
out["bias_free_detector"] = {
    "bias_is_none":conv.bias is None,
    "code_response_at_x1":float(conv(torch.ones(1,1,1,1)).item()),
    "gaussian_detector_with_fitted_bias_at_x1":-1.0,
}

# Gauge-equivalent logits BEFORE calibration need not have comparable centered
# softmax-relevant spread AFTER global raw-logit standard deviation matching.
class GaugeModel(nn.Module):
    def __init__(self, common_gain):
        super().__init__()
        self.classifier = nn.Linear(2,3,bias=True).double()
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.,common_gain],[0.,common_gain],[-1.,common_gain]],dtype=DT))
            self.classifier.bias.zero_()
    @property
    def output_layer(self):
        return self.classifier
    def forward(self,x):
        return self.classifier(x)

gx = torch.randn(200,2,dtype=DT)
gdl = DataLoader(TensorDataset(gx,torch.zeros(200,dtype=torch.long)),batch_size=50)
models = [GaugeModel(0),GaugeModel(100)]
before = [m(gx).detach() for m in models]
scales = [calibrate_logit_scale(m,gdl,CPU) for m in models]
after = [m(gx).detach() for m in models]
out["logit_gauge_test"] = {
    "pre_calibration_softmax_max_abs_difference":float((before[0].softmax(1)-before[1].softmax(1)).abs().max()),
    "calibration_scales":scales,
    "post_raw_logit_std":[float(z.std()) for z in after],
    "post_class_centered_std":[float((z-z.mean(1,keepdim=True)).std()) for z in after],
    "post_softmax_max_abs_difference":float((after[0].softmax(1)-after[1].softmax(1)).abs().max()),
    "argmax_unchanged_each_model":[bool((a.argmax(1)==b.argmax(1)).all()) for a,b in zip(before,after)],
}

# Exact diagonal/low-rank nesting limit differs for zero residual variance dims.
vr = torch.tensor([0.,0.,1.,3.],dtype=DT)
out["zero_variance_shrinkage_target_difference"] = {
    "diagonal_helper":_stabilise_diagonal_variance(vr,.1,1e-6).tolist(),
    "lowrank_zero_U_limit":(.9*vr+.1*vr.mean()).clamp_min(1e-6).tolist(),
}

print(json.dumps(out,indent=2))
