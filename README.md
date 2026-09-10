# DI-WMF: domain-informed matched-filter initialisation

Research code accompanying Yingqi Wang's MSc dissertation, *Domain-Informed
Convolutional Neural Networks through Matched-Filter Initialisation*.

DI-WMF constructs class-dependent templates from labelled training patches,
whitens them using regularised covariance estimates, and uses them to initialise
trainable convolutional networks. The experiments distinguish the initial
predictions of the complete fitted pipeline from subsequent learning and from
the contribution of the fitted classifier.

## Repository contents

- `project/`: frozen experiment, evaluation and figure-generation sources,
  including the `domain_mf` implementation and tests.
- `results/`: selected aggregate image and physiological results used in the
  dissertation; no individual physiological waveforms are included.
- `PUBLIC_CODE_MANIFEST.json`: source-file hashes and snapshot provenance.
- `PUBLIC_RESULTS_MANIFEST.json`: hashes and source aliases for the result files.
- `DATASETS.md`: dataset access and redistribution notes.
- `protocols/implementation_settings.tex`: recorded implementation settings.

This public repository is the **code and aggregate-results component** of the
research materials. It does not include raw datasets, trained checkpoints,
per-record predictions, processed reference waveforms, or the complete 2.87 GB
local research archive. Some analysis drivers require those additional inputs.
The source snapshots were collected in September 2026; their Git publication
history is not the original experiment-time development history.

## Installation and inspection

Create a separate Python environment, then install dependencies appropriate for
your hardware:

```bash
python -m pip install -r requirements-public.txt
python verify_public.py
```

`project/requirements-review.txt` preserves the supplied review-environment
dependency list. `requirements-public.txt` additionally includes `h5py`, which
the regression code imports. This is an installation aid, not an experiment-time
lockfile. Select a compatible PyTorch/CUDA installation for GPU work.

The recorded review environment used Python 3.11.13 and PyTorch 2.8.0+cu128 on
Windows with an NVIDIA GeForce RTX 5070 Laptop GPU. A new installation is not a
guarantee of bitwise-identical GPU results.

`verify_public.py` checks all files listed in the two public manifests, using
only the Python standard library. It does not run training or inference.

## Main source entry points

Run experiment scripts from `project/`. Preserve the nested directory layout:
several stored workflows resolve their project root from their script location.

| Experiment or analysis | Path relative to `project/` |
| --- | --- |
| Image trainer and experiment matrix | `run_experiment.py`, `run_matrix.py` |
| Shared-downstream first-layer study | `output/repairs/FYP6_REPAIR_20260908/first_layer/run_shared_first_layer.py` |
| First-layer summary | `output/repairs/FYP6_REPAIR_20260908/first_layer/analyze_shared_first_layer.py` |
| Corruptions and reliability | `evaluate_corruptions.py`, `evaluate_reliability.py` |
| Template semantics and deletion tests | `evaluate_template_semantics.py`, `evaluate_faithfulness.py` |
| Computational cost | `evaluate_structured_efficiency.py`, `analyze_total_cost.py` |
| Patient-group regression | `run_correncoder_patient_loso.py` |
| Final physiological evaluation | `output/repairs/FYP6_REPAIR_20260908/evaluate_rr_patient46.py` |

The nested first-layer and regression directories preserve intermediate and
final workflow sources. Consult their command-line arguments and configuration
requirements before executing them. Historical paths or expected checkpoint and
configuration files may need to be supplied in a separate working copy.

## Result conventions

The final main image comparison uses ten paired seeds. Reported post-update
AULC excludes Epoch 0. The fitted-head training control is a distinct comparison
from the complete-initialisation comparison; its effect sizes should not be
substituted for the main registry values.

The final physiological summary uses 46 patient-identity groups covering 53
BIDMC records and fixed Epoch-80 neural endpoints. Its statistical comparisons
use the corresponding patient-level analysis and 120-contrast Holm family.
See the supplied result columns and dissertation for each endpoint and scope.

## Citation and versioning

For a fixed version, cite this repository using the full commit identifier or
GitHub's commit-specific tree URL. No Zenodo DOI has been assigned to this
repository. Do not describe the complete local checkpoint archive as publicly
available unless it has been published separately.

## Rights

No new blanket licence for source code or third-party data is granted by this
upload. See `DATASETS.md` for external dataset sources and terms. Dataset access
must be arranged through the original providers.
