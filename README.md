# DI-WMF: domain-informed matched-filter initialisation

Research code accompanying Yingqi Wang's MSc dissertation, *Domain-Informed
Convolutional Neural Networks through Matched-Filter Initialisation*.

DI-WMF constructs class-dependent templates from labelled training patches,
whitens them using regularised covariance estimates, and uses them to initialise
trainable convolutional networks. The experiments distinguish the initial
predictions of the complete fitted pipeline from subsequent learning and from
the contribution of the fitted classifier.

## Repository contents

- `project/`: frozen experiment, evaluation, figure-generation and supporting
  workflow sources, including the `domain_mf` implementation and tests.
- `results/`: selected aggregate image and physiological results used in the
  dissertation; no individual physiological waveforms are included.
- `PUBLIC_CODE_MANIFEST.json`: source-file hashes and snapshot provenance.
- `PUBLIC_RESULTS_MANIFEST.json`: hashes and source aliases for the result files.
- `PUBLIC_SUPPLEMENT_MANIFEST.json`: additional preserved workflow sources
  and supporting documentation.
- `DATASETS.md`: dataset access and redistribution notes.
- `protocols/implementation_settings.tex`: recorded implementation settings.
- `release/`: complete research-material download instructions, checksums
  and file inventory.

## Download research materials

**Upload in progress:** the checkpoint attachments are not yet published.
The release link below will become available after upload and verification.

[Research materials: code, results and checkpoints](https://github.com/RRRRFGCK/DI-WMF-FYP/releases/tag/research-materials-20260911)

The Release contains per-run records, all 5,246 saved model/tensor files and
1,403 BIDMC prediction/reference NPZ files, including all 598 final patient-group
checkpoint destinations. These larger files are Release attachments rather than
Git-tracked files.

Download all six numbered ZIP parts, `release_parts.json` and
`restore_release.py` into one directory, then run:

```bash
python restore_release.py
```

The script verifies the parts and reconstructs
`DI_WMF_Research_Materials_20260911.zip`. Extract it and follow
`README_RELEASE.md`. The complete inventory is in
[`release/MANIFEST.csv`](release/MANIFEST.csv).

Raw dataset distributions are not included. The release omits 17 files that
reproduce Sign source images with unconfirmed redistribution rights, one old
thesis ZIP, and one manuscript-preparation script containing personal metadata.
See [the exact exclusion list](release/PUBLICATION_EXCLUSIONS.json).
All included scientific records and tensors preserve their original bytes.
The source snapshots were collected in September 2026; the Git history records
their publication, not the original experiment-time development history.

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

`verify_public.py` checks all files listed in the three public manifests, using
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

For a fixed version, cite the
[research-materials release](https://github.com/RRRRFGCK/DI-WMF-FYP/releases/tag/research-materials-20260911)
or the corresponding Git commit. No Zenodo DOI has been assigned.

## Rights

The release contains information from the
[BIDMC PPG and Respiration Dataset v1.0.0](https://physionet.org/content/bidmc/1.0.0/),
made available under the
[Open Data Commons Attribution License v1.0](https://physionet.org/content/bidmc/view-license/1.0.0/).
Preserve this attribution and licence link with BIDMC-derived records.
No new blanket licence for source code or other third-party data is granted.
See `DATASETS.md` for dataset sources and terms.
