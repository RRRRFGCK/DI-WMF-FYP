# Code, original records and review checks

This is the submission audit snapshot dated 6 September 2026. Start with
`RESPONSE_TO_REVIEWER.md`. Current source is supplied alongside original
experimental records; it is not an authenticated run-date source commit.

## Archive layout

Extract `FYP_Code_Records_Audit.zip`. It creates `FYP_Code_Records_Audit/`.
On Windows use a short extraction directory (for example a folder directly
under your user directory), because original run names are deliberately long.
The smaller review archive contains experiment drivers, the `domain_mf/`
package, tests, original histories/configurations/final metrics, evaluation
rows, saved physiological predictions, statistical tables, current thesis
figure assets, new audits, all reported regression checkpoints and 40 image
checkpoints used for the alignment audit.

`FYP_Checkpoint_Archive.zip` holds the remaining `.pt` weight/filter files.
Extract it beside the first archive to merge into the SAME directory. It is
not required to recompute the CSV-based checks. It is required for the full
saved-filter drift audit and some original checkpoint evaluations.

`PACKAGE_MANIFEST.csv` lists the relative path, archive membership, byte size,
SHA-256 and source modification time of each payload file. Timestamps are
filesystem metadata, not proof of preregistration. The manifest excludes itself
to avoid a circular hash. The original teacher ZIP is retained under
`provenance/original_teacher_FYP3.zip`.

Historical `outputs*/` and `*_results/` paths are preserved. They include
exploratory and smoke runs as an archive, not as additional formal evidence.
Use the thesis tier and audit registry to identify reported runs. Original
unqualified AULC fields can include Epoch 0. In particular, the original
`analyze_cross_architecture_post_aulc.py` and its original result are retained
to document the selector error; use the final driver below for the corrected
publication comparison. Historical reports are not substitutes for the current
thesis or review response.

## Environment

The successful checkpoint replay used the existing CL2 environment: Python
3.11, PyTorch 2.8.0+cu128 and torchvision 0.23.0 with CUDA on the recorded GPU.
`ENVIRONMENT_REVIEW.json` describes the installed review-time environment.
The original recorded environment and its provenance limits remain in Appendix A.

`requirements-review.txt` lists Python dependencies for these scripts. Use an
appropriate existing scientific environment or create one separately; do not
replace an active environment merely to open the archive. The tabular audits
need NumPy/SciPy, while tensor replay needs PyTorch/torchvision. Figure scripts
also need Matplotlib and Pillow. The scripts below never start model training.

## Step 1: verify the delivered files

Open a terminal in `FYP_Code_Records_Audit/`:

```text
python verify_package.py
```

Without the companion archive, remaining weight files are reported as not yet
extracted. After extracting both archives, require all entries:

```text
python verify_package.py --require-all
```

## Step 2: reproduce numerical audits in a NEW directory

```text
python review_audit/aulc/verify_aulc_from_histories.py --root . --output audit_replay/aulc
python review_audit/aulc/analyze_cross_architecture_post_aulc_final.py --root . --output audit_replay/architecture_final
python review_audit/regression/verify_regression_records.py --root . --out audit_replay/regression --check-spectral
python review_audit/anchor/verify_anchor_alignment.py --repo-root . --output audit_replay/anchor
```

Expected checks: 319 AULC histories, 175 stored post-update comparisons, 1,804
historical Holm values, 689 regression folds and 3,200 fixed channel references.
The architecture driver selects low-rank rank 16 explicitly. The original
902-row file is checked as a historical mixed-metric registry, not relabelled
as 902 post-update tests. Rounded historical t-critical constants may produce
small differences from exact SciPy intervals; the audit states its tolerances.

## Step 3: optionally replay checkpoint inference

Raw datasets are not included. For regression, obtain the same BIDMC MATLAB
release and place it at `data_bidmc/bidmc_data.mat` beneath this root. Dataset
references are in the thesis; do not use a different processed array as if it
were the original input. Then:

```text
python review_audit/regression/verify_regression_records.py --root . --out audit_replay/regression_cuda --check-spectral --replay-baseline-checkpoints --replay-device cuda
```

For image alignment, supply your Fashion-MNIST/CIFAR-10 data root and the
original processed Sign root (containing `train_10/` and `test_10/`):

```text
python review_audit/anchor/verify_anchor_alignment.py --repo-root . --output audit_replay/anchor_cuda --replay --device cuda --data-root PATH_TO_IMAGE_DATA --sign-root PATH_TO_PROCESSED_SIGN
```

The Sign source has no explicit data licence; the images are not redistributed.
Archived configurations retain original local paths for traceability. Override
them for replay rather than editing the preserved records. CUDA/backend and
batching differences can affect near ties in top-five responses. The canonical
successful replay used the recorded batch sizes and original CL2/CUDA runtime.
Earlier CPU/alternative-backend diagnostics are preserved and labelled in their
audit folders; they do not replace the canonical exact replays.

## Step 4: regenerate the four revised source figures

These scripts use supplied frozen numerical inputs and illustration assets;
they do not need raw datasets or checkpoints:

```text
python review_audit/figures/redraw_teacher_figures.py --figure-dir regenerated_figures
python review_audit/figures/redraw_cross_architecture.py --figure-dir regenerated_figures
```

Original figure generators and experiment drivers are also supplied at the
package root. Some original sample-level generators require raw data and their
original directory layout. Full training is optional and separate from these
checks; `run_experiment.py --help`, `run_correncoder_regression.py --help` and
the preserved run configs document the available launch parameters.

## Provenance limits

No original per-run source commit/hash was recovered. The earliest pretrained-MSE
regression group has no later-format experiment-config JSON. Its histories,
checkpoints, fold outputs and CapnoBase metadata remain available, and its 53
checkpoint predictions were replayed exactly. We have not fabricated missing
historical files. The 902 families and new review checks are retrospective.
Per-record physiology normalisation, shared LOSO training subjects, and
arbitrary Kaiming reference indices retain the limitations stated in the thesis.
