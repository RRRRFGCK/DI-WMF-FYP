# DI-WMF research materials

Code, saved experimental results and trained checkpoints for Yingqi Wang's MSc
dissertation on domain-informed matched-filter initialisation.

Published release:
https://github.com/RRRRFGCK/DI-WMF-FYP/releases/tag/research-materials-20260911

## Download and open

Download all six numbered ZIP parts, `release_parts.json` and
`restore_release.py` from the release into one directory. Run:

```text
python restore_release.py
```

The script checks each part and reconstructs
`DI_WMF_Research_Materials_20260911.zip`. Extract that ZIP into a short path;
it creates `FYP_Research_Release/`.

## Contents

| Directory or file | Contents |
| --- | --- |
| `project/` | Training and evaluation code, per-run configurations, histories, metrics, model tensors and analysis outputs |
| `results/main/` | Final ten-seed image comparison |
| `results/fitted_head/` | Fitted-classifier control results |
| `results/first_layer/` | Shared-downstream first-layer comparisons |
| `results/physiology/` | Patient-level physiological summaries and paired comparisons |
| `protocols/` | Implementation settings and dataset access notes |
| `ENVIRONMENT.md` | Recorded software and hardware |
| `MANIFEST.csv` | File sizes, SHA-256 checksums and source mapping |
| `PUBLICATION_EXCLUSIONS.json` | The 19 files withheld from the original local package |

All 5,246 model/tensor files and 1,403 BIDMC prediction/reference NPZ files from
the local research package are included without changes. The 598 final
patient-group checkpoint paths are present. Per-run CSV and JSON records are
also unchanged.

The project tree includes earlier runs and intermediate analyses. For the
dissertation's final comparisons, start with the named `results/` directories
above and the recorded analysis scopes, rather than treating every historical
output as a final result.

## Check integrity

From the extracted `FYP_Research_Release/` directory:

```text
python verify_release.py --root .
python reproduce.py --list
```

Verification reads every file listed in the manifest. The second command lists
analysis entry points; it does not start training. Use a separate writable copy
of `project/` to rerun an analysis, for example:

```text
python reproduce.py image-metrics --work-root PATH_TO_WORKING_COPY
python reproduce.py image-metrics --work-root PATH_TO_WORKING_COPY --execute
```

Some saved configurations contain paths from the original computer. Set data
paths for the new installation and use the dependencies in `ENVIRONMENT.md`.
Dataset-dependent training and inference require separately downloaded datasets.

## Data and publication scope

The package contains processed BIDMC reference signals and model predictions,
not the original dataset distributions. These records contain information from
the BIDMC PPG and Respiration Dataset v1.0.0, available under the Open Data Commons
Attribution License v1.0:

- Dataset: https://physionet.org/content/bidmc/1.0.0/
- Licence: https://physionet.org/content/bidmc/view-license/1.0.0/

Keep this attribution and the licence link with redistributed BIDMC-derived
records. Other dataset sources and terms are documented in `protocols/DATASETS.md`.

The public package omits 17 image/figure files containing Sign source images
whose redistribution licence has not been established, one historical thesis
ZIP, and one manuscript-preparation script containing personal metadata.
They remain unchanged in the author's local archive. Raw datasets are not
bundled, and no new blanket licence for third-party material is granted.

The scientific files are the preserved September 2026 snapshots. Publication
on GitHub did not retrain models or recalculate results.
