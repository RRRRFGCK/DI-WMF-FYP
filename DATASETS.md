# Dataset sources and attribution

Raw dataset distributions are not included. Obtain them from the original
providers for training or dataset-dependent replay.

## BIDMC

This release contains predictions and processed reference signals derived from
the **BIDMC PPG and Respiration Dataset, version 1.0.0**, with 53 records grouped
into 46 patient identities.

- Dataset: https://physionet.org/content/bidmc/1.0.0/
- DOI: https://doi.org/10.13026/C2208R
- Licence: https://physionet.org/content/bidmc/view-license/1.0.0/

Contains information from the BIDMC PPG and Respiration Dataset which is made
available under the Open Data Commons Attribution License v1.0. BIDMC-derived
records are distributed subject to that licence; preserve the dataset
attribution and licence URI when redistributing them.

The records use the dataset's existing identifiers; no re-identification was
performed. Evaluation programs expect a separately obtained
`data_bidmc/bidmc_data.mat` in the working project.

## CapnoBase

Dataset: https://doi.org/10.5683/SP2/NLB8IT

The original CapnoBase signals are not included. Saved official version 1.1
metadata is included as `protocols/capnobase_metadata.json`, with the custom
terms of use and citation requirements. These terms are not a CC-BY licence.
Pretrained model checkpoints are included; no CapnoBase reference-signal NPZ
files were found in the research package.

## Image datasets

Fashion-MNIST, CIFAR-10 and Sign must be obtained separately using the sources
described in the dissertation and project documentation. The Sign source
repository has no explicit redistribution licence in the recorded materials;
its standalone source thumbnail and figures reproducing its images are omitted
from this public release. The statistical results and model tensors remain
included.

No additional rights over third-party datasets or source material are granted
by this publication.
