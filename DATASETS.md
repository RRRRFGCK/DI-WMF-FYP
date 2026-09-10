# Dataset access

Raw datasets, processed physiological reference signals and individual-level
physiological predictions are not distributed in this repository. Aggregate
result tables do not replace the original datasets required by the trainers.

## BIDMC PPG and Respiration Dataset

The study uses version 1.0.0: 53 records grouped by 46 patient identities in
the MATLAB metadata.

- Official source: https://physionet.org/content/bidmc/1.0.0/
- Version-specific terms (Open Data Commons Attribution License v1.0):
  https://physionet.org/content/bidmc/view-license/1.0.0/

For dataset-dependent replay, obtain the original MATLAB distribution and
configure `data_bidmc/bidmc_data.mat` in your project working copy. Retain the
dataset's required attribution.

## CapnoBase

- Dataset identifier: https://doi.org/10.5683/SP2/NLB8IT

The source uses custom terms of use, not a CC-BY licence. Consult the provider's
current terms when obtaining the data. No raw CapnoBase signals or derived
reference waveforms are included in this repository.

## Image datasets

Obtain Fashion-MNIST and CIFAR-10 from their original distributions. The Sign
source used in the dissertation does not state an explicit dataset licence;
its images are not redistributed here. The preserved training sources document
the dataset loading and preprocessing used by the study.

This repository does not grant additional rights over any third-party dataset.
