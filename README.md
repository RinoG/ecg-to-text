
# ECG to Text: Medical Report Generation with Sequence-To-Sequence Models

## Abstract

This thesis explores the development of a sequence-to-sequence model for generating medical reports from electrocardiogram (ECG) data. The primary objective is to improve the interpretability of ECG signals through text generation, providing cardiologists with a more informative alternative to traditional classification models. The research is based on deep learning techniques using the ISIBrno model and the Bahdanau attention mechanism to process ECG signals and generate coherent textual summaries. The PTB-XL dataset, consisting of over 21,000 ECG recordings, serves as the basis for model training and evaluation. The methodology includes extensive data pre-processing, model architecture design, and performance evaluation using metrics such as F1 score, Jaccard score, ROUGE, and METEOR. Experimental results show that the proposed approach effectively captures the nuances of ECG signals and generates accurate and linguistically rich medical reports. This work demonstrates the potential of sequence-to-sequence models in clinical applications, paving the way for improved automated interpretation of ECG data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Results](#results)
- [License](#license)
- [References](#references)

## Introduction

The purpose of this repository is to provide the necessary code and materials to replicate the experiments conducted in the thesis. This includes data preprocessing, model training, evaluation, and generation of medical reports from ECG data.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/RinoG/ecg-to-text.git
cd ecg-to-text
pip install -r requirements.txt
```

Ensure you have the necessary datasets downloaded and placed in the appropriate directory. Instructions for downloading the PTB-XL dataset can be found [here](https://doi.org/10.13026/kfzx-aw45).


## Results

The results of the experiments, including model performance metrics and example generated reports, can be found in the `notebooks` directory.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## References

1. Classification of ECG Using Ensemble of Residual CNNs with Attention Mechanism / Petr Nejedly, Adam Ivora, Radovan Smisek et al. // Computing in Cardiology / Institute of Scientific Instruments of the Czech Academy of Sciences. — Brno, Czech Republic: 2021. — URL: https://www.cinc.org/archives/2021/pdf/CinC2021-014.pdf.
2. Wagner Patrick, Strodthoff Nils, Bousseljot Ralf et al. PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). — 2022. — URL: https://doi.org/10.13026/kfzx-aw45.
3. Machine learning-based detection of cardiovascular disease using ECG signals: performance vs. complexity / Huy Pham, Konstantin Egorov, Alexey Kazakov, Semen Budennyy // Frontiers in Cardiovascular Medicine. — 2023. — Vol. 10. — URL: https://www.frontiersin.org/articles/10.3389/fcvm.2023.1229743.

For a full list of references, please see the thesis document.
