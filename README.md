<p align="center">
  <img src="https://github.com/user-attachments/assets/766f446c-c17c-44d9-bee6-02be56eaec12" alt="Modern neural networks for small tabular datasets"/>
</p>

An evaluation framework from the paper **[Modern Neural Networks for Small Tabular Datasets: The New Default for Field-Scale Digital Soil Mapping?](https://doi.org/10.1111/ejss.70299)**

This repository provides a unified framework for evaluating modern deep neural networks on small tabular datasets, evaluated on 31 field- and farm-scale digital soil mapping datasets from [LimeSoDa](https://github.com/a11to1n3/LimeSoDa).

## Overview

- **Datasets**: Uses soil datasets from the [LimeSoDa](https://github.com/a11to1n3/LimeSoDa) repository with proximal soil sensing and remote sensing features.

- **Models**: Implements 15+ models with a unified interface:
  - **Classical ML**: Linear Regression, Ridge, Lasso, PLSR, Random Forest, XGBoost
  - **MLP-based NNs**: MLP, TabM, RealMLP
  - **Retrieval-based NNs**: TabR, ModernNCA
  - **Attention-based NNs**: AutoInt, FT-Transformer, ExcelFormer, T2G-Former, AMFormer
  - **In-context learning foundation models**: TabPFN

- **Configuration**: Experiment settings defined via YAML configuration files. Configuration files for datasets with feature-to-sample ratio < 1 are in the [config/pss/](config/pss/) folder, while configurations for high-dimensional datasets with ratio > 1 (including MIR/NIR spectroscopy features) are in the [config/spectroscopic/](config/spectroscopic/) folder.
- **Preprocessing**: Built-in support for PCA, feature scaling, numerical embeddings

## Setup

Requirements: Python 3.10+

```bash
pip install -r requirements.txt
```

## Usage

Run experiments using YAML configuration files:

```bash
python benchmark.py --config config/pss/limesoda_mlp.yaml
```

Example configuration files are provided in [config/pss/](config/pss/) and [config/spectroscopic/](config/spectroscopic/) folders.

## Results & Data

Complete experimental results, including optimized hyperparameters for all dataset-model combinations and model predictions, are available: [results.tar.gz](https://github.com/slavabarkov/smalltabnets/releases/download/v1.0/results.tar.gz)

## Citation

```bibtex
@article{barkov2026modern,
  title   = {Modern neural networks for small tabular datasets: {The} new default for field-scale {Digital} {Soil} {Mapping}?},
  author  = {Barkov, Viacheslav and Schmidinger, Jonas and Gebbers, Robin and Atzmueller, Martin},
  journal = {European Journal of Soil Science},
  volume  = {77},
  year    = {2026},
  pages   = {e70299},
  number  = {2},
  doi     = {10.1111/ejss.70299},
}
```
