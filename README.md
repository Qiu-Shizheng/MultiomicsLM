# MultiomicsLM

<p align="center">
  <b>MultiomicsLM</b> is a pretrained multi-omics foundation model for UK Biobank-scale proteomics and metabolomics.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img src="https://img.shields.io/badge/omics-proteomics%20%7C%20metabolomics-green">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Overview

**MultiomicsLM** is a pretrained multi-modal model for individual-level representation learning from:

- proteomics
- metabolomics
- paired proteomics + metabolomics

The model was pretrained on UK Biobank proteomics and metabolomics data using a multi-modal masked autoencoder objective, cross-modal protein-to-metabolite prediction, contrastive alignment, and temporal metabolomics consistency.

This repository provides an easy-to-use Python package for:

1. loading pretrained MultiomicsLM weights
2. imputing missing metabolomics or proteomics values
3. predicting metabolomics profiles from proteomics
4. extracting individual-level embeddings
5. fine-tuning for current disease diagnosis or future disease prediction

---

## Contact

Maintainer:

**Qiushi Zheng**  
Email: **qiushizheng@hit.edu.cn**

---

## Installation

### Option 1: Install from local source

```bash
git clone https://github.com/YOUR_USERNAME/MultiomicsLM.git
cd MultiomicsLM
pip install -e .
