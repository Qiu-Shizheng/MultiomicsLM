# MultiomicsLM

<p align="center">
  <b>MultiomicsLM</b> is a pretrained multi-omics foundation model for proteomics and metabolomics.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c">
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

## Installation

### Install from local source

```bash
git clone https://github.com/YOUR_USERNAME/MultiomicsLM.git
cd MultiomicsLM
pip install -e .
```

---

## Missing value imputation
### Impute missing metabolomics values
Observed values are preserved. Missing values are predicted by MultiomicsLM.
```bash
multiomicslm impute \
  --metabolite examples/metabolite_input_example.csv \
  --mode metabolite \
  --output outputs/metabolite_imputed.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12 \
  --n-iters 3
```

Output:
```bash
outputs/metabolite_imputed.csv
```


### Impute missing proteomics values
```bash
multiomicslm impute \
  --protein examples/protein_input_example.csv \
  --mode protein \
  --output outputs/protein_imputed.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12 \
  --n-iters 3
```

### Predict metabolites from proteomics
This mode performs protein-to-metabolite prediction.
```bash
multiomicslm impute \
  --protein examples/protein_input_example.csv \
  --mode p2m \
  --output outputs/predicted_metabolites_from_proteins.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12
```



---

## Extract individual embeddings
### Protein-only embeddings

```bash
multiomicslm embed \
  --protein examples/protein_input_example.csv \
  --mode protein \
  --output outputs/protein_embeddings.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12
```

### Metabolite-only embeddings
```bash
multiomicslm embed \
  --metabolite examples/metabolite_input_example.csv \
  --mode metabolite \
  --output outputs/metabolite_embeddings.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12
```

### Fused multi-omics embeddings
```bash
multiomicslm embed \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --mode fused \
  --output outputs/fused_embeddings.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12
```

### Extract all embedding types
```bash
multiomicslm embed \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --mode all \
  --output outputs/all_embeddings.csv \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0
```


---

## Fine-tuning for disease prediction
### Binary label fine-tuning
```bash
multiomicslm finetune \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --labels examples/binary_labels_example.csv \
  --label-col label \
  --modality fused \
  --finetune frozen \
  --output-dir outputs/finetune_binary_fused \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --batch-size 12 \
  --epochs 50
```

### Current disease diagnosis using diagnosis dates

```bash
multiomicslm finetune \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --labels examples/date_labels_example.csv \
  --baseline-date-col baseline_date \
  --diagnosis-date-col diagnosis_date \
  --task baseline \
  --modality fused \
  --finetune frozen \
  --output-dir outputs/current_disease_prediction \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0
```

### Future disease prediction using diagnosis dates
```bash
multiomicslm finetune \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --labels examples/date_labels_example.csv \
  --baseline-date-col baseline_date \
  --diagnosis-date-col diagnosis_date \
  --task future \
  --modality fused \
  --finetune full \
  --output-dir outputs/future_disease_prediction \
  --pretrained-dir src/multiomicslm/assets/pretrained \
  --device cuda:0 \
  --epochs 80 \
  --lr-head 1e-4 \
  --lr-encoder 1e-6
```



### Fine-tuning outputs
Each fine-tuning run generates:

```bash

output_dir/
├── checkpoints/
│   └── best_model.pt
├── figures/
│   ├── training_curves.png
│   ├── training_curves.pdf
│   ├── roc_pr_val.png
│   ├── roc_pr_val.pdf
│   ├── probability_val.png
│   └── probability_val.pdf
├── metrics/
│   ├── summary.json
│   └── training_history.csv
└── predictions/
    ├── predictions_train.csv
    ├── predictions_val.csv
    ├── predictions_all.csv
    ├── train_embeddings.npy
    └── val_embeddings.npy

```

## Important notes
The proteomics and metabolomics columns must follow the same order as the pretrained model. \
Metabolite values are internally processed using negative-to-missing conversion, log1p transform, and z-score normalization. \
Protein values are internally processed using z-score normalization. \
Missing values are zeroed after normalization, consistent with pretraining. 

---

## Citation
If you use MultiomicsLM in your work, please cite the corresponding paper: **MultiomicsLM: A foundation model for characterizing the multi-omics landscape of the general population**.


---

## Contact

Maintainer:

**Qiu Shizheng**  
Email: **qiushizheng@hit.edu.cn**






