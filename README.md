# MultiomicsLM

MultiomicsLM provides command line and Python utilities for proteomics and metabolomics data:

- extract sample-level embeddings from protein, metabolite, or paired inputs
- impute missing values in protein or metabolite tables
- predict metabolite profiles from protein inputs
- fine-tune binary classifiers for current or future health outcomes

The package loads bundled model assets automatically.

## Installation

Large model assets are stored with Git LFS.

```bash
git lfs install
git clone https://github.com/Qiu-Shizheng/MultiomicsLM
cd MultiomicsLM
git lfs pull
pip install -e .
```

Requirements: Python 3.9 or newer, PyTorch 2.0 or newer, numpy, pandas, scipy, scikit-learn, matplotlib, and tqdm. CPU is supported with `--device cpu`; CUDA is faster for fine-tuning.

## Input Files

Inputs are CSV, TSV, or tab-delimited TXT files. Each file should contain:

- one sample identifier column, default `sample_id`
- numeric feature columns
- missing values as blank cells, `NaN`, or `NA`

Proteomics inputs should contain 2923 protein columns. Metabolomics inputs should contain 249 metabolite columns. The expected column names are provided in:

- `src/multiomicslm/assets/pretrained/protein_columns.txt`
- `src/multiomicslm/assets/pretrained/metabolite_columns.txt`

Minimal example files are provided in `examples/`.

## Command Line Examples

Inspect the model:

```bash
multiomicslm info --device cpu
```

Extract fused embeddings:

```bash
multiomicslm embed \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --output embeddings.csv \
  --mode fused \
  --batch-size 12 \
  --device cpu
```

Other embedding modes are `protein`, `metabolite`, `protein_enriched`, `metabolite_enriched`, and `all`.

Impute missing metabolite values:

```bash
multiomicslm impute \
  --metabolite examples/metabolite_input_example.csv \
  --output imputed_metabolites.csv \
  --mode metabolite \
  --batch-size 12 \
  --device cpu
```

Impute missing protein values:

```bash
multiomicslm impute \
  --protein examples/protein_input_example.csv \
  --output imputed_proteins.csv \
  --mode protein \
  --batch-size 12 \
  --device cpu
```

Predict metabolite profiles from protein values:

```bash
multiomicslm impute \
  --protein examples/protein_input_example.csv \
  --output predicted_metabolites.csv \
  --mode p2m \
  --batch-size 12 \
  --device cpu
```

Observed values are preserved in imputation outputs.

## Fine-Tuning

Direct binary labels:

```csv
sample_id,label
sample_001,0
sample_002,1
```

```bash
multiomicslm finetune \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --labels examples/binary_labels_example.csv \
  --label-col label \
  --output-dir runs/current_task \
  --modality fused \
  --finetune frozen \
  --epochs 50 \
  --batch-size 12 \
  --device cpu
```

Current or future labels from dates:

```csv
sample_id,baseline_date,diagnosis_date
sample_001,2020-01-10,
sample_002,2020-01-10,2022-06-05
sample_003,2020-01-10,2019-03-12
```

For `--task baseline`, samples diagnosed on or before baseline are positive and samples never diagnosed are negative; later diagnoses are excluded. For `--task future`, later diagnoses are positive and samples never diagnosed are negative; baseline positives are excluded.

```bash
multiomicslm finetune \
  --protein examples/protein_input_example.csv \
  --metabolite examples/metabolite_input_example.csv \
  --labels examples/date_labels_example.csv \
  --baseline-date-col baseline_date \
  --diagnosis-date-col diagnosis_date \
  --task future \
  --output-dir runs/future_task \
  --modality fused \
  --finetune full \
  --epochs 50 \
  --batch-size 12
```

Fine-tuning writes:

```text
checkpoints/best_model.pt
metrics/training_history.csv
metrics/summary.json
predictions/predictions_train.csv
predictions/predictions_val.csv
predictions/predictions_all.csv
predictions/train_embeddings.npy
predictions/val_embeddings.npy
figures/training_curves.png
figures/roc_pr_val.png
figures/probability_val.png
```

## Output Examples

Embedding output:

```csv
sample_id,fused_embedding_0001,fused_embedding_0002,fused_embedding_0003
sample_001,0.0132,-0.0815,0.2047
sample_002,-0.0441,0.0973,0.1186
```

Imputation output:

```csv
sample_id,Total Cholesterol,Total Cholesterol Minus HDL-C,HDL Cholesterol
sample_001,4.53,3.21,1.32
sample_002,5.02,3.87,1.15
```

Prediction output:

```csv
sample_id,split,y_true,pred_prob,pred_label
sample_001,train,0,0.184,0
sample_002,val,1,0.762,1
```

## Python API

```python
from multiomicslm import load_model

model = load_model(device="cpu")

data = model.load_arrays(
    protein_path="examples/protein_input_example.csv",
    metabolite_path="examples/metabolite_input_example.csv",
    id_col="sample_id",
)

embeddings = model.extract_embeddings_array(
    protein_values=data["protein_values"],
    protein_mask=data["protein_mask"],
    metabolite_values=data["metabolite_values"],
    metabolite_mask=data["metabolite_mask"],
    mode="fused",
    batch_size=12,
)

imputed_metabolites = model.impute_metabolites_array(
    data["metabolite_values"],
    data["metabolite_mask"],
    batch_size=12,
    n_iters=3,
)
```

## License

MIT
