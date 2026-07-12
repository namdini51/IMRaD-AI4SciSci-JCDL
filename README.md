# IMRaD Project (JCDL 2026)

This repository contains the code and final dataset used to reproduce the section-level IMRaD classification and article-level layout reconstruction experiments.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── PMCOA_samples_01234_overlap_clean_split_headers.tsv
│   └── data_summary.ipynb
└── scripts/
    ├── data.py
    ├── evaluate.py
    ├── 01_generate_oof_probabilities.ipynb
    ├── 02_metalearner.ipynb
    ├── 03_global_section_order_prior.ipynb
    └── 04_section_level_bootstrap_ci.ipynb
```

The `results/` and `cache/` directories are created automatically when the notebooks are run.

## Setup

Clone the repository, change into its root directory, and install the dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
jupyter lab
```

Run Jupyter from the repository root so that all relative paths resolve correctly.

Tested with Python 3.12.13. The exact package versions are listed in `requirements.txt`.

The embedding notebook automatically uses a CUDA GPU when available and otherwise falls back to CPU.

## Dataset

The full processed dataset is available through an anonymous Zenodo review link:

[Download the full dataset][dataset-link]

The downloaded file is:

```text
PMCOA_samples_01234_overlap_clean_split_headers.zip
```

Place it in the `data/` directory and extract the zip file.

The file contains the final article-level train, validation, and test assignments used in the experiments.

`data/data_summary.ipynb` provides optional dataset validation and descriptive summaries. Earlier exploratory cleanup scripts are not included.

## Reproduction workflow

### 1. Generate component probabilities

Run:

```text
scripts/01_generate_oof_probabilities.ipynb
```

Use the final settings:

```python
K = 5
ENCODER_SHORT_NAME = "pubmedbert"
RUN_PROGRESSION = True
RUN_TEXT = True
RUN_HEADER = True
```

Run the notebook twice:

```python
RUN_MODE = "validation"
```

and then:

```python
RUN_MODE = "test"
```

Validation mode creates OOF predictions for training and predictions for validation. Test mode creates OOF predictions for train+validation and predictions for the held-out test set.

### 2. Train the local meta-classifier

Run:

```text
scripts/02_metalearner.ipynb
```

For each model variant, run validation first and test second:

```python
COMPONENTS = ["text"]
COMPONENTS = ["text", "progression"]
COMPONENTS = ["text", "progression", "header"]  # auxiliary analysis
```

The progression-only baseline is evaluated directly in Notebook 01 and should not be rerun as a progression-only meta-classifier.

### 3. Apply the global layout prior

Run:

```text
scripts/03_global_section_order_prior.ipynb
```

Use the same `COMPONENTS`, `K`, encoder, and fusion settings used in Notebook 02.

For each model variant:

1. Run validation mode to select the prior weight.
2. Run test mode to load the validation-selected weight and evaluate the final decoder.

Keep:

```python
SELECTED_PRIOR_WEIGHT = None
```

unless intentionally overriding the selected value.

### 4. Compute section-level bootstrap intervals

Run:

```text
scripts/04_section_level_bootstrap_ci.ipynb
```

Set `MODEL_RUN_NAME` to the decoder output being evaluated. The notebook computes article-resampled confidence intervals for section-level metrics.

## Main outputs

Generated files are saved under:

```text
results/oof_components/
results/validation/local_meta/
results/validation/global_layout_prior/
results/test/local_meta/
results/test/global_layout_prior/
```

The progression-only section-level test metrics are saved at:

```text
results/oof_components/test/progression/k5_logreg/test_metrics.json
```

The global decoder saves section-level metrics, article-level layout metrics, selected prior weights, predictions, and bootstrap summaries in the corresponding model directory.

## Notes

- Run the notebooks in numerical order.
- Validation runs must be completed before the corresponding test decoder runs.
- Generated embedding caches are stored under `cache/`.
- Small numerical differences may occur across hardware or software versions.

[dataset-link]: https://zenodo.org/records/21328518?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc4Mzg5OTY4OCwiZXhwIjoxNzkwODEyNzk5fQ.eyJpZCI6IjI1Mzc1YTc4LTVkNjUtNDQ5NS1hYjA2LTY2MzhjNjc2ZTNkZSIsImRhdGEiOnt9LCJyYW5kb20iOiI5YWE1ZjEyMzQ5YzY1Njk0NDJmNGVjMTA0OTFkMmNmOCJ9.NIPZwjRygmji56cLbbWa2tk6lVg7xrlEP3rlerH4g6VBH4Vq3Unj2mjqGrXhRcjxs-Am4fzgdb5kW1MCGgsLfQ
