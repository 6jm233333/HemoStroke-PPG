# HemoStroke-PPG

Accompanying artifact/code repository for **In-Hospital Stroke Prediction from PPG-Derived Hemodynamic Features**.

This repository contains the reproducible code path for building anchor-aligned photoplethysmography (PPG) hemodynamic features and training stroke-warning models across MIMIC-III and MC-MED cohorts. It is a source-code repository, not a manuscript archive: paper source, compiled PDFs, and paper-ready figure files are intentionally kept outside version control.

## Scope

Included:

- Cohort mining, onset anchoring, and waveform alignment utilities.
- PPG feature extraction, feature cleaning, relative-feature engineering, and temporal relabeling.
- ResNet-1D main model and LSTM baseline.
- Patient-level split checks, evaluation, SHAP, ROC, subgroup, sensitivity, and quality-control scripts.
- Minimal tests for label logic, patient-level splitting, filter alignment, and model tensor shapes.

Not included:

- Restricted clinical records, waveforms, note-derived LLM outputs, physician-reviewed timestamps, derived patient tables, `.npy` datasets, checkpoints, predictions, manuscript source, manuscript PDFs, or paper figure PDFs.

## Main Results

| Horizon | Cohort | ResNet-1D F1 | LSTM F1 |
|---:|---|---:|---:|
| 240 min | MIMIC-III | 0.7956 +/- 0.0027 | 0.7801 +/- 0.0161 |
| 240 min | MC-MED | 0.9256 +/- 0.0211 | 0.7456 +/- 0.0520 |
| 300 min | MIMIC-III | 0.8759 +/- 0.0105 | 0.8651 +/- 0.0047 |
| 300 min | MC-MED | 0.9595 +/- 0.0151 | 0.9576 +/- 0.0275 |
| 360 min | MIMIC-III | 0.9406 +/- 0.0015 | 0.9346 +/- 0.0040 |
| 360 min | MC-MED | 0.9888 +/- 0.0025 | 0.9854 +/- 0.0088 |

Full benchmark tables are in `docs/benchmark_results.md`. The 17-feature definition is documented in `docs/feature_dictionary.md`.

## Repository Layout

```text
configs/                 Experiment and path configuration
data/                    Local-only data mount point; tracked README only
docs/                    Data access, reproducibility, results, and artifact notes 
prompts/                 LLM onset timestamp extraction prompt
scripts/qc/              Quality-control utilities
scripts/reproduce/       Table and figure reproduction entry points
src/                     Maintained implementation
tests/                   Lightweight unit tests
outputs/                 Local-only generated outputs; tracked README only
```

New work should use `src/`, `scripts/`, and `configs/`.

## Installation

```bash
conda env create -f environment.yml
conda activate hemostroke-ppg
pip install -e .
pytest
```

Without Conda:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

## Data

Because MIMIC-III and MC-MED are credentialed clinical datasets, this repository cannot provide raw clinical records, waveforms, note-derived timestamps, derived feature arrays, checkpoints, or predictions. Full numerical reproduction requires authorized local access to the required datasets and placement of the files under the expected local directory structure. Without restricted data, users can still inspect the code path, configuration files, feature definitions, model implementation, split checks, and unit tests.

Full reproduction requires credentialed local access to:

- MIMIC-III Clinical Database.
- MIMIC-III Waveform Database Matched Subset.
- MC-MED waveform/EHR resources.

Expected local layout:

```text
data/raw/mimic/tables/
data/raw/mimic/waveforms/
data/raw/mcmed/
data/interim/mimic/
data/interim/mcmed/
data/processed/mimic/
data/processed/mcmed/
```

See `docs/data_access.md` before running experiments. The `.gitignore` excludes raw data, intermediate clinical artifacts, processed datasets, checkpoints, predictions, manuscript files, and generated outputs.

## Reproduction

Run from the repository root.

```bash
# 1. Build MIMIC note candidates and LLM chunks.
python -m src.data.mimic.build_stroke_note_table --config configs/mimic_data.yaml
python -m src.data.mimic.export_llm_chunks --config configs/mimic_data.yaml

# 2. After LLM extraction and physician review, anchor waveforms to onset times.
python -m src.data.mimic.anchor_waveforms_to_notes --config configs/mimic_data.yaml
python -m src.data.mcmed.filter_prewarning_segments --config configs/mcmed_data.yaml

# 3. Extract and label PPG windows.
python -m src.features.extract_ppg_features --dataset mimic --feature-config configs/feature_extraction.yaml --data-config configs/mimic_data.yaml
python -m src.features.extract_ppg_features --dataset mcmed --feature-config configs/feature_extraction.yaml --data-config configs/mcmed_data.yaml
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mimic --output-dir data/processed/mimic/features_labeled
python -m src.labels.relabel_time_windows --config configs/feature_extraction.yaml --dataset mcmed --output-dir data/processed/mcmed/features_labeled

# 4. Train and evaluate the main model.
python -m src.models.train --config configs/training.yaml
python -m src.models.evaluate --config configs/training.yaml
```

Baseline:

```bash
python -m src.models.train --config configs/lstm_baseline.yaml
python -m src.models.evaluate --config configs/lstm_baseline.yaml
```

Figure, table, and robustness scripts:

```bash
python scripts/reproduce/figure_roc.py --help
python scripts/reproduce/figure_shap.py --help
python scripts/reproduce/figure_temporal_cases.py --help
python scripts/reproduce/figure_subgroup_f1.py --help
python scripts/reproduce/table1_mcmed_cohort_stats.py --help
python -m src.models.sensitivity --help
```

See `docs/reproducibility.md` for the staged checklist.

## Citation

Citation information will be added after acceptance or publication. For now, use `CITATION.cff` for repository-level metadata.

## Clinical Disclaimer

This code is for retrospective research reproduction. It is not a medical device, not a clinical alerting system, and must not be used for patient care without prospective validation, calibration, governance review, and local clinical oversight.
