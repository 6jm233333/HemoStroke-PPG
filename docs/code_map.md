# Code Map

This file maps the study workflow to maintained public code. The entry points below should be used for reproduction and review.

| Study stage | Maintained public code |
|---|---|
| Note screening and timestamp extraction | `src/data/mimic/*`, `prompts/stroke_timestamp_extraction.md` |
| Waveform anchoring | `src/data/mimic/anchor_waveforms_to_notes.py`, `src/data/mcmed/filter_prewarning_segments.py` |
| PPG feature extraction | `src/features/extract_ppg_features.py` |
| Feature cleaning and engineering | `src/features/clean_feature_table.py`, `src/features/engineer_features.py`, `src/features/select_features.py` |
| Temporal labeling | `src/labels/relabel_time_windows.py` |
| Numpy packaging | `src/datasets/build_external_eval_set.py`, `src/datasets/build_subgroup_sets.py` |
| ResNet-1D training and evaluation | `src/models/train.py`, `src/models/evaluate.py`, `src/models/resnet1d.py` |
| LSTM baseline | `src/models/lstm.py`, `configs/lstm_baseline.yaml` |
| SHAP and temporal trajectories | `src/explain/shap_analysis.py`, `src/explain/plot_feature_trajectories.py` |
| ROC, subgroup, and robustness analysis | `src/analysis/*`, `src/models/sensitivity.py`, `scripts/reproduce/*` |

New experiments should use `src/`, `scripts/`, and `configs/`. The public repository intentionally excludes restricted clinical artifacts, generated arrays, checkpoints, predictions, and manuscript files.
