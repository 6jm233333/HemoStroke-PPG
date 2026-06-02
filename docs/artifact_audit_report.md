# Artifact Audit Report

## Summary Verdict

Mostly ready for ICDM Applied Track artifact submission.

The repository is now suitable for code-path inspection, environment setup, configuration review, unit testing, and full numerical reproduction after authorized access to the restricted MIMIC-III and MC-MED data. The public repository alone cannot regenerate all paper tables because raw clinical records, waveforms, reviewed onset anchors, derived arrays, checkpoints, predictions, and patient-level outputs are intentionally not redistributed.

## Files Changed

- `README.md`: clarified that this is the accompanying artifact/code repository, removed premature citation wording, and linked the 17-feature dictionary.
- `CITATION.cff`: kept repository-level citation metadata only and removed any implication of a published paper citation.
- `data/README.md`: fixed the placeholder code block and reinforced local-only data handling.
- `docs/artifact_checklist.md`: clarified public-vs-local artifact boundaries and added checks for local paths and generated outputs.
- `docs/feature_dictionary.md`: renamed the feature dictionary to the expected public documentation path and replaced non-ASCII formula notation with ASCII-safe notation.
- `configs/feature_extraction.yaml`: aligned the default 240-minute label boundaries with the tested buffer and lead-time semantics.
- `src/models/train.py`: changed frozen MC-MED external testing to use fold-trained checkpoints by default rather than a full-internal retrain path.
- `src/datasets/build_main_horizon_sets.py`: added the standard entry point for paper-aligned MIMIC train/validation/test and frozen MC-MED test arrays.
- `src/models/train.py` and `src/models/evaluate.py`: apply the MIMIC-validation operating threshold instead of implicit `argmax`.
- `scripts/reproduce/table4_false_alert_burden.py`: added the Table IV false-alert burden entry point with the documented `ID+` rule.
- `src/analysis/subgroup_statistics.py`: removed star-style significance labels and changed them to p-value labels for optional reports.
- `src/datasets/build_subgroup_sets.py`: removed the independent per-file transform so subgroup packaging consumes values from the same frozen upstream feature pipeline.
- `src/features/extract_ppg_features.py`: replaced a non-English compatibility column literal with an ASCII Unicode-escape constant.
- `docs/artifact_audit_report.md`: added this audit report.

## Paper Alignment Checklist

- Title: aligned with "In-Hospital Stroke Prediction from PPG-Derived Hemodynamic Features".
- Datasets: aligned with MIMIC-III as internal development/patient-level validation and MC-MED as frozen external evaluation.
- Cohort sizes: benchmark documentation reports 176 MIMIC-III stroke patients and 154 MC-MED stroke patients.
- Horizons: configs and docs use 240, 300, and 360 minutes, corresponding to 4 h, 5 h, and 6 h.
- Feature set: `configs/feature_set_17.json` and `docs/feature_dictionary.md` document the 17 PPG-derived hemodynamic features.
- Models: ResNet-1D is the main model and LSTM is the baseline; tests verify both accept `[batch, time, 17]` inputs.
- Patient-level split: configs use group/patient-level five-fold splitting; tests verify group disjointness and patient-window leakage prevention.
- Frozen MC-MED testing: training config disables full-internal retraining, and `src/models/train.py` now evaluates external data using fold-trained frozen checkpoints by default.
- Main Table III F1 values: `docs/benchmark_results.md` reports the paper F1 values for ResNet-1D and LSTM across 240/300/360 minutes.
- Robustness analyses: docs and scripts cover false-alert burden, relative-feature ablation, onset-anchor perturbation at -60/-30/-15/0/15/30/60 minutes, signal-quality/retention audit, SHAP interpretability, and descriptive subgroup heterogeneity.
- Subgroup analysis: the subgroup figure script no longer draws star markers or bracket annotations; subgroup results are treated as exploratory descriptive analysis.

## Privacy Checklist

- No raw MIMIC-III or MC-MED records are committed.
- No clinical note text, generated LLM chunks, physician-reviewed onset anchors, raw waveforms, derived arrays, checkpoints, predictions, generated figures, or patient-level outputs are committed.
- `data/` and `outputs/` contain README placeholders only.
- `.gitignore` excludes raw/interim/processed data, outputs, arrays, serialized models, checkpoints, generated CSV/JSON/PDF/PNG artifacts, and common experiment outputs.
- Schema column names such as patient or visit identifier fields remain in code/config only as expected input schema names; no real values are committed.
- Fixed-string searches for Windows drive-root local paths returned no committed-path hits.

## Commands Run

- `python -m compileall src scripts tests`: passed.
- `pytest`: passed with 17 tests passed and 2 Torch-dependent tests skipped because Torch was unavailable in the audit environment.
- `git status --short`: not run successfully because `git` is not installed in the current shell environment.
- `rg` equivalent searches for stale project/venue wording, old folder references, figure-number conflicts, retrain defaults, and local drive-root patterns: passed after the fixes above.
- File-system inspection of `data/` and `outputs/`: passed; only README placeholders are present.
- Recursive search for generated model/data artifacts: passed; no arrays, serialized model files, checkpoints, or named prediction artifacts were found in the repository tree.
- Recursive search for non-English public comments: passed after converting comments and escaping the compatibility column literal.

## Remaining Limitations

- Full numerical reproduction cannot be verified without credentialed MIMIC-III and MC-MED data, reviewed onset-anchor files, and the same local preprocessing artifacts used by the study.
- The benchmark tables were checked for consistency with the provided paper facts, but they were not recomputed from raw data in this public artifact audit.
- End-to-end training and MC-MED external evaluation were not executed because the restricted arrays and checkpoints are intentionally not present.
- Generated external metrics, predictions, confusion matrices, robustness summaries, and figures should remain local-only under ignored output directories.
