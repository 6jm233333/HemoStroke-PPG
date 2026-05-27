# Artifact Checklist

This checklist is intended for conference artifact review and public repository release.

## Publicly Versioned

- Maintained source code under `src/`.
- Reproduction scripts under `scripts/`.
- Experiment configuration under `configs/`.
- LLM extraction prompt under `prompts/`.
- Documentation under `docs/`.
- Unit tests under `tests/`.
- Citation, license, environment, and packaging metadata at the repository root.
- No old exploratory script archive is part of the public artifact.

## Local Only

- Credentialed clinical records and waveform files.
- LLM outputs containing note-derived protected clinical content.
- Physician-reviewed timestamp tables.
- Per-patient or per-window derived feature tables.
- Packaged `.npy` arrays.
- Model checkpoints, predictions, generated figures, and generated reports.
- Manuscript source, compiled manuscript PDFs, and paper-ready figure PDFs.
- Local absolute paths and patient-level outputs.

## Review Before Publishing

- Run `pytest`.
- Run `python -m compileall src scripts tests`.
- Confirm `git status` does not include restricted data or generated outputs.
- Confirm no `*.tex` or `*.pdf` manuscript artifacts are staged.
- Confirm no local absolute paths, raw data, patient-level files, or generated outputs are staged.
- Confirm README commands match current configs.
- Confirm `docs/benchmark_results.md` matches the released study numbers.
